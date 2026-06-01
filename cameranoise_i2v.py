import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
CAMERANOISE_ROOT = PROJECT_ROOT / "cameranoise_warping"

for path in (PROJECT_ROOT, SCRIPTS_ROOT, CAMERANOISE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_cameranoise import build_cameranoise, load_vggt_model
from caption_image_qwenvl import caption_image_qwenvl, load_qwenvl
from generate_camera_control_video import DEFAULT_NEGATIVE_PROMPT, run_camera_noise_inference


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def classify_file(path):
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix == ".npy":
        return "cameranoise"
    if suffix == ".txt" and path.name == "caption.txt":
        return "caption"
    if suffix == ".pt":
        return "camera_pose"
    return "other"


def relative_path(path, root):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def list_files(root):
    root = Path(root)
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        files.append(
            {
                "path": relative_path(path, root),
                "type": classify_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def pick_one(paths, label):
    paths = sorted(path for path in paths if path.is_file() and path.name != ".DS_Store")
    if not paths:
        return None
    if len(paths) > 1:
        joined = "\n".join(str(path) for path in paths)
        raise ValueError(f"Expected one {label}, found {len(paths)}:\n{joined}")
    return paths[0]


def parse_hw(value):
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected H,W or HxW.")
    try:
        height, width = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected integer H,W.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Height and width must be positive.")
    return [height, width]


def infer_latent_hw(height, width, vae_stride=8):
    if height % vae_stride != 0 or width % vae_stride != 0:
        raise ValueError(
            f"--height and --width must be divisible by {vae_stride} to infer CameraNoise size; "
            "or pass --cameranoise-downscale-size explicitly."
        )
    return [height // vae_stride, width // vae_stride]


def release_cuda_cache():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def camera_noise_sample_indices(camera_noise_path, frames, sample_mode):
    import numpy as np

    camera_noise = np.load(camera_noise_path, mmap_mode="r")
    if camera_noise.ndim != 4:
        raise ValueError(f"CameraNoise should have shape [T,H,W,C], got {camera_noise.shape}.")

    total_frames = camera_noise.shape[0]
    if frames <= 0:
        raise ValueError("--frames must be positive.")
    if frames > total_frames:
        raise ValueError(f"--frames={frames} exceeds CameraNoise length {total_frames}.")

    if sample_mode == "front":
        return np.arange(frames, dtype=np.int64)
    if sample_mode == "even":
        return np.linspace(0, total_frames - 1, frames).astype(np.int64)
    raise ValueError(f"Unknown sample mode: {sample_mode}")


def resize_rgb_frame(frame, height, width):
    import numpy as np
    from PIL import Image

    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    image = Image.fromarray(frame).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BICUBIC)
    return np.asarray(image)


def noise_frame_to_rgb(noise_frame):
    import numpy as np

    noise_frame = np.asarray(noise_frame, dtype=np.float32)
    if noise_frame.ndim != 3:
        raise ValueError(f"CameraNoise frame should have shape [H,W,C], got {noise_frame.shape}.")

    channels = min(noise_frame.shape[-1], 3)
    image = np.zeros((*noise_frame.shape[:2], 3), dtype=np.float32)
    image[..., :channels] = noise_frame[..., :channels]
    return np.clip(image / 3.0 + 0.5, 0.0, 1.0)


def add_video_labels(frame, labels, bar_height=32):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    height, width = frame.shape[:2]
    canvas = np.ones((height + bar_height, width, 3), dtype=np.uint8) * 255
    canvas[bar_height:] = frame

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    col_width = width // len(labels)
    for col, label in enumerate(labels):
        x0 = col * col_width
        x1 = width if col == len(labels) - 1 else (col + 1) * col_width
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = x0 + max(0, (x1 - x0 - text_width) // 2)
        y = max(0, (bar_height - text_height) // 2)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    return np.asarray(image)


def save_input_output_comparison_video(
    reference_image,
    camera_noise_path,
    generated_video,
    output_path,
    indices,
    height,
    width,
    fps,
):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    reference_image = Path(reference_image).expanduser().resolve()
    camera_noise_path = Path(camera_noise_path).expanduser().resolve()
    generated_video = Path(generated_video).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not reference_image.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    if not camera_noise_path.exists():
        raise FileNotFoundError(f"CameraNoise npy not found: {camera_noise_path}")

    camera_noise = np.load(camera_noise_path, mmap_mode="r")
    if camera_noise.ndim != 4:
        raise ValueError(f"CameraNoise should have shape [T,H,W,C], got {camera_noise.shape}.")
    if camera_noise.shape[0] <= int(indices.max()):
        raise ValueError(
            f"CameraNoise has {camera_noise.shape[0]} frames, but sampled index "
            f"{int(indices.max())} is required."
        )

    reference_frame = np.asarray(Image.open(reference_image).convert("RGB"))
    reference_frame = resize_rgb_frame(reference_frame, height, width)

    generated_reader = imageio.get_reader(str(generated_video))
    generated_frames = []
    try:
        for frame in generated_reader:
            generated_frames.append(frame)
            if len(generated_frames) >= len(indices):
                break
    finally:
        generated_reader.close()

    if len(generated_frames) < len(indices):
        raise ValueError(
            f"Generated video has {len(generated_frames)} readable frames, "
            f"but {len(indices)} comparison frames are required."
        )

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for out_index, noise_index in enumerate(tqdm(indices, desc="Saving comparison")):
            noise_frame = noise_frame_to_rgb(camera_noise[int(noise_index)])
            generated_frame = generated_frames[out_index]
            noise_frame = resize_rgb_frame((noise_frame * 255).astype(np.uint8), height, width)
            generated_frame = resize_rgb_frame(generated_frame, height, width)
            combined = np.concatenate([reference_frame, noise_frame, generated_frame], axis=1)
            writer.append_data(add_video_labels(combined, ["Reference Image", "CameraNoise", "Generated"]))

    return output_path


def discover_demo_files(demo_dir):
    inputs_dir = demo_dir / "inputs"
    conditions_dir = demo_dir / "conditions"
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Demo inputs directory not found: {inputs_dir}")

    reference_image = pick_one(
        [path for path in inputs_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS],
        "reference image in inputs/",
    )
    reference_video = pick_one(
        [path for path in inputs_dir.iterdir() if path.suffix.lower() in VIDEO_EXTS],
        "reference video in inputs/",
    )
    input_cameranoise = pick_one(
        [path for path in inputs_dir.iterdir() if path.suffix.lower() == ".npy"],
        "CameraNoise npy in inputs/",
    )

    condition_noises = list((conditions_dir / "noises").glob("*_noises.npy"))
    condition_noises += [
        path for path in conditions_dir.glob("*.npy") if path.name.endswith("_noises.npy")
    ]
    existing_cameranoise = pick_one(condition_noises, "CameraNoise npy in conditions/")
    caption_path = conditions_dir / "caption.txt"

    if reference_image is None:
        raise FileNotFoundError(f"No reference image found in {inputs_dir}")
    if existing_cameranoise is None and input_cameranoise is None and reference_video is None:
        raise FileNotFoundError(
            f"No CameraNoise npy or reference video found. Put .npy or video in {inputs_dir}"
        )

    return {
        "reference_image": reference_image,
        "reference_video": reference_video,
        "input_cameranoise": input_cameranoise,
        "existing_cameranoise": existing_cameranoise,
        "caption_path": caption_path if caption_path.exists() else None,
    }


def infer_condition_paths(conditions_dir, camera_noise_path, saved_flow=False):
    camera_noise_path = Path(camera_noise_path)
    video_id = camera_noise_path.name
    if video_id.endswith("_noises.npy"):
        video_id = video_id[: -len("_noises.npy")]
    else:
        video_id = camera_noise_path.stem

    return {
        "video_id": video_id,
        "output_root": conditions_dir,
        "noise_path": camera_noise_path,
        "visualization_path": conditions_dir / "noises" / f"{video_id}_visualization.mp4",
        "intrinsic_path": conditions_dir / "camerapose" / video_id / "intrinsic.pt",
        "extrinsic_path": conditions_dir / "camerapose" / video_id / "extrinsic.pt",
        "flow_path": conditions_dir / "flows" / f"{video_id}_flows.npy" if saved_flow else None,
    }


def write_manifest(
    demo_dir,
    *,
    discovered,
    cameranoise_result,
    caption_path,
    video_path,
    comparison_path,
    comparison_indices,
    args,
):
    demo_dir = demo_dir.expanduser().resolve()
    conditions_dir = demo_dir / "conditions"
    samples_dir = demo_dir / "samples"

    manifest = {
        "demo_id": demo_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "demo_dir": str(demo_dir),
        "inputs": {
            "reference_image": relative_path(discovered["reference_image"], demo_dir),
            "reference_video": relative_path(discovered["reference_video"], demo_dir),
            "input_cameranoise": relative_path(discovered["input_cameranoise"], demo_dir),
            "files": list_files(demo_dir / "inputs"),
        },
        "conditions": {
            "caption": relative_path(caption_path, demo_dir),
            "cameranoise": relative_path(cameranoise_result["noise_path"], demo_dir),
            "cameranoise_visualization": relative_path(cameranoise_result["visualization_path"], demo_dir),
            "intrinsic": relative_path(cameranoise_result["intrinsic_path"], demo_dir),
            "extrinsic": relative_path(cameranoise_result["extrinsic_path"], demo_dir),
            "flow": relative_path(cameranoise_result.get("flow_path"), demo_dir),
            "files": list_files(conditions_dir),
        },
        "samples": {
            "generated_video": relative_path(video_path, demo_dir),
            "comparison_video": relative_path(comparison_path, demo_dir),
            "files": list_files(samples_dir),
        },
        "params": {
            "cameranoise_config": relative_path(args.cameranoise_config or args.vggt_config, demo_dir),
            "cameranoise_std_reference_size": args.cameranoise_std_reference_size,
            "cameranoise_downscale_size": args.cameranoise_downscale_size,
            "saved_flow": args.saved_flow,
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "sample_mode": args.sample_mode,
            "comparison_indices": [int(index) for index in comparison_indices],
            "cfg": args.cfg,
            "degradation_value": args.degradation_value,
            "steps": args.steps,
            "seed": args.seed,
            "output_type": args.output_type,
            "fps": args.fps,
        },
    }

    manifest_path = demo_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one demo folder end to end. The demo folder must contain inputs/ with "
            "one reference image and either one reference video or one CameraNoise npy."
        )
    )

    parser.add_argument("--demo-dir", required=True, type=Path, help="Demo directory, for example outputs/demo1.")

    parser.add_argument("--vggt-ckpt", type=str, help="VGGT checkpoint path. Required only when CameraNoise must be generated.")
    parser.add_argument(
        "--cameranoise-config",
        type=Path,
        help="Optional CameraNoise yaml config merged on top of cameranoise_warping/configs/default.yaml.",
    )
    parser.add_argument("--vggt-config", type=Path, help="Deprecated alias of --cameranoise-config.")
    parser.add_argument("--vggt-device", default=None, help="VGGT device, for example cuda:0 or cpu.")
    parser.add_argument("--cameranoise-overwrite", action="store_true", help="Regenerate CameraNoise even if conditions/noises exists.")
    parser.add_argument("--saved-flow", action="store_true", help="Save CameraNoise flow npy.")
    parser.add_argument("--no-cameranoise-visualization", action="store_true", help="Disable CameraNoise visualization mp4.")
    parser.add_argument(
        "--cameranoise-std-reference-size",
        type=int,
        default=96,
        help="Amplitude/std reference size for CameraNoise. Default: 96.",
    )
    parser.add_argument(
        "--cameranoise-downscale-size",
        type=parse_hw,
        help=(
            "Saved CameraNoise size as H,W or HxW. "
            "Defaults to [height/8, width/8], for example 576x1024 -> 72,128."
        ),
    )

    parser.add_argument("--qwenvl-model-path", type=str, help="QwenVL model path. Required only when caption.txt is missing.")
    parser.add_argument("--qwenvl-device", default="cuda", help="QwenVL input tensor device.")
    parser.add_argument("--qwenvl-device-map", default="auto", help="QwenVL transformers device_map.")
    parser.add_argument("--qwenvl-torch-dtype", default="auto", help="QwenVL transformers torch_dtype.")
    parser.add_argument(
        "--caption-question",
        default="Please describe the content of the image in detail.",
        help="Question used to caption the reference image.",
    )
    parser.add_argument("--caption-max-new-tokens", type=int, default=128, help="QwenVL max generated tokens.")
    parser.add_argument("--overwrite-caption", action="store_true", help="Regenerate conditions/caption.txt.")

    parser.add_argument("--model-root", required=True, type=Path, help="Path to Wan2.1-I2V-14B-720P model directory.")
    parser.add_argument("--lora-path", type=Path, help="Optional CameraNoise LoRA checkpoint.")
    parser.add_argument("--height", required=True, type=int, help="Output video height.")
    parser.add_argument("--width", required=True, type=int, help="Output video width.")
    parser.add_argument("--frames", type=int, default=81, help="Number of generated frames.")
    parser.add_argument("--sample-mode", choices=["front", "even"], default="front", help="How to sample CameraNoise frames.")
    parser.add_argument("--cfg", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--degradation-value", type=float, default=None, help="Random in [0,0.6] if omitted.")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    parser.add_argument("--device", default="cuda", help="Wan pipeline device.")
    parser.add_argument("--output-type", choices=["single", "concat", "ct1"], default="single")
    parser.add_argument("--output-name", type=str, help="Optional final mp4 filename. Defaults to <demo_id>.mp4.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cameranoise_config and args.vggt_config and args.cameranoise_config != args.vggt_config:
        raise ValueError("Use only one CameraNoise config path: --cameranoise-config or --vggt-config.")
    cameranoise_config = args.cameranoise_config or args.vggt_config

    demo_dir = args.demo_dir.expanduser().resolve()
    inputs_dir = demo_dir / "inputs"
    conditions_dir = demo_dir / "conditions"
    samples_dir = demo_dir / "samples"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_demo_files(demo_dir)
    reference_image = discovered["reference_image"]

    if args.cameranoise_downscale_size is None:
        args.cameranoise_downscale_size = infer_latent_hw(args.height, args.width)

    camera_noise_path = discovered["existing_cameranoise"]
    if args.cameranoise_overwrite:
        camera_noise_path = None
    if camera_noise_path is None:
        camera_noise_path = discovered["input_cameranoise"]

    if camera_noise_path is not None and not args.cameranoise_overwrite:
        print(f"Step 1/3: using existing CameraNoise: {camera_noise_path}")
        cameranoise_result = infer_condition_paths(conditions_dir, camera_noise_path, saved_flow=args.saved_flow)
    else:
        if not args.vggt_ckpt:
            raise ValueError("--vggt-ckpt is required when CameraNoise must be generated from reference video.")
        if discovered["reference_video"] is None:
            raise FileNotFoundError("No reference video found for CameraNoise generation.")
        print("Step 1/3: loading VGGT and building CameraNoise")
        vggt_model = load_vggt_model(args.vggt_ckpt, args.vggt_device)
        cameranoise_result = build_cameranoise(
            vggt_model,
            reference_video=discovered["reference_video"],
            output_root=conditions_dir,
            device=args.vggt_device,
            config_path=cameranoise_config,
            overwrite=args.cameranoise_overwrite,
            saved_flow=args.saved_flow,
            cameranoise_visualize=not args.no_cameranoise_visualization,
            cameranoise_std_reference_size=args.cameranoise_std_reference_size,
            cameranoise_downscale_size=args.cameranoise_downscale_size,
        )
        del vggt_model
        release_cuda_cache()
        camera_noise_path = cameranoise_result["noise_path"]

    if not Path(camera_noise_path).exists():
        raise FileNotFoundError(f"CameraNoise npy was not generated or found: {camera_noise_path}")

    caption_path = conditions_dir / "caption.txt"
    if caption_path.exists() and not args.overwrite_caption:
        print(f"Step 2/3: using existing caption: {caption_path}")
        caption = caption_path.read_text(encoding="utf-8").strip()
    else:
        if not args.qwenvl_model_path:
            raise ValueError("--qwenvl-model-path is required when conditions/caption.txt is missing.")
        print("Step 2/3: loading QwenVL and captioning reference image")
        qwenvl_model, qwenvl_processor = load_qwenvl(
            args.qwenvl_model_path,
            device_map=args.qwenvl_device_map,
            torch_dtype=args.qwenvl_torch_dtype,
        )
        caption = caption_image_qwenvl(
            qwenvl_model,
            qwenvl_processor,
            reference_image=reference_image,
            output_dir=conditions_dir,
            question=args.caption_question,
            device=args.qwenvl_device,
            max_new_tokens=args.caption_max_new_tokens,
        )
        del qwenvl_model, qwenvl_processor
        release_cuda_cache()

    print("Step 3/3: generating final video")
    output_name = args.output_name or f"{demo_dir.name}.mp4"
    video_path = run_camera_noise_inference(
        reference_image=reference_image,
        cameranoise=camera_noise_path,
        caption=caption,
        output_dir=samples_dir,
        height=args.height,
        width=args.width,
        model_root=args.model_root,
        lora_path=args.lora_path,
        output_name=output_name,
        frames=args.frames,
        sample_mode=args.sample_mode,
        cfg=args.cfg,
        degradation_value=args.degradation_value,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        output_type=args.output_type,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )

    comparison_path = None
    comparison_indices = camera_noise_sample_indices(camera_noise_path, args.frames, args.sample_mode)
    comparison_name = f"{Path(video_path).stem}_compare.mp4"
    comparison_path = save_input_output_comparison_video(
        reference_image=reference_image,
        camera_noise_path=camera_noise_path,
        generated_video=video_path,
        output_path=samples_dir / comparison_name,
        indices=comparison_indices,
        height=args.height,
        width=args.width,
        fps=args.fps,
    )
    print(f"comparison_video: {comparison_path}")

    manifest_path = write_manifest(
        demo_dir,
        discovered=discovered,
        cameranoise_result=cameranoise_result,
        caption_path=caption_path,
        video_path=video_path,
        comparison_path=comparison_path,
        comparison_indices=comparison_indices,
        args=args,
    )

    print("\nJoint inference completed.")
    print(f"demo_dir: {demo_dir}")
    print(f"reference_image: {reference_image}")
    print(f"reference_video: {discovered['reference_video']}")
    print(f"cameranoise_npy: {camera_noise_path}")
    print(f"caption_txt: {caption_path}")
    print(f"generated_video: {video_path}")
    print(f"comparison_video: {comparison_path}")
    print(f"manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
