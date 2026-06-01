import argparse
import random
import time
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def image_compose_width(image, image_1):
    from PIL import Image

    width, height = image.size
    width1 = image_1.size[0]
    to_image = Image.new("RGB", (width + width1, height))
    to_image.paste(image, (0, 0))
    to_image.paste(image_1, (width, 0))
    return to_image


def read_caption(args):
    if args.caption and args.caption_path:
        raise ValueError("Use only one of --caption or --caption-path.")
    if args.caption:
        return args.caption.strip()
    if args.caption_path:
        return Path(args.caption_path).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --caption or --caption-path.")


def build_i2v_model_configs(model_root):
    from diffsynth.pipelines.wan_video_new import ModelConfig

    model_root = Path(model_root)

    return [
        ModelConfig(
            path=[
                str(model_root / "diffusion_pytorch_model-00001-of-00007.safetensors"),
                str(model_root / "diffusion_pytorch_model-00002-of-00007.safetensors"),
                str(model_root / "diffusion_pytorch_model-00003-of-00007.safetensors"),
                str(model_root / "diffusion_pytorch_model-00004-of-00007.safetensors"),
                str(model_root / "diffusion_pytorch_model-00005-of-00007.safetensors"),
                str(model_root / "diffusion_pytorch_model-00006-of-00007.safetensors"),
                str(model_root / "diffusion_pytorch_model-00007-of-00007.safetensors"),
            ],
            offload_device="cpu",
        ),
        ModelConfig(path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth"), offload_device="cpu"),
        ModelConfig(path=str(model_root / "Wan2.1_VAE.pth"), offload_device="cpu"),
        ModelConfig(path=str(model_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), offload_device="cpu"),
    ]


def load_camera_noise(camera_noise_path, frames, sample_mode):
    import numpy as np
    import torch

    camera_noise = np.load(camera_noise_path)
    if camera_noise.ndim != 4:
        raise ValueError(f"CameraNoise should have shape [F,H,W,C], got {camera_noise.shape}.")

    camera_noise = torch.tensor(camera_noise)
    camera_noise = camera_noise.permute(0, 3, 1, 2).contiguous()

    total_frames = camera_noise.shape[0]
    if frames is None:
        frames = total_frames
    if frames <= 0:
        raise ValueError("--frames must be positive.")
    if frames > total_frames:
        raise ValueError(f"--frames={frames} exceeds CameraNoise length {total_frames}.")

    if sample_mode == "front":
        indices = torch.arange(frames)
    elif sample_mode == "even":
        indices = torch.linspace(0, total_frames - 1, frames).long()
    else:
        raise ValueError(f"Unknown sample mode: {sample_mode}")

    return camera_noise[indices], indices


def save_generated_video(video, output_type, output_path, reference_image_path, caption, fps):
    from PIL import Image
    from diffsynth import save_video
    from utils.save_video import ct1_save_video

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_type == "single":
        save_video(list(video), str(output_path), fps=fps, quality=5)
        return

    if output_type == "concat":
        ref_image = Image.open(reference_image_path).convert("RGB")
        video_out = []
        for frame in video:
            ref = ref_image.resize(frame.size)
            video_out.append(image_compose_width(ref, frame))
        save_video(video_out, str(output_path), fps=fps, quality=5)
        return

    if output_type == "ct1":
        ct1_save_video(
            list(video),
            str(output_path),
            fps=fps,
            ref_image=str(reference_image_path),
            caption=caption,
            bar_height=60,
            font_size=32,
            quality=5,
        )
        return

    raise ValueError(f"Unknown output type: {output_type}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-sample Wan I2V inference with reference image, CameraNoise, and caption."
    )
    parser.add_argument("--model-root", required=True, type=Path, help="Path to Wan2.1-I2V-14B-720P model directory.")
    parser.add_argument("--lora-path", type=Path, help="Optional CameraNoise LoRA checkpoint.")
    parser.add_argument("--reference-image", required=True, type=Path, help="Input reference image.")
    parser.add_argument("--cameranoise", required=True, type=Path, help="CameraNoise npy path with shape [F,H,W,C].")
    parser.add_argument("--caption", type=str, help="Prompt text.")
    parser.add_argument("--caption-path", type=Path, help="Path to caption.txt.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save generated video.")
    parser.add_argument("--output-name", type=str, help="Optional output mp4 filename.")
    parser.add_argument("--height", required=True, type=int, help="Output video height.")
    parser.add_argument("--width", required=True, type=int, help="Output video width.")
    parser.add_argument("--frames", type=int, default=81, help="Number of output frames.")
    parser.add_argument("--sample-mode", choices=["front", "even"], default="front", help="How to sample CameraNoise frames.")
    parser.add_argument("--cfg", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--degradation-value", type=float, default=None, help="Degradation value. Random in [0,0.6] if omitted.")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Pipeline device.")
    parser.add_argument("--output-type", choices=["single", "concat", "ct1"], default="single")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def run_camera_noise_inference(
    *,
    reference_image,
    cameranoise,
    caption,
    output_dir,
    height,
    width,
    model_root,
    lora_path=None,
    output_name=None,
    frames=81,
    sample_mode="front",
    cfg=None,
    degradation_value=None,
    steps=25,
    seed=42,
    device="cuda",
    output_type="single",
    fps=20,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
):
    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline

    if degradation_value is None:
        degradation_value = random.uniform(0, 0.6)

    reference_image_path = Path(reference_image).expanduser().resolve()
    camera_noise_path = Path(cameranoise).expanduser().resolve()
    if not reference_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
    if not camera_noise_path.exists():
        raise FileNotFoundError(f"CameraNoise npy not found: {camera_noise_path}")

    image = Image.open(reference_image_path).convert("RGB")
    camera_noise, indices = load_camera_noise(camera_noise_path, frames, sample_mode)
    print(
        f"CameraNoise shape after sampling: {tuple(camera_noise.shape)}; "
        f"indices: {indices[0].item()}..{indices[-1].item()}"
    )

    model_configs = build_i2v_model_configs(model_root)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )

    if lora_path:
        pipe.load_lora(pipe.dit, str(lora_path), alpha=1)
    pipe.enable_vram_management()

    video = pipe(
        prompt=caption,
        negative_prompt=negative_prompt,
        input_image=image,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        seed=seed,
        tiled=True,
        camera_noise=camera_noise,
        degradation_value=degradation_value,
        cfg_scale=cfg,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    if output_name:
        final_output_name = output_name
        if not final_output_name.endswith(".mp4"):
            final_output_name += ".mp4"
    else:
        final_output_name = (
            f"{reference_image_path.stem}_{camera_noise_path.stem}_"
            f"{int(time.time())}.mp4"
        )
    output_path = output_dir / final_output_name
    save_generated_video(video, output_type, output_path, reference_image_path, caption, fps)
    print(f"video saved: {output_path}")
    return output_path


def main():
    args = parse_args()
    caption = read_caption(args)
    run_camera_noise_inference(
        reference_image=args.reference_image,
        cameranoise=args.cameranoise,
        caption=caption,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        model_root=args.model_root,
        lora_path=args.lora_path,
        output_name=args.output_name,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
