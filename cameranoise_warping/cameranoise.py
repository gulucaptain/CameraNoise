import argparse
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def create_save_dirs(dir_list):
    for target_dir in dir_list:
        Path(target_dir).mkdir(parents=True, exist_ok=True)


def read_yaml(path):
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return read_simple_yaml(path)


def parse_scalar(value):
    value = value.strip()
    if value in ("", '""', "''"):
        return ""
    if value == "[]":
        return []
    if value == "{}":
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def read_simple_yaml(path):
    """Small fallback parser for this project's flat config files."""
    data = {}
    current_list_key = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            stripped = line.strip()
            if stripped.startswith("- "):
                if current_list_key is None:
                    raise ValueError(f"List item without a key in {path}: {raw_line.rstrip()}")
                data[current_list_key].append(parse_scalar(stripped[2:]))
                continue

            if ":" not in stripped:
                raise ValueError(f"Unsupported yaml line in {path}: {raw_line.rstrip()}")

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = parse_scalar(value)
                current_list_key = None

    return data


def merge_config(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    config = read_yaml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    if config_path:
        override = read_yaml(config_path)
        config = merge_config(config, override)
    return SimpleNamespace(**config)


def read_video_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def collect_video_paths(cfg, args):
    if args.videos:
        videos = args.videos
    elif args.videos_txt:
        videos = read_video_txt(args.videos_txt)
    elif hasattr(cfg, "videos") and cfg.videos:
        videos = cfg.videos
    elif hasattr(cfg, "videos_txt") and cfg.videos_txt:
        videos = read_video_txt(cfg.videos_txt)
    elif hasattr(cfg, "inference_video_datas"):
        # Backward compatibility with the old inference.yaml schema.
        if isinstance(cfg.inference_video_datas, list):
            videos = cfg.inference_video_datas
        elif isinstance(cfg.inference_video_datas, str) and cfg.inference_video_datas.endswith(".txt"):
            videos = read_video_txt(cfg.inference_video_datas)
        elif isinstance(cfg.inference_video_datas, str):
            videos = [cfg.inference_video_datas]
        else:
            raise ValueError("inference_video_datas must be a list, video path, or .txt path")
    else:
        raise ValueError("No videos configured. Use videos, videos_txt, --videos, or --videos-txt.")

    videos = [str(video).strip() for video in videos if str(video).strip()]
    if not videos:
        raise ValueError("Video list is empty.")
    return videos


def select_video_paths(videos, args):
    selected = videos

    if args.shard_id is not None or args.num_shards is not None:
        if args.shard_id is None or args.num_shards is None:
            raise ValueError("--shard-id and --num-shards must be used together.")
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be positive.")
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError("--shard-id must be in [0, num_shards).")
        selected = [video for index, video in enumerate(selected) if index % args.num_shards == args.shard_id]

    if args.start is not None:
        if args.start < 0:
            raise ValueError("--start must be >= 0.")
        selected = selected[args.start :]

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be >= 0.")
        selected = selected[: args.limit]

    return selected


def require_config(cfg, names):
    missing = [name for name in names if not hasattr(cfg, name) or getattr(cfg, name) in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")


def cfg_get(cfg, name, default):
    return getattr(cfg, name, default)


def cameranoise_std_reference_size(cfg):
    return cfg_get(cfg, "cameranoise_std_reference_size", None)


def video_stem(video_path):
    return Path(str(video_path).strip()).stem


def expected_noise_hw(resized_height, resized_width, cfg, downscale_factor):
    downscale_size = cfg_get(cfg, "cameranoise_downscale_size", None)
    std_reference_size = cameranoise_std_reference_size(cfg)

    if downscale_size is not None:
        if len(downscale_size) != 2:
            raise ValueError("cameranoise_downscale_size must be [height, width].")
        return int(downscale_size[0]), int(downscale_size[1])

    noise_height = int(resized_height * cfg.FLOW)
    noise_width = int(resized_width * cfg.FLOW)
    if std_reference_size is not None:
        derived_downscale = noise_width / std_reference_size
        return max(1, int(round(noise_height / derived_downscale))), int(std_reference_size)

    return (
        max(1, int(round(noise_height / downscale_factor))),
        max(1, int(round(noise_width / downscale_factor))),
    )


def validate_noise_output(numpy_noises, *, video_name, expected_frames, expected_channels, expected_height, expected_width):
    if numpy_noises is None:
        raise ValueError(f"{video_name}: CameraNoise output is None.")
    if len(numpy_noises.shape) != 4:
        raise ValueError(f"{video_name}: expected CameraNoise shape [T,H,W,C], got {numpy_noises.shape}.")

    expected_shape = (expected_frames, expected_height, expected_width, expected_channels)
    if tuple(numpy_noises.shape) != expected_shape:
        raise ValueError(f"{video_name}: expected CameraNoise shape {expected_shape}, got {numpy_noises.shape}.")


def convert_camera_pose_to_GRFlow(
    frame_nums,
    video_name,
    intrinsics,
    extrinsics,
    resolution,
    transformation_smoothing_alpha,
    depth,
    bs,
    resized_height,
    resized_width,
    grflow_visualization,
    grflow_saved_dir,
    device,
):
    import torch
    from tqdm import tqdm

    from CameraWarp.grflow_reprojection import GRFlowReprojector
    from utils.flow_visualization import grflow_save_as_video

    grflow_reprojector = GRFlowReprojector(
        resolution,
        transformation_smoothing_alpha,
        b=bs,
        h=resized_height,
        w=resized_width,
        device=device,
    )

    depth1 = torch.full((bs, 1, resized_height, resized_width), depth, dtype=torch.float32, device=device)
    grflows = []
    for i in tqdm(range(frame_nums - 1), desc=f"GRFlow {video_name}"):
        intrinsic1 = intrinsics[i, ...].unsqueeze(0)
        intrinsic2 = intrinsics[i + 1, ...].unsqueeze(0)
        extrinsic1 = extrinsics[i, ...]
        extrinsic2 = extrinsics[i + 1, ...]

        new_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        extrinsic1 = torch.cat([extrinsic1, new_row], dim=0).unsqueeze(0)
        extrinsic2 = torch.cat([extrinsic2, new_row], dim=0).unsqueeze(0)

        grflow = grflow_reprojector.forward_grflow(
            depth1,
            extrinsic1,
            extrinsic2,
            intrinsic1,
            intrinsic2,
            is_image=False,
            frame1=None,
        )
        grflows.append(grflow)

    if grflow_visualization:
        grflow_saved_pth = Path(grflow_saved_dir) / f"{video_name}.mp4"
        grflow_save_as_video(grflows, str(grflow_saved_pth))

    return grflows


def load_or_estimate_camera(vggt_model, cfg, video_path, camera_pose_saved_root, device):
    import torch

    from vggt.vggt_inference import VGGT_estimation

    extrinsic_saved_pth = camera_pose_saved_root / "extrinsic.pt"
    intrinsic_saved_pth = camera_pose_saved_root / "intrinsic.pt"

    if extrinsic_saved_pth.exists() and intrinsic_saved_pth.exists():
        intrinsics = torch.load(intrinsic_saved_pth, map_location=device).squeeze(0)
        extrinsics = torch.load(extrinsic_saved_pth, map_location=device).squeeze(0)
        return extrinsics, intrinsics

    extrinsics, intrinsics, _, _ = VGGT_estimation(
        vggt_model,
        cfg.vggt_estimation_target_size,
        str(video_path),
        str(extrinsic_saved_pth),
        str(intrinsic_saved_pth),
        device,
        cfg_get(cfg, "warmup_intrinsics", False),
        cfg_get(cfg, "return_estimated_depth", False),
    )
    return extrinsics.squeeze(0), intrinsics.squeeze(0)


def process_one_video(vggt_model, cfg, video_path, device):
    from decord import VideoReader

    from CameraWarp.camerawarp import get_noise_from_video
    from utils.intrinsic_kalmanfilter import smooth_intrinsics

    begin_time = time.time()
    video_path = str(video_path).strip()
    video_name = video_stem(video_path)

    vr = VideoReader(video_path)
    video_length = len(vr)
    if video_length < 2:
        raise ValueError(f"Video must contain at least 2 frames: {video_path}")

    data_saved_root = Path(cfg.data_saved_root)
    grflow_saved_dir = data_saved_root / "grflows"
    camera_pose_saved_root = data_saved_root / "camerapose" / video_name
    camera_noise_saved_root = data_saved_root / "noises"
    flow_saved_root = data_saved_root / "flows"
    vis_mp4_path = camera_noise_saved_root / f"{video_name}_visualization.mp4"
    noises_path = camera_noise_saved_root / f"{video_name}_noises.npy"
    flows_path = flow_saved_root / f"{video_name}_flows.npy"

    if noises_path.exists() and not cfg_get(cfg, "overwrite", False):
        print(f"skip existing noise: {noises_path}")
        return

    save_dirs = [grflow_saved_dir, camera_pose_saved_root, camera_noise_saved_root]
    if cfg_get(cfg, "saved_flow", False):
        save_dirs.append(flow_saved_root)
    create_save_dirs(save_dirs)
    extrinsics, intrinsics = load_or_estimate_camera(vggt_model, cfg, video_path, camera_pose_saved_root, device)

    if not cfg_get(cfg, "get_cameranoise_or_not", True):
        print(f"camera pose saved only: {video_name}")
        return

    if cfg_get(cfg, "intrinsic_smoothing", True):
        intrinsics = smooth_intrinsics(intrinsics, device=intrinsics.device)

    frame_nums = extrinsics.shape[0]
    height, width = vr[0].shape[0], vr[0].shape[1]
    resized_width = int(width * cfg.FRAME)
    resized_height = int(height * cfg.FRAME)
    print(
        f"{video_name}: frames={video_length}, "
        f"origin={width}x{height}, resized={resized_width}x{resized_height}"
    )

    grflows = convert_camera_pose_to_GRFlow(
        frame_nums,
        video_name,
        intrinsics,
        extrinsics,
        (resized_height, resized_width),
        cfg.transformation_smoothing_alpha,
        cfg.depth,
        cfg.bs,
        resized_height,
        resized_width,
        cfg_get(cfg, "grflow_visualization", False),
        grflow_saved_dir,
        device,
    )

    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
    except ImportError:
        process = None
        mem_before = math.nan
    downscale_factor = round(cfg.FRAME * cfg.FLOW) * cfg.LATENT
    output = get_noise_from_video(
        video_path,
        grflows,
        visualize=cfg_get(cfg, "cameranoise_visualize", False),
        save_files=cfg_get(cfg, "cameranoise_save_files", True),
        noise_channels=cfg_get(cfg, "noise_channels", 16),
        resize_frames=cfg.FRAME,
        resize_flow=cfg.FLOW,
        downscale_factor=downscale_factor,
        device=device,
        vis_mp4_path=str(vis_mp4_path),
        noises_path=str(noises_path),
        std_reference_size=cameranoise_std_reference_size(cfg),
        downscale_size=cfg_get(cfg, "cameranoise_downscale_size", None),
        visualize_match_video_size=cfg_get(cfg, "cameranoise_visualize_match_video_size", True),
        return_flows=cfg_get(cfg, "saved_flow", False),
        progressive_noise_alpha=cfg_get(cfg, "progressive_noise_alpha", 0),
        post_noise_alpha=cfg_get(cfg, "post_noise_alpha", 0),
    )
    mem_after = process.memory_info().rss / 1024 / 1024 if process else math.nan

    expected_height, expected_width = expected_noise_hw(resized_height, resized_width, cfg, downscale_factor)
    validate_noise_output(
        output["numpy_noises"],
        video_name=video_name,
        expected_frames=frame_nums,
        expected_channels=cfg_get(cfg, "noise_channels", 16),
        expected_height=expected_height,
        expected_width=expected_width,
    )

    if cfg_get(cfg, "cameranoise_save_files", True):
        print(f"saved noise: {noises_path}")
    print(f"noise shape [T,H,W,C]: {output['numpy_noises'].shape}")
    if cfg_get(cfg, "saved_flow", False):
        import numpy as np

        if output["numpy_flows"] is None:
            raise RuntimeError(f"No flows were generated for {video_path}")
        np.save(flows_path, output["numpy_flows"])
        print(f"saved flow: {flows_path}")
    if process:
        print(f"memory delta: {mem_after - mem_before:.2f} MB")
    print(f"time cost: {time.time() - begin_time:.2f}s")


def write_error(error_file, video_path, exc):
    error_file.parent.mkdir(parents=True, exist_ok=True)
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(f"video: {video_path}\n")
        f.write(f"error: {repr(exc)}\n")
        f.write(traceback.format_exc())
        f.write("\n---\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CameraNoise from videos with VGGT camera poses.")
    parser.add_argument("--config", type=Path, help="Experiment yaml. Defaults are loaded from configs/default.yaml.")
    parser.add_argument("--videos", nargs="+", help="Video paths. Overrides videos/videos_txt in yaml.")
    parser.add_argument("--videos-txt", type=Path, help="Text file with one video path per line.")
    parser.add_argument("--output-root", type=Path, help="Override data_saved_root.")
    parser.add_argument("--ckpt", type=str, help="Override vggt_ckpt_pth.")
    parser.add_argument("--device", type=str, help="Override device, for example cuda:0 or cpu.")
    parser.add_argument("--start", type=int, help="Start index after optional sharding.")
    parser.add_argument("--limit", type=int, help="Maximum number of videos to process after optional sharding.")
    parser.add_argument("--shard-id", type=int, help="Shard id in [0, num_shards).")
    parser.add_argument("--num-shards", type=int, help="Total shard count.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected videos and exit before loading VGGT.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing noise npy files.")
    parser.add_argument("--no-gallery", action="store_true", help="Do not build index.html after processing.")
    parser.add_argument("--gallery-output", type=Path, help="Output HTML path. Defaults to data_saved_root/index.html.")
    parser.add_argument("--gallery-max-items", type=int, default=24, help="Maximum gallery items. Use 0 to show all.")
    parser.add_argument(
        "--gallery-sample",
        choices=["first", "even", "random"],
        default="even",
        help="Gallery sampling strategy when result count exceeds --gallery-max-items.",
    )
    parser.add_argument("--gallery-seed", type=int, default=0, help="Random seed for --gallery-sample random.")
    return parser.parse_args()


def build_result_gallery(cfg, args):
    try:
        from build_gallery import build_gallery
    except ModuleNotFoundError:
        from cameranoise_warping.build_gallery import build_gallery

    result = build_gallery(
        cfg.data_saved_root,
        output=args.gallery_output,
        max_items=args.gallery_max_items,
        sample=args.gallery_sample,
        seed=args.gallery_seed,
    )
    print(f"gallery saved: {result['html_path']}")
    print(f"gallery shown: {result['shown']} / {result['total']}")
    return result


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.output_root:
        cfg.data_saved_root = str(args.output_root)
    if args.ckpt:
        cfg.vggt_ckpt_pth = args.ckpt
    if args.overwrite:
        cfg.overwrite = True

    require_config(
        cfg,
        [
            "vggt_ckpt_pth",
            "vggt_estimation_target_size",
            "data_saved_root",
            "FRAME",
            "FLOW",
            "LATENT",
            "bs",
            "depth",
            "transformation_smoothing_alpha",
        ],
    )

    videos = select_video_paths(collect_video_paths(cfg, args), args)
    print(f"selected videos: {len(videos)}")
    if args.dry_run:
        for index, video in enumerate(videos):
            print(f"{index}: {video}")
        return 0
    if not videos:
        print("No videos selected.")
        return 0

    import torch
    from tqdm import tqdm

    from vggt.models.vggt import VGGT

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vggt_model = VGGT.from_pretrained(cfg.vggt_ckpt_pth).to(device)
    error_file = Path(cfg.data_saved_root) / "errors.txt"

    failed = 0
    for video_path in tqdm(videos, desc="videos"):
        try:
            process_one_video(vggt_model, cfg, video_path, device)
        except Exception as exc:
            failed += 1
            write_error(error_file, video_path, exc)
            print(f"failed: {video_path}: {exc}")

    if not args.no_gallery:
        build_result_gallery(cfg, args)

    if failed:
        print(f"done with failures: {failed}/{len(videos)}. See {error_file}")
        return 1

    print(f"done: {len(videos)} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
