import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERANOISE_ROOT = PROJECT_ROOT / "cameranoise_warping"


def ensure_import_paths():
    for path in (PROJECT_ROOT, CAMERANOISE_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def infer_device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_paths(output_root, video_id, saved_flow=False):
    output_root = Path(output_root)
    paths = {
        "video_id": video_id,
        "output_root": output_root,
        "noise_path": output_root / "noises" / f"{video_id}_noises.npy",
        "visualization_path": output_root / "noises" / f"{video_id}_visualization.mp4",
        "intrinsic_path": output_root / "camerapose" / video_id / "intrinsic.pt",
        "extrinsic_path": output_root / "camerapose" / video_id / "extrinsic.pt",
        "flow_path": None,
    }
    if saved_flow:
        paths["flow_path"] = output_root / "flows" / f"{video_id}_flows.npy"
    return paths


def build_cameranoise(
    vggt_model,
    reference_video,
    output_root,
    *,
    device=None,
    config_path=None,
    move_model_to_device=True,
    overwrite=None,
    **overrides,
):
    """Build CameraNoise for one reference video with an externally supplied VGGT model.

    Parameters
    ----------
    vggt_model:
        A loaded VGGT model instance.
    reference_video:
        Path to the input reference video.
    output_root:
        Directory where CameraNoise outputs will be saved.
    device:
        Torch device string or object. Defaults to cuda when available, otherwise cpu.
    config_path:
        Optional yaml config. Defaults are always loaded from cameranoise_warping/configs/default.yaml.
    move_model_to_device:
        Move the provided model to `device` before running.
    overwrite:
        Optional override for cfg.overwrite.
    **overrides:
        Extra config overrides, for example saved_flow=True,
        cameranoise_std_reference_size=96, or cameranoise_downscale_size=[72, 128].

    Returns
    -------
    dict
        Paths to generated or expected output files.
    """
    ensure_import_paths()

    from cameranoise import load_config, process_one_video, video_stem

    reference_video = Path(reference_video).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not reference_video.exists():
        raise FileNotFoundError(f"Reference video not found: {reference_video}")

    cfg = load_config(config_path)
    cfg.data_saved_root = str(output_root)
    if overwrite is not None:
        cfg.overwrite = bool(overwrite)
    for key, value in overrides.items():
        setattr(cfg, key, value)

    torch_device = infer_device(device)
    if move_model_to_device and hasattr(vggt_model, "to"):
        vggt_model = vggt_model.to(torch_device)

    process_one_video(vggt_model, cfg, str(reference_video), torch_device)
    return make_output_paths(
        output_root,
        video_stem(reference_video),
        saved_flow=bool(getattr(cfg, "saved_flow", False)),
    )


def load_vggt_model(ckpt_path, device=None):
    ensure_import_paths()

    from vggt.models.vggt import VGGT

    torch_device = infer_device(device)
    return VGGT.from_pretrained(ckpt_path).to(torch_device)


def parse_args():
    parser = argparse.ArgumentParser(description="Build CameraNoise for one reference video.")
    parser.add_argument("--reference-video", required=True, type=Path, help="Input reference video path.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output directory.")
    parser.add_argument("--ckpt", type=str, help="VGGT checkpoint path. Overrides vggt_ckpt_pth in config.")
    parser.add_argument("--config", type=Path, help="Optional CameraNoise yaml config.")
    parser.add_argument("--device", type=str, help="Device, for example cuda:0 or cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing noise outputs.")
    parser.add_argument("--saved-flow", action="store_true", help="Save generated flow npy.")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization mp4 output.")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_import_paths()

    from cameranoise import load_config

    cfg = load_config(args.config)
    ckpt = args.ckpt or getattr(cfg, "vggt_ckpt_pth", "")
    if not ckpt:
        raise SystemExit("Provide --ckpt or set vggt_ckpt_pth in --config/default config.")

    vggt_model = load_vggt_model(ckpt, args.device)
    result = build_cameranoise(
        vggt_model,
        args.reference_video,
        args.output_root,
        device=args.device,
        config_path=args.config,
        overwrite=args.overwrite,
        saved_flow=args.saved_flow or getattr(cfg, "saved_flow", False),
        cameranoise_visualize=not args.no_visualization,
    )

    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
