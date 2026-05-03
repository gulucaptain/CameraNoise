import json as _j
import os as _o

import gradio as _g

from utils.pipeline_shell import run_single_with_pose_files as _run_single_with_pose_files


def _maybe(v):
    return None if v in ("", None) else v


def _fmt_meta(x):
    return _j.dumps(x, ensure_ascii=False, indent=2)


def _go(
    video_file,
    intrinsic_file,
    extrinsic_file,
):
    if not video_file:
        raise _g.Error("请先选择视频文件。")
    if not intrinsic_file:
        raise _g.Error("请先选择内参文件。")
    if not extrinsic_file:
        raise _g.Error("请先选择外参文件。")
    video_path = video_file if isinstance(video_file, str) else video_file.get("path", "")
    intrinsic_path = intrinsic_file if isinstance(intrinsic_file, str) else intrinsic_file.get("path", "")
    extrinsic_path = extrinsic_file if isinstance(extrinsic_file, str) else extrinsic_file.get("path", "")
    for p, name in ((video_path, "视频文件"), (intrinsic_path, "内参文件"), (extrinsic_path, "外参文件")):
        if not p:
            raise _g.Error(f"没有拿到{name}路径。")
        if not _o.path.exists(p):
            raise _g.Error(f"{name}不存在: {p}")
    out = _run_single_with_pose_files(
        video_path=video_path,
        intrinsic_path=intrinsic_path,
        extrinsic_path=extrinsic_path,
    )
    vis = out["visualization_path"] if _o.path.exists(out["visualization_path"]) else None
    npy = out["noises_path"] if _o.path.exists(out["noises_path"]) else None
    # gif = out.get("preview_gif_path")
    status = "\n".join(
        [
            f"video: {out['video_name']}",
            f"device: {out['device']}",
            f"visualization: {vis or 'not generated'}",
            f"noises: {npy or 'not generated'}",
            f"config: {out['config_path']}",
            f"input extrinsic: {out['input_extrinsic_path']}",
            f"input intrinsic: {out['input_intrinsic_path']}",
            f"extrinsic: {out['extrinsic_path']}",
            f"intrinsic: {out['intrinsic_path']}",
        ]
    )
    meta = dict(out)
    meta["result"] = str(type(out["result"]).__name__) if out["result"] is not None else "None"
    return status, vis, npy, _fmt_meta(meta)


with _g.Blocks(title="CameraNoise Runner", theme=_g.themes.Soft()) as demo:
    _g.Markdown(
        """
        # CameraNoise Runner
        从文件夹中选择视频、内参文件、外参文件。
        其余参数统一从 `assets/inference.yaml` 读取，并运行 `GRFlow -> CameraNoise` 流程。
        """
    )
    with _g.Column():
        video_file = _g.File(label="视频文件", file_types=[".mp4", ".mov", ".avi", ".mkv"], type="filepath")
        intrinsic_file = _g.File(label="内参文件 (.pt)", file_types=[".pt"], type="filepath")
        extrinsic_file = _g.File(label="外参文件 (.pt)", file_types=[".pt"], type="filepath")
        # video_path = _g.Textbox(label="video_url")
        # intrinsic_path = _g.Textbox(label="intrinsic_url")
        # extrinsic_path = _g.Textbox(label="extrinsic_url")
    run_btn = _g.Button("运行", variant="primary")
    with _g.Row():
        status = _g.Textbox(label="状态", lines=8)
        meta = _g.Code(label="运行信息", language="json")
    with _g.Row():
        vis_video = _g.Video(label="可视化视频")
        noises_file = _g.File(label="Noise NPY")
        # preview_gif = _g.Image(label="Preview GIF")
    run_btn.click(
        fn=_go,
        inputs=[
            video_file,
            intrinsic_file,
            extrinsic_file,
        ],
        outputs=[status, vis_video, noises_file, meta],
    )


if __name__ == "__main__":
    demo.launch()
