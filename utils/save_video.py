import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def ct1_save_video(
    frames,
    save_path,
    fps,
    ref_image=None,        # str path or None
    caption="",            # str
    bar_height=80,         # caption bar height
    font_path=None,        # optional ttf
    font_size=28,
    margin=20,
    quality=5,
    ffmpeg_params=None,
    assume_bgr=False,
):
    def to_uint8_rgb(frame):
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()

        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        frame = np.asarray(frame)

        # (C,H,W) -> (H,W,C)
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))

        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]

        if assume_bgr and frame.shape[-1] == 3:
            frame = frame[..., ::-1]

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                vmin, vmax = float(frame.min()), float(frame.max())
                if vmin >= -1 and vmax <= 1:
                    if vmin < 0:
                        frame = (frame + 1) / 2
                    frame = frame * 255
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        new_w = int(round(w * target_h / h))
        return np.array(
            Image.fromarray(img).resize((new_w, target_h), Image.BICUBIC)
        )

    def wrap_text(draw, text, font, max_width):
        if not text:
            return [""]

        lines, cur = [], ""
        for ch in text:
            test = cur + ch
            if draw.textlength(test, font=font) <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    if ffmpeg_params is None:
        ffmpeg_params = ["-pix_fmt", "yuv420p"]

    # load reference image
    ref_arr = None
    if ref_image is not None:
        ref_arr = np.array(Image.open(ref_image).convert("RGB"))

    # font
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    with imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params) as writer:
        for frame in tqdm(frames, desc="Saving video"):
            fr = to_uint8_rgb(frame)

            # left-right concat with ref image
            if ref_arr is not None:
                ref_resized = resize_to_height(ref_arr, fr.shape[0])
                combined = np.concatenate([ref_resized, fr], axis=1)
            else:
                combined = fr

            H, W = combined.shape[:2]

            # ⬅ 修改点 1：白色 bar 在「上方」
            out = np.ones((H + bar_height, W, 3), dtype=np.uint8) * 255
            out[bar_height:, :, :] = combined   # 视频下移

            if caption:
                img = Image.fromarray(out)
                draw = ImageDraw.Draw(img)

                max_text_w = W - 2 * margin
                lines = wrap_text(draw, caption, font, max_text_w)

                line_h = font_size + 6
                total_h = line_h * len(lines)
                y = max(0, (bar_height - total_h) // 2)  # ⬅ 修改点 2：文字在顶部 bar 内居中

                for line in lines:
                    draw.text((margin, y), line, fill=(0, 0, 0), font=font)
                    y += line_h

                out = np.array(img)

            writer.append_data(out)


if __name__=="__main__":
    video_saved_pth = f"{saved_video_dir}/{int(time.time())}_Image_{image_name}_Camera_{camera_name}.mp4"
    save_video(
        video_out,
        video_saved_pth,
        fps=20,
        ref_image="/path/to/ref.png",
        caption="A man walks into the room and picks up the toolbox.",
        bar_height=90,
        font_size=28
    )
