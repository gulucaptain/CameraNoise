import numpy as np
import torch
from tqdm import tqdm

from CameraWarp.noise_warp import NoiseWarper
from CameraWarp.load_video import load_video_file
from CameraWarp.utils import resize_images
from utils.image_tensor_utils import (
    as_numpy_array,
    as_numpy_image,
    as_rgb_images,
    get_image_dimensions,
    optical_flow_to_image,
    torch_resize_image,
)

import imageio
import cv2

def downscale_noise(noise, downscale_factor, downscale_size=None):
    resize_size = 1 / downscale_factor if downscale_size is None else tuple(downscale_size)
    down_noise = torch_resize_image(noise, resize_size, interp="area")
    down_noise = down_noise * downscale_factor
    return down_noise

def add_text_above(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
    H, W, C = image.shape
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_h = text_size[1] + 6  # 文字区域高度
    new_img = np.zeros((H + text_h, W, C), dtype=image.dtype)

    new_img[text_h:, :, :] = image  

    text_x = (W - text_size[0]) // 2
    text_y = text_size[1] + 3
    cv2.putText(new_img, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return new_img

def concat_with_text(img1, img2, text1="Input Video", text2="Output CameraNoise", add_arrow=True, arrow_color=(255, 255, 255), arrow_thickness=3, gap_width=30):
    if text1 != "" and text2 != "":
        img1_text = add_text_above(img1, text1)
        img2_text = add_text_above(img2, text2)
    else:
        img1_text = img1
        img2_text = img2
    
    H = max(img1_text.shape[0], img2_text.shape[0])
    W1 = img1_text.shape[1]
    W2 = img2_text.shape[1]

    if add_arrow:
        gap = np.ones((H, gap_width, 3), dtype=img1.dtype) * 0
        w = gap_width
        h = H
        center = (w // 2, h // 2)

        start_x = 8
        end_x = w - 15
        cv2.line(gap, (start_x, center[1]), (end_x, center[1]), (255,255,255), 3)

        tip = (w - 8, center[1])
        left = (end_x, center[1] - 10)
        right = (end_x, center[1] + 10)
        pts = np.array([tip, left, right], np.int32).reshape((-1,1,2))
        cv2.fillPoly(gap, [pts], (255,255,255))
        
        canvas = np.zeros((H, W1 + gap_width + W2, 3), dtype=img1.dtype)
        canvas[:img1_text.shape[0], :W1] = img1_text
        canvas[:H, W1:W1+gap_width] = gap
        canvas[:img2_text.shape[0], W1+gap_width:W1+gap_width+W2] = img2_text
        
        return canvas
    else:
        canvas = np.zeros((H, W1 + W2, 3), dtype=img1.dtype)
        canvas[:img1_text.shape[0], :W1] = img1_text
        canvas[:img2_text.shape[0], W1:W1+W2] = img2_text

        return canvas


def _clamp_float_image(image):
    image = np.clip(image, 0, 1)

    return image


def pad_frames_to_multiple(frames, multiple=16):
    height, width = frames.shape[1:3]
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple
    if (padded_height, padded_width) == (height, width):
        return frames

    pad_bottom = padded_height - height
    pad_right = padded_width - width
    return np.pad(
        frames,
        ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )

def get_noise_from_video(
    video_path: str,
    optical_flows,
    noise_channels: int = 3,
    visualize: bool = True,
    resize_frames: tuple = None,
    resize_flow: int = 1,
    downscale_factor: int = 1,
    device=None,
    vis_mp4_path="",
    noises_path="",
    save_files=True,
    progressive_noise_alpha = 0,
    post_noise_alpha = 0,
    visualize_flow_sensitivity=None,
    std_reference_size=None,
    downscale_size=None,
    visualize_match_video_size=True,
    return_flows=False,
    warp_kwargs=dict(),
):
    assert isinstance(resize_flow, int) and resize_flow >= 1, resize_flow

    if not isinstance(video_path, str):
        raise TypeError(f"video_path must be a string path, got {type(video_path)}")
    video_frames = load_video_file(video_path)
    
    if resize_frames is not None:
        video_frames = resize_images(video_frames, size=resize_frames)
        
    video_frames = as_rgb_images(video_frames)
    video_frames = np.stack(video_frames)
    video_frames = video_frames.astype(np.float16)/255
    _, h, w, _ = video_frames.shape
    
    with torch.no_grad():
        warper = NoiseWarper(
            c = noise_channels,
            h = resize_flow * h,
            w = resize_flow * w,
            device = device,
            post_noise_alpha = post_noise_alpha,
            progressive_noise_alpha = progressive_noise_alpha,
            warp_kwargs = warp_kwargs,
        )
        
        noise = warper.noise
        if std_reference_size is not None:
            given_size = noise.shape[-1]
            downscale_factor = given_size / std_reference_size
        down_noise = downscale_noise(noise, downscale_factor, downscale_size)
        numpy_noise = as_numpy_image(down_noise).astype(np.float16)
        
        numpy_noises = [numpy_noise]
        numpy_flows = []
        vis_frames = []

        for index, optical_flow in enumerate(tqdm(optical_flows, desc="warp noise")):
            optical_flow = optical_flow.squeeze(0)
            dx, dy = torch.split(optical_flow, 1, dim=0)
            dx = dx.squeeze(0)
            dy = dy.squeeze(0)
            
            noise = warper(dx, dy).noise

            if return_flows:
                numpy_flow = np.stack(
                    [
                        as_numpy_array(dx).astype(np.float16),
                        as_numpy_array(dy).astype(np.float16),
                    ]
                )
                numpy_flows.append(numpy_flow)

            down_noise = downscale_noise(noise, downscale_factor, downscale_size)
            numpy_noise = as_numpy_image(down_noise).astype(np.float16)
            numpy_noises.append(numpy_noise)

            if visualize:
                video_frame = video_frames[index]
                flow_rgb = optical_flow_to_image(dx, dy, sensitivity = visualize_flow_sensitivity)
                down_noise_image = np.zeros((*numpy_noise.shape[:2], 3))
                down_noise_image_c = min(noise_channels, 3)
                down_noise_image[:, :, :down_noise_image_c] = numpy_noise[:, :, :down_noise_image_c]

                if visualize_match_video_size:
                    visualize_size = get_image_dimensions(video_frame)
                    down_video_frame = video_frame
                    vis_noise_image, _ = resize_images([down_noise_image, flow_rgb], size=visualize_size)
                else:
                    visualize_size = get_image_dimensions(down_noise_image)
                    down_video_frame, _ = resize_images([video_frame, flow_rgb], size=visualize_size)
                    vis_noise_image = down_noise_image

                concat_frame_with_text = concat_with_text(
                    down_video_frame,
                    vis_noise_image / 3 + 0.5,
                    text1="Input Video",
                    text2="Output CameraNoise",
                    add_arrow=True,
                    arrow_color=(255, 255, 255),
                    arrow_thickness=3,
                    gap_width=40
                )
                per_video_frame = (_clamp_float_image(concat_frame_with_text) * 255).astype(np.uint8)
                vis_frames.append(per_video_frame)
    
    numpy_noises = np.stack(numpy_noises).astype(np.float16)
    numpy_flows = np.stack(numpy_flows).astype(np.float16) if numpy_flows else None
    if vis_frames:
        vis_frames = pad_frames_to_multiple(np.stack(vis_frames), multiple=16)
    
    if visualize:
        fps = 30
        imageio.mimwrite(
            vis_mp4_path,
            vis_frames,
            fps=fps,
            codec="libx264",
            quality=8,       # 默认是 5，越低越压缩
            ffmpeg_params=["-crf", "28"]  # 控制码率，18 高质量大文件，28 较低质量小文件
        )
        print(f"Video has been saved in: {vis_mp4_path}")

    if save_files:
        np.save(noises_path, numpy_noises)

    return {
        "numpy_noises": numpy_noises,
        "numpy_flows": numpy_flows,
        "vis_frames": vis_frames,
    }
