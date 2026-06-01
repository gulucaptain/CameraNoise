from typing import List
import os
from PIL import Image
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip

def process_single_img(img_tensor: torch.Tensor, is_gpu: bool = True, target_size: tuple = None,) -> Image.Image:
    # 移除batch维度 + 转移到CPU
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    if is_gpu:
        img_np = img_tensor.cpu().detach().numpy()
    else:
        img_np = img_tensor.detach().numpy()
    
    # 维度转置 + 归一化到 0-255
    img_np = np.transpose(img_np, (1, 2, 0))
    min_val = img_np.min()
    max_val = img_np.max()
    img_np = (img_np - min_val) / (max_val - min_val) * 255
    img_np = img_np.astype(np.uint8)
    
    # 统一分辨率
    if target_size is not None:
        img_pil = Image.fromarray(img_np).resize((target_size[1], target_size[0]))
    else:
        img_pil = Image.fromarray(img_np)
    
    return img_pil

def tensor_list_to_mp4_moviepy(
    img_list: List[torch.Tensor],
    output_path: str = "output_moviepy.mp4",
    fps: int = 24,
    target_size: tuple = None,
    is_gpu: bool = True
):
    """
    使用MoviePy将张量列表保存为MP4
    :param img_list: 图像张量列表
    :param output_path: 输出路径
    :param fps: 帧率
    :param target_size: 统一分辨率 (H, W)（可选）
    :param is_gpu: 张量是否在GPU上
    """
    # 1. 处理所有图像，转为PIL Image列表
    pil_img_list = [process_single_img(img, is_gpu=is_gpu, target_size=target_size) for img in img_list]
    
    # 2. 生成视频（MoviePy自动处理格式）
    clip = ImageSequenceClip([np.array(img) for img in pil_img_list], fps=fps)
    # 保存为MP4（使用libx264编码器，确保兼容性）
    clip.write_videofile(output_path, codec="libx264")
    
    print(f"视频已保存至：{output_path}")
