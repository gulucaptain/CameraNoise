import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

def flow_to_color(flow):
    """
    将光流 (H, W, 2) 转换为 RGB 可视化图
    flow[..., 0] = u, 水平位移
    flow[..., 1] = v, 垂直位移
    """
    u = flow[..., 0]
    v = flow[..., 1]

    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2                 # 方向 -> 色调 (0-180)
    hsv[..., 1] = 255                     # 饱和度固定
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 速度大小 -> 亮度
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def grflow_save_as_video(flow_list, flow_saved_pth):
    flow_list = [flow.cpu() for flow in flow_list]
    flow_array = np.array(flow_list) # shape: [T, 2, H, W]

    flow_array = np.squeeze(flow_array)
    T, C, H, W = flow_array.shape

    flow_array = np.transpose(flow_array, (0, 2, 3, 1)) # reshape: to [T, H, W, C]

    frames = []
    for t in range(T):
        vis = flow_to_color(flow_array[t, :, :, :2])  # 只用前两个通道
        frames.append(vis)

    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames], fps=20)
    clip.write_videofile(flow_saved_pth, codec="libx264")
