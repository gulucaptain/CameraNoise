import torch
import cv2
import numpy as np

def scale_rotation_matrix(transformation, alpha=0.5):
    """
    R: [B,3,3] 旋转矩阵
    alpha: 缩放旋转幅度
    return: 缩放后的旋转矩阵 [B,3,3]
    """
    R = transformation[:, :3, :3]
    t = transformation[:, :3, 3:]
    
    B = R.shape[0]
    R_scaled = torch.zeros_like(R)
    for b in range(B):
        R_b = R[b].cpu().numpy().astype(np.float64)
        rvec, _ = cv2.Rodrigues(R_b)        # 旋转矩阵 -> 旋转向量
        rvec_scaled = alpha * rvec          # 缩放旋转幅度
        R_scaled[b] = torch.from_numpy(cv2.Rodrigues(rvec_scaled)[0]).contiguous().to(R.dtype)
    
    transformation[:, :3, :3] = R_scaled
    transformation[:, :3, 3:] = t  # 平移可保留原值，或者略微缩放
    
    return transformation
