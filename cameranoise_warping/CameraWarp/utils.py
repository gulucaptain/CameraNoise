import cv2
import torch
import numpy as np
from tqdm import tqdm

def resize_images(images, size, interp=cv2.INTER_LINEAR):
    resized = []
    for img in images:
        if isinstance(img, np.ndarray) and img.dtype == np.float16:
            img = img.astype(np.float32)
        if isinstance(size, (int, float)):  # 缩放比例
            h, w = img.shape[:2]
            out = cv2.resize(img, (0, 0), fx=size, fy=size, interpolation=interp)
        else:  # 指定目标尺寸 (h, w)
            h, w = size
            out = cv2.resize(img, (w, h), interpolation=interp)
        resized.append(out)
    return resized

def so3_log(R):
    """SO(3) → so(3)，旋转矩阵转轴角向量"""
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)  # 数值稳定
    theta = torch.acos(cos_theta)
    if theta < 1e-6:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    w = (1 / (2 * torch.sin(theta))) * torch.tensor([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ], device=R.device, dtype=R.dtype)
    return theta * w

def so3_exp(w):
    """so(3) → SO(3)，轴角向量转旋转矩阵"""
    theta = torch.norm(w)
    if theta < 1e-6:
        return torch.eye(3, device=w.device, dtype=w.dtype)
    k = w / theta
    K = torch.tensor([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], device=w.device, dtype=w.dtype)
    R = torch.eye(3, device=w.device, dtype=w.dtype) + \
        torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R

def se3_log(T):
    """SE(3) → se(3)，4x4矩阵转6D向量 (w,v)"""
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    return torch.cat([w, t])

def se3_exp(xi):
    """se(3) → SE(3)，6D向量转4x4矩阵"""
    w, v = xi[:3], xi[3:]
    R = so3_exp(w)
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = v
    return T


def rotation_matrix_to_angle_axis(rot_matrix: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为轴角表示（旋转向量）
    rot_matrix: [B, 3, 3] 旋转矩阵
    return: [B, 3] 旋转向量，其中向量的模长是旋转角度，方向是旋转轴
    """
    B, _, _ = rot_matrix.shape
    angle_axis = torch.zeros(B, 3, device=rot_matrix.device, dtype=rot_matrix.dtype)
    trace = rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
    
    # 计算旋转角度
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    
    # 处理接近0的角度（避免除以零）
    small_theta = theta < 1e-6
    if small_theta.any():
        angle_axis[small_theta] = (rot_matrix[small_theta] - rot_matrix[small_theta].transpose(1, 2)).reshape(-1, 9)[:, [7, 2, 3]] / 2
    
    # 处理非小角度
    non_small_theta = ~small_theta
    if non_small_theta.any():
        r = torch.stack([
            rot_matrix[non_small_theta, 2, 1] - rot_matrix[non_small_theta, 1, 2],
            rot_matrix[non_small_theta, 0, 2] - rot_matrix[non_small_theta, 2, 0],
            rot_matrix[non_small_theta, 1, 0] - rot_matrix[non_small_theta, 0, 1]
        ], dim=1)
        r_norm = torch.norm(r, dim=1, keepdim=True)
        angle_axis[non_small_theta] = (theta[non_small_theta].unsqueeze(1) / r_norm) * r
    
    return angle_axis

def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    将轴角表示（旋转向量）转换为旋转矩阵
    angle_axis: [B, 3] 旋转向量
    return: [B, 3, 3] 旋转矩阵
    """
    B = angle_axis.shape[0]
    device = angle_axis.device
    dtype = angle_axis.dtype
    
    # 计算旋转角度和轴
    theta = torch.norm(angle_axis, dim=1, keepdim=True)
    theta2 = theta * theta
    theta3 = theta2 * theta
    
    # 处理零角度情况
    zero_theta = theta < 1e-6
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    if zero_theta.all():
        return eye
    
    # 计算旋转轴的单位向量
    k = angle_axis / theta
    
    # 构建反对称矩阵
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zeros = torch.zeros(B, device=device, dtype=dtype)
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=1),
        torch.stack([kz, zeros, -kx], dim=1),
        torch.stack([-ky, kx, zeros], dim=1)
    ], dim=1)
    
    # 使用罗德里格斯公式计算旋转矩阵
    R = eye + K * torch.sin(theta).unsqueeze(1) + \
        K @ K * (1 - torch.cos(theta)).unsqueeze(1)
    
    # 对零角度使用单位矩阵
    R[zero_theta] = eye[zero_theta]
    
    return R

def scale_rotation_matrix(R: torch.Tensor, alpha=0.5):
    """
    缩放旋转矩阵的旋转幅度
    R: [B, 3, 3] 旋转矩阵
    alpha: 缩放因子
    return: [B, 3, 3] 缩放后的旋转矩阵
    """
    # 旋转矩阵 -> 旋转向量（轴角表示）
    angle_axis = rotation_matrix_to_angle_axis(R)
    
    # 缩放旋转向量（等价于缩放旋转角度）
    scaled_angle_axis = angle_axis * alpha
    
    # 旋转向量 -> 旋转矩阵
    R_scaled = angle_axis_to_rotation_matrix(scaled_angle_axis)
    
    return R_scaled
