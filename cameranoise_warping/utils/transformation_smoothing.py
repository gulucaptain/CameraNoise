import torch
import numpy as np

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
