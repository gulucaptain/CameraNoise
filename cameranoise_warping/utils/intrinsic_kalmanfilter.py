import torch

class IntrinsicKalmanFilter:
    def __init__(self, process_var=1e-3, measure_var=1e-2, device=None):
        self.state = None
        self.P = None
        self.Q = torch.eye(4) * process_var
        self.R = torch.eye(4) * measure_var
        self.F = torch.eye(4)
        self.H = torch.eye(4)
        self.device = device

        self.Q = self.Q.contiguous().to(self.device)
        self.R = self.R.contiguous().to(self.device)
        self.F = self.F.contiguous().to(self.device)
        self.H = self.H.contiguous().to(self.device)


    def initialize_from_buffer(self, init_buffer: torch.Tensor):
        """
        init_buffer: [N, 3, 3] 前 N 帧的内参矩阵
        """
        fx = init_buffer[:, 0, 0]
        fy = init_buffer[:, 1, 1]
        cx = init_buffer[:, 0, 2]
        cy = init_buffer[:, 1, 2]
        self.state = torch.stack([fx, fy, cx, cy], dim=1).mean(dim=0)
        self.P = torch.eye(4) * 0.1  # 大初始协方差，加快收敛

        self.state = self.state.to(self.device)
        self.P = self.P.to(self.device)

    def update(self, K: torch.Tensor):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        z = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).contiguous().to(self.device)

        if self.state is None:
            return K

        # 预测
        x_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新
        y = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K_gain = P_pred @ self.H.T @ torch.linalg.inv(S)

        self.state = x_pred + K_gain @ y
        self.P = (torch.eye(4).contiguous().to(self.device) - K_gain @ self.H) @ P_pred

        # 返回平滑后的内参
        K_smooth = K.clone()
        K_smooth[0, 0] = self.state[0]
        K_smooth[1, 1] = self.state[1]
        K_smooth[0, 2] = self.state[2]
        K_smooth[1, 2] = self.state[3]

        return K_smooth

def smooth_intrinsics(intrinsic_seq: torch.Tensor, init_buffer_size=3, process_var=1e-3, measure_var=1e-2, device=None):
    """
    对整个序列的内参进行平滑
    intrinsic_seq: [B, 3, 3]
    """
    kf = IntrinsicKalmanFilter(process_var=process_var, measure_var=measure_var, device=device)
    kf.initialize_from_buffer(intrinsic_seq[:init_buffer_size])

    smoothed_intrinsics = []
    for i in range(intrinsic_seq.shape[0]):
        K_smooth = kf.update(intrinsic_seq[i])
        smoothed_intrinsics.append(K_smooth)
    return torch.stack(smoothed_intrinsics, dim=0)
