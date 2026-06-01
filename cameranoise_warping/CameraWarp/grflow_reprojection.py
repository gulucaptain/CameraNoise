from typing import Tuple, Optional
import torch
from einops import rearrange

from utils.scale_rotation_matrix import scale_rotation_matrix
from utils.transformation_smoothing import se3_log, se3_exp

class GRFlowReprojector:
    def __init__(self, resolution: tuple = None, transformation_smoothing_alpha=None, b=1, h=None, w=None, device=None):
        self.resolution = resolution
        self.batch_size = b
        self.width = w
        self.height = h
        self.device = device
        
        self.alpha = transformation_smoothing_alpha
        self.prev_xi = None
        self.intrinsic1 = None
    
    def forward_grflow(self, depth1: torch.Tensor, extrinsic1: torch.Tensor, extrinsic2: torch.Tensor, intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor], 
                        frame1: Optional[torch.Tensor], is_image=True) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations extrinsic1 and extrinsic2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param depth1: (b, 1, h, w)
        :param extrinsic1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param extrinsic2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone() # 传入相机内参（当前帧和下一帧）
        
        if frame1 is not None:
            assert frame1.shape == (self.batch_size, 3, self.height, self.width) or frame1.shape == (self.batch_size, 2, self.height, self.width) # flow b2hw
        assert depth1.shape == (self.batch_size, 1, self.height, self.width)
        assert extrinsic1.shape == (self.batch_size, 4, 4)
        assert extrinsic2.shape == (self.batch_size, 4, 4)
        assert intrinsic1.shape == (self.batch_size, 3, 3)
        assert intrinsic2.shape == (self.batch_size, 3, 3)
        
        """
        计算相机1中每个像素在相机2视角下的齐次投影坐标；
        将齐次投影坐标转换为相机2的像素坐标: 透视除法--分子表示相机2视角下的齐次像素坐标前两位，分母表示齐次坐标深度分量 (避免除0，加了微小值);
        得到的是每个像素对应相机2中的(x,y)坐标
        """
        trans_points1 = self.compute_transformed_points(depth1, extrinsic1, extrinsic2, intrinsic1, intrinsic2)
        trans_coordinates = trans_points1[:, :, :, :2, 0] / (trans_points1[:, :, :, 2:3, 0]+1e-7) # 得到相机2视角下的深度值
        
        """
        计算相机1到相机2的光流，flow12是相机1像素到相机2像素的位移向量(dx, dy)=(x2-x1, y2-y1);
        这里trans_coordinates表示根据相机外参得到的原始坐标在frame2中的位置；grid表示frame1的位置
        """
        grid = self.create_grid(self.batch_size, self.height, self.width, dtype=trans_points1.dtype, device=trans_points1.device) # 生成相机1的像素坐标网格
        grflow = rearrange(trans_coordinates, "b h w c -> b c h w") - grid # shape: torch.Size([1, 2, resized_height, resized_width])

        if frame1 is not None:
            trans_depth1 = rearrange(trans_points1[:, :, :, 2:3, 0], "b h w c -> b c h w")
            warped_frame2 = self.bilinear_splatting(frame1, trans_depth1, grflow, is_image=is_image) # warping frame

        return grflow

    def transformation_smoothing(self, transformation):
        smoothed_trans = []
        for i in range(self.batch_size):
            T = transformation[i]
            xi = se3_log(T)

            if self.prev_xi is None:
                xi_smooth = xi
            else:
                xi_smooth = self.alpha * xi + (1 - self.alpha) * self.prev_xi

            T_smooth = se3_exp(xi_smooth)
            smoothed_trans.append(T_smooth)
            self.prev_xi = xi_smooth
        
        transformation = torch.stack(smoothed_trans, dim=0) # (b,4,4)
        
        return transformation

    def compute_transformed_points(self, depth1: torch.Tensor, extrinsic1: torch.Tensor, extrinsic2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor], flow1: Optional[torch.Tensor]=None):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        
        transformation = torch.bmm(extrinsic2, torch.linalg.inv(extrinsic1))  # (b, 4, 4)

        transformation = scale_rotation_matrix(transformation, alpha=1.0) # Dynamic scaling rotation.
        transformation = self.transformation_smoothing(transformation) # Transformation Smoothing.

        x1d = torch.arange(0, self.width)[None].contiguous().to(transformation)
        y1d = torch.arange(0, self.height)[:, None].contiguous().to(transformation)
        x2d = x1d.repeat([self.height, 1])  # (h, w)
        y2d = y1d.repeat([1, self.width])  # (h, w)
        
        ones_2d = torch.ones(size=(self.height, self.width)).contiguous().to(transformation) # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([self.batch_size, 1, 1, 1, 1])  # (b, h, w, 1, 1)
        
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1) # torch.Size([1, resized_height, resized_width, 1, 1])
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)
        
        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        
        """
        Step 1: 将相机1的3D点转换为齐次坐标，即拼接ones_4d，拼结后，每个3D点边为[x1, y1, z1, 1]，维度=(b, h, w, 4, 1);
        Step 2: 用外参矩阵Trans_4d做乘法，得到相机2的齐次3D点;
        Step 3: 还原为非齐次3D点(取前3位);
        """
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)

        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)

        return trans_norm_points

    def bilinear_splatting(self, frame1: torch.Tensor, depth1: torch.Tensor, flow12: torch.Tensor, is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape

        grid = self.create_grid(b, h, w, dtype=frame1.dtype, device=frame1.device)
        trans_pos = flow12 + grid # 每个像素加上光流, 得到在目标帧中的新位置

        """
        variables functions:
        """
        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long() # 向下取整
        trans_pos_ceil = torch.ceil(trans_pos_offset).long() # 向上取整
        
        """
        torch.clamp()函数用于限幅: 按照min和max的值, 将小于min的值全部拉到min, 将大于max到值全都降低到max
        作用: 将计算得到的光流限制在当前h和w的区间内;
        """
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        """
        双线性权重插值计算，对trans_pos，计算四个邻近点: 左上、右上、左下、右下; 根据偏移量计算这四个方向的插值权重
        trans_pos_offet的作用: 除了加1避免越界，还用于双线性插值权重: (trans_pos_offset - trans_pos_floor) 就是 小数部分 (dx, dy)
        """
        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000) # 我们的depth值还是0.5
        log_depth1 = torch.log(1 + sat_depth1) # 对深度做对数
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50) # 指数进行缩放,越近的点权重越大

        """
        最终的splatting权重=插值权重*掩码*(深度倒数)
        """
        weight_nw = torch.moveaxis(prox_weight_nw / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).contiguous().to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).contiguous().to(frame1)

        """
        散射累积: 用index_put_把源像素值分配到目标像素的四个邻居，并加权累积
        同时对权重做同样的累积，方便后续归一化
        """
        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].contiguous().to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        return cropped_warped_frame


    @staticmethod
    def create_grid(b, h, w, dtype, device):
        x_1d = torch.arange(0, w)[None].contiguous().to(dtype=dtype, device=device)
        y_1d = torch.arange(0, h)[:, None].contiguous().to(dtype=dtype, device=device)
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid
