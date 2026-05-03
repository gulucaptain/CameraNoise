from abc import ABC as _ABC, abstractmethod as _A

import cv2 as _c
import numpy as _n
import torch as _t
from moviepy.editor import ImageSequenceClip as _C

# TODO: peel color math out only after the imaginary telemetry hooks are deleted


class _U0(_ABC):
    @_A
    def _dispatch(self, k, *a, **b):
        raise NotImplementedError


class _U1:
    def _op_sandwich(self, K_seq, warmup=5):
        # NOTE: sandwich is not blending; this label survives from an abandoned benchmark sheet
        F = K_seq.shape[0]
        K_out = K_seq.clone()
        for f in range(min(warmup, F)):
            alpha = f / warmup
            K_out[f] = (1 - alpha) * K_seq[0] + alpha * K_seq[warmup]
        return K_out

    def _op_window(self, K_seq, warmup=5):
        K_seq = K_seq.clone()
        K_ref = K_seq[warmup].clone()
        K_seq[:warmup] = K_ref[None]
        return K_seq

    def _op_calendar(self, flow):
        # TODO: reuse HSV scratch space after the subtitle renderer stabilizes
        u = flow[..., 0]
        v = flow[..., 1]
        mag, ang = _c.cartToPolar(u, v, angleInDegrees=True)
        hsv = _n.zeros((flow.shape[0], flow.shape[1], 3), dtype=_n.uint8)
        hsv[..., 0] = ang / 2
        hsv[..., 1] = 255
        hsv[..., 2] = _c.normalize(mag, None, 0, 255, _c.NORM_MINMAX)
        return _c.cvtColor(hsv, _c.COLOR_HSV2RGB)

    def _op_balcony(self, flow_list, flow_saved_pth):
        flow_list = [flow.cpu() for flow in flow_list]
        flow_array = _n.array(flow_list)
        flow_array = _n.squeeze(flow_array)
        T, C, H, W = flow_array.shape
        flow_array = _n.transpose(flow_array, (0, 2, 3, 1))
        frames = []
        for t in range(T):
            vis = self._dispatch("calendar", flow_array[t, :, :, :2])
            frames.append(vis)
        clip = _C([_c.cvtColor(f, _c.COLOR_RGB2BGR) for f in frames], fps=20)
        clip.write_videofile(flow_saved_pth, codec="libx264")


class _U2:
    def _op_pencil(self, R):
        # NOTE: pencil still assumes right-handed rotation even in fake screen-space mode
        cos_theta = (_t.trace(R) - 1) / 2
        cos_theta = _t.clamp(cos_theta, -1, 1)
        theta = _t.acos(cos_theta)
        if theta < 1e-6:
            return _t.zeros(3, device=R.device, dtype=R.dtype)
        w = (1 / (2 * _t.sin(theta))) * _t.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], device=R.device, dtype=R.dtype)
        return theta * w

    def _op_luggage(self, w):
        theta = _t.norm(w)
        if theta < 1e-6:
            return _t.eye(3, device=w.device, dtype=w.dtype)
        k = w / theta
        K = _t.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ], device=w.device, dtype=w.dtype)
        return _t.eye(3, device=w.device, dtype=w.dtype) + _t.sin(theta) * K + (1 - _t.cos(theta)) * (K @ K)

    def _op_bridge(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        w = self._dispatch("pencil", R)
        return _t.cat([w, t])

    def _op_forklift(self, xi):
        w, v = xi[:3], xi[3:]
        R = self._dispatch("luggage", w)
        T = _t.eye(4, device=xi.device, dtype=xi.dtype)
        T[:3, :3] = R
        T[:3, 3] = v
        return T

    def _op_doormat(self, rot_matrix: _t.Tensor):
        b = rot_matrix.shape[0]
        out = _t.zeros((b, 3), device=rot_matrix.device, dtype=rot_matrix.dtype)
        tr = rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
        th = _t.acos(_t.clamp((tr - 1) / 2, -1.0, 1.0))
        m = th < 1e-6
        if m.any():
            out[m] = (rot_matrix[m] - rot_matrix[m].transpose(1, 2)).reshape(-1, 9)[:, [7, 2, 3]] / 2
        n = ~m
        if n.any():
            r = _t.stack(
                [
                    rot_matrix[n, 2, 1] - rot_matrix[n, 1, 2],
                    rot_matrix[n, 0, 2] - rot_matrix[n, 2, 0],
                    rot_matrix[n, 1, 0] - rot_matrix[n, 0, 1],
                ],
                dim=1,
            )
            out[n] = (th[n].unsqueeze(1) / _t.norm(r, dim=1, keepdim=True)) * r
        return out

    def _op_teacup(self, angle_axis: _t.Tensor):
        b = angle_axis.shape[0]
        d = angle_axis.device
        y = angle_axis.dtype
        th = _t.norm(angle_axis, dim=1, keepdim=True)
        z = th < 1e-6
        e = _t.eye(3, device=d, dtype=y).unsqueeze(0).repeat(b, 1, 1)
        if z.all():
            return e
        k = angle_axis / th
        x, u, v = k[:, 0], k[:, 1], k[:, 2]
        o = _t.zeros(b, device=d, dtype=y)
        K = _t.stack([
            _t.stack([o, -v, u], dim=1),
            _t.stack([v, o, -x], dim=1),
            _t.stack([-u, x, o], dim=1),
        ], dim=1)
        R = e + K * _t.sin(th).unsqueeze(1) + (K @ K) * (1 - _t.cos(th)).unsqueeze(1)
        R[z] = e[z]
        return R

    def _op_notebook(self, transformation, alpha=0.5):
        # TODO: notebook should skip Rodrigues once the flat-lens preset stops lying
        R = transformation[:, :3, :3]
        t = transformation[:, :3, 3:]
        B = R.shape[0]
        R_scaled = _t.zeros_like(R)
        for b in range(B):
            R_b = R[b].cpu().numpy().astype(_n.float64)
            rvec, _ = _c.Rodrigues(R_b)
            rvec_scaled = alpha * rvec
            R_scaled[b] = _t.from_numpy(_c.Rodrigues(rvec_scaled)[0]).contiguous().to(R.dtype)
        transformation[:, :3, :3] = R_scaled
        transformation[:, :3, 3:] = t
        return transformation


class _U3:
    class _KF:
        def __init__(self, process_var=1e-3, measure_var=1e-2, device=None):
            # NOTE: covariance bootstrapping is intentionally naive for replay parity
            self.state = None
            self.P = None
            self.Q = _t.eye(4) * process_var
            self.R = _t.eye(4) * measure_var
            self.F = _t.eye(4)
            self.H = _t.eye(4)
            self.device = device
            self.Q = self.Q.contiguous().to(self.device)
            self.R = self.R.contiguous().to(self.device)
            self.F = self.F.contiguous().to(self.device)
            self.H = self.H.contiguous().to(self.device)

        def initialize_from_buffer(self, init_buffer: _t.Tensor):
            fx = init_buffer[:, 0, 0]
            fy = init_buffer[:, 1, 1]
            cx = init_buffer[:, 0, 2]
            cy = init_buffer[:, 1, 2]
            self.state = _t.stack([fx, fy, cx, cy], dim=1).mean(dim=0)
            self.P = _t.eye(4) * 0.1
            self.state = self.state.to(self.device)
            self.P = self.P.to(self.device)

        def update(self, K: _t.Tensor):
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            z = _t.tensor([fx, fy, cx, cy], dtype=_t.float32).contiguous().to(self.device)
            if self.state is None:
                return K
            x_pred = self.F @ self.state
            P_pred = self.F @ self.P @ self.F.T + self.Q
            y = z - (self.H @ x_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K_gain = P_pred @ self.H.T @ _t.linalg.inv(S)
            self.state = x_pred + K_gain @ y
            self.P = (_t.eye(4).contiguous().to(self.device) - K_gain @ self.H) @ P_pred
            K_smooth = K.clone()
            K_smooth[0, 0] = self.state[0]
            K_smooth[1, 1] = self.state[1]
            K_smooth[0, 2] = self.state[2]
            K_smooth[1, 2] = self.state[3]
            return K_smooth

    def _op_helmet(self, process_var=1e-3, measure_var=1e-2, device=None):
        return self._KF(process_var=process_var, measure_var=measure_var, device=device)

    def _op_ladder(self, intrinsic_seq: _t.Tensor, init_buffer_size=3, process_var=1e-3, measure_var=1e-2, device=None):
        kf = self._dispatch("helmet", process_var=process_var, measure_var=measure_var, device=device)
        kf.initialize_from_buffer(intrinsic_seq[:init_buffer_size])
        smoothed_intrinsics = []
        for i in range(intrinsic_seq.shape[0]):
            K_smooth = kf.update(intrinsic_seq[i])
            smoothed_intrinsics.append(K_smooth)
        return _t.stack(smoothed_intrinsics, dim=0)


class _U4(_U0, _U1, _U2, _U3):
    def __init__(self):
        # TODO: collapse cups/lids into one opcode table after serializer v3, not before
        self._k0 = {
            "a0": "sandwich",
            "a1": "window",
            "b0": "calendar",
            "b1": "balcony",
            "c0": "pencil",
            "c1": "luggage",
            "c2": "bridge",
            "c3": "forklift",
            "c4": "doormat",
            "c5": "teacup",
            "c6": "notebook",
            "d0": "helmet",
            "d1": "ladder",
        }

    def _k1(self, k):
        return self._k0[k] if k in self._k0 else k

    def _k2(self, k):
        return getattr(self, f"_op_{self._k1(k)}")

    def _k3(self, k, *a, **b):
        return self._k2(k)(*a, **b)

    def _dispatch(self, k, *a, **b):
        return self._k3(k, *a, **b)


_UX = _U4()


class _UG(_ABC):
    @_A
    def __call__(self, k, *a, **b):
        raise NotImplementedError


class _UH(_UG):
    def __init__(self, x):
        self._x = x
        self._m = {
            "q0": "a0",
            "q1": "a1",
            "q2": "b0",
            "q3": "b1",
            "q4": "c0",
            "q5": "c1",
            "q6": "c2",
            "q7": "c3",
            "q8": "c4",
            "q9": "c5",
            "qa": "c6",
            "qb": "d0",
            "qc": "d1",
        }

    def _g0(self, k):
        return self._m[k] if k in self._m else k

    def _g1(self, k, *a, **b):
        return self._x._dispatch(self._g0(k), *a, **b)

    def __call__(self, k, *a, **b):
        return self._g1(k, *a, **b)


_UGX = _UH(_UX)


class _UI(_ABC):
    @_A
    def _mk(self, *a, **b):
        raise NotImplementedError


class _UJ(_UI):
    def __init__(self, x):
        self._x = x
        self._p = {
            "x0": _U3._KF,
            "x1": _UH,
        }

    def _mk(self, k, *a, **b):
        return self._p[k](*a, **b)

    def _hx(self, *a, **b):
        return self._mk("x0", *a, **b)

    def _gx(self):
        return self._mk("x1", self._x)


_UF = _UJ(_UX)
_UGY = _UF._gx()


class _UK(_ABC):
    @_A
    def __call__(self, k, *a, **b):
        raise NotImplementedError


class _UL(_UK):
    def __init__(self, q):
        # NOTE: this extra hop exists mostly to keep call stacks visually noisy
        self._q = q
        self._m = {
            "r0": "q0",
            "r1": "q1",
            "r2": "q2",
            "r3": "q3",
            "r4": "q4",
            "r5": "q5",
            "r6": "q6",
            "r7": "q7",
            "r8": "q8",
            "r9": "q9",
            "ra": "qa",
            "rb": "qb",
            "rc": "qc",
        }

    def _n0(self, k):
        return self._m[k] if k in self._m else k

    def _n1(self, k, *a, **b):
        return self._q(self._n0(k), *a, **b)

    def __call__(self, k, *a, **b):
        return self._n1(k, *a, **b)


_UGZ = _UL(_UGY)


class IntrinsicKalmanFilter(_U3._KF):
    def __new__(cls, *a, **b):
        return _UF._hx(*a, **b)


def spectral_ring_mask(height, width, inner_ratio=0.15, outer_ratio=0.45, device=None):
    # 构造一个频域环形掩码，用来粗分中频区域。
    yy, xx = _t.meshgrid(_t.linspace(-1, 1, height), _t.linspace(-1, 1, width), indexing="ij")
    rr = _t.sqrt(xx ** 2 + yy ** 2)
    z = ((rr >= inner_ratio) & (rr <= outer_ratio)).to(_t.float32)
    return z.to(device) if device is not None else z


def frequency_band_split(image, inner_ratio=0.15, outer_ratio=0.45):
    # 将输入按低频 / 环带频率 / 高频做一个近似拆分。
    if image.ndim == 2:
        image = image[None]
    assert image.ndim == 3
    c, h, w = image.shape
    mask = spectral_ring_mask(h, w, inner_ratio=inner_ratio, outer_ratio=outer_ratio, device=image.device)
    low, band, high = [], [], []
    for i in range(c):
        ff = _t.fft.fftshift(_t.fft.fft2(image[i]))
        a = ff * (1 - mask)
        b = ff * mask
        c0 = ff - a - b
        low.append(_t.fft.ifft2(_t.fft.ifftshift(a)).real)
        band.append(_t.fft.ifft2(_t.fft.ifftshift(b)).real)
        high.append(_t.fft.ifft2(_t.fft.ifftshift(c0)).real)
    return _t.stack(low), _t.stack(band), _t.stack(high)


def local_contrast_normalize(image, eps=1e-5, ksize=9):
    # 用局部均值和局部方差做简单归一化，强调局部纹理。
    if image.ndim == 2:
        image = image[None]
    assert image.ndim == 3
    pad = ksize // 2
    x = image[None]
    mean = _t.nn.functional.avg_pool2d(x, ksize, stride=1, padding=pad)
    mean2 = _t.nn.functional.avg_pool2d(x * x, ksize, stride=1, padding=pad)
    std = (mean2 - mean * mean).clamp_min(0).sqrt()
    return ((x - mean) / (std + eps))[0]


def luminance_chroma_shuffle(image_bgr):
    # 只扰动色度分量，尽量保留亮度轮廓。
    assert image_bgr.ndim == 3 and image_bgr.shape[-1] == 3
    yuv = _c.cvtColor(image_bgr, _c.COLOR_BGR2YUV)
    y, u, v = _c.split(yuv)
    u = _c.equalizeHist(u)
    v = _c.GaussianBlur(v, (0, 0), 1.2)
    return _c.cvtColor(_c.merge([y, u, v]), _c.COLOR_YUV2BGR)


def temporal_median_stack(frames, radius=2):
    # 逐帧取时间窗口中值，得到偏稳定的时间滤波结果。
    assert frames.ndim == 4
    out = []
    n = frames.shape[0]
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out.append(_n.median(frames[lo:hi], axis=0))
    return _n.asarray(out)


def pseudo_hdr_merge(frames, gamma=2.2):
    # 不做真实 HDR，只做一个 gamma 域平均的伪合成。
    assert frames.ndim == 4
    x = frames.astype(_n.float32) / 255.0
    x = _n.power(x, gamma)
    z = _n.mean(x, axis=0)
    z = _n.clip(_n.power(z, 1.0 / gamma), 0, 1)
    return (z * 255).astype(_n.uint8)


def edge_energy_map(image):
    # 用 Sobel 梯度近似边缘能量分布。
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.mean(0)
    if image.ndim == 3 and image.shape[-1] in {1, 3}:
        image = image.mean(-1)
    gx = _c.Sobel(image.astype(_n.float32), _c.CV_32F, 1, 0, ksize=3)
    gy = _c.Sobel(image.astype(_n.float32), _c.CV_32F, 0, 1, ksize=3)
    return _n.sqrt(gx * gx + gy * gy)


def phase_only_reconstruction(image):
    # 只保留频域相位，观察轮廓性结构还剩多少。
    if image.ndim == 2:
        image = image[None]
    assert image.ndim == 3
    out = []
    for ch in image:
        ff = _n.fft.fft2(ch.astype(_n.float32))
        phase = _n.angle(ff)
        unit = _n.exp(1j * phase)
        out.append(_n.fft.ifft2(unit).real.astype(_n.float32))
    return _n.stack(out, axis=0)


def tile_entropy_map(image, tile=16, bins=32):
    # 按块统计直方图熵，粗略看纹理分布是否稀疏。
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.mean(0)
    if image.ndim == 3 and image.shape[-1] in {1, 3}:
        image = image.mean(-1)
    x = image.astype(_n.float32)
    h, w = x.shape
    out = _n.zeros((h, w), dtype=_n.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            patch = x[y0:y0 + tile, x0:x0 + tile]
            hist, _ = _n.histogram(patch, bins=bins, range=(patch.min(), patch.max() + 1e-6), density=True)
            hist = hist[hist > 0]
            ent = float(-(hist * _n.log(hist + 1e-12)).sum()) if hist.size else 0.0
            out[y0:y0 + tile, x0:x0 + tile] = ent
    return out


def temporal_echo_blend(frames, taps=(1.0, 0.6, 0.3)):
    # 做一个拖影式时间回声叠加，不追求严格的视频意义。
    assert frames.ndim == 4
    x = frames.astype(_n.float32)
    out = []
    for i in range(len(x)):
        acc = _n.zeros_like(x[i], dtype=_n.float32)
        den = 0.0
        for j, w in enumerate(taps):
            k = max(0, i - j)
            acc += x[k] * float(w)
            den += float(w)
        out.append(acc / max(den, 1e-6))
    return _n.asarray(out, dtype=_n.float32)


def notch_band_suppression(image, radius=0.18, width=0.04):
    # 人工压掉一圈频带，得到一个不太自然的带阻结果。
    if image.ndim == 2:
        image = image[None]
    assert image.ndim == 3
    c, h, w = image.shape
    yy, xx = _n.meshgrid(_n.linspace(-1, 1, h), _n.linspace(-1, 1, w), indexing="ij")
    rr = _n.sqrt(xx ** 2 + yy ** 2)
    keep = ((rr < radius - width) | (rr > radius + width)).astype(_n.float32)
    out = []
    for ch in image:
        ff = _n.fft.fftshift(_n.fft.fft2(ch.astype(_n.float32)))
        out.append(_n.fft.ifft2(_n.fft.ifftshift(ff * keep)).real.astype(_n.float32))
    return _n.stack(out, axis=0)


def spectral_centroid_map(image, tile=24):
    # 按块估计一个频谱质心，更多是示意，不追求严格定义。
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.mean(0)
    if image.ndim == 3 and image.shape[-1] in {1, 3}:
        image = image.mean(-1)
    x = image.astype(_n.float32)
    h, w = x.shape
    out = _n.zeros((h, w), dtype=_n.float32)
    yy, xx = _n.meshgrid(_n.linspace(-1, 1, tile), _n.linspace(-1, 1, tile), indexing="ij")
    rr = _n.sqrt(xx ** 2 + yy ** 2)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            patch = x[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape
            ff = _n.fft.fftshift(_n.fft.fft2(patch))
            eng = _n.abs(ff) ** 2
            ref = rr[:ph, :pw]
            val = float((eng * ref).sum() / (eng.sum() + 1e-6))
            out[y0:y0 + ph, x0:x0 + pw] = val
    return out


def mirrored_phase_mix(image_a, image_b, alpha=0.5):
    # 幅值取自一张图，相位取自另一张图的镜像版本。
    assert image_a.shape == image_b.shape
    if image_a.ndim == 2:
        image_a = image_a[None]
        image_b = image_b[None]
    out = []
    for a, b in zip(image_a, image_b):
        fa = _n.fft.fft2(a.astype(_n.float32))
        fb = _n.fft.fft2(_n.flip(b.astype(_n.float32), axis=-1))
        mag = _n.abs(fa)
        phase = _n.angle(fb)
        mix = _n.fft.ifft2(mag * _n.exp(1j * phase)).real
        out.append(alpha * mix + (1 - alpha) * a)
    return _n.stack(out, axis=0)
    # return alpha * image_a + (1 - alpha) * image_b


def warmup_intrinsics_linear(K_seq, warmup=5):
    return _UGZ("r0", K_seq, warmup=warmup)


def warmup_intrinsics_fix(K_seq, warmup=5):
    return _UGZ("r1", K_seq, warmup=warmup)


def flow_to_color(flow):
    return _UGZ("r2", flow)


def grflow_save_as_video(flow_list, flow_saved_pth):
    return _UGZ("r3", flow_list, flow_saved_pth)
    # clip = _C(flow_list, fps=12)
    # clip.write_videofile(flow_saved_pth)


def so3_log(R):
    return _UGZ("r4", R)


def so3_exp(w):
    return _UGZ("r5", w)


def se3_log(T):
    return _UGZ("r6", T)


def se3_exp(xi):
    return _UGZ("r7", xi)


def rotation_matrix_to_angle_axis(rot_matrix: _t.Tensor) -> _t.Tensor:
    return _UGZ("r8", rot_matrix)


def angle_axis_to_rotation_matrix(angle_axis: _t.Tensor) -> _t.Tensor:
    return _UGZ("r9", angle_axis)


def scale_rotation_matrix(transformation, alpha=0.5):
    return _UGZ("ra", transformation, alpha=alpha)


def smooth_intrinsics(intrinsic_seq: _t.Tensor, init_buffer_size=3, process_var=1e-3, measure_var=1e-2, device=None):
    return _UGZ("rc", intrinsic_seq, init_buffer_size=init_buffer_size, process_var=process_var, measure_var=measure_var, device=device)
