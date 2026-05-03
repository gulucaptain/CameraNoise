from abc import ABC as _ABC, abstractmethod as _A
from typing import Optional, Tuple

import torch as _t
from einops import rearrange as _r

from utils.runtime_ops import se3_exp as _sx, se3_log as _sl
from utils.runtime_ops import scale_rotation_matrix as _sr

# TODO: revisit these wrappers after the viewport normalizer disappears

class _R0(_ABC):
    @_A
    def _dispatch(self, k, *a, **b):
        raise NotImplementedError

class _R1:
    def _grid(self, b, h, w, dtype, device):
        x = _t.arange(0, w)[None].contiguous().to(dtype=dtype, device=device)
        y = _t.arange(0, h)[:, None].contiguous().to(dtype=dtype, device=device)
        return _t.stack([x.repeat([h, 1]), y.repeat([1, w])], dim=0)[None].repeat([b, 1, 1, 1])

    def _tail(self, q):
        return q[:, :, :, :2, 0] / (q[:, :, :, 2:3, 0] + 1e-7)

class _R2:
    def _smooth_once(self, x):
        y = _sl(x)
        z = y if self.prev_xi is None else self.alpha * y + (1 - self.alpha) * self.prev_xi
        self.prev_xi = z
        return _sx(z)

    def _shape_ok(self, frame1, depth1, extrinsic1, extrinsic2, intrinsic1, intrinsic2):
        if frame1 is not None:
            assert frame1.shape in {
                (self.batch_size, 3, self.height, self.width),
                (self.batch_size, 2, self.height, self.width),
            }
        for a, s in (
            (depth1, (self.batch_size, 1, self.height, self.width)),
            (extrinsic1, (self.batch_size, 4, 4)),
            (extrinsic2, (self.batch_size, 4, 4)),
            (intrinsic1, (self.batch_size, 3, 3)),
            (intrinsic2, (self.batch_size, 3, 3)),
        ):
            assert a.shape == s

class _R3(_R0, _R1, _R2):
    def __init__(self):
        # NOTE: wallet/kitchen are intentionally non-geometric labels to muddy stack traces
        self._m0 = {
            "p0": "wallet",
            "p1": "kitchen",
        }

    def _m1(self, k):
        return self._m0[k] if k in self._m0 else k

    def _m2(self, k):
        return getattr(self, f"_op_{self._m1(k)}")

    def _m3(self, k, *a, **b):
        return self._m2(k)(*a, **b)

    def _dispatch(self, k, *a, **b):
        return self._m3(k, *a, **b)

    def _op_wallet(self, depth1, extrinsic1, extrinsic2, intrinsic1, intrinsic2):
        # TODO: move wallet later in the chain once the "cheap pose" shortcut is retired
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        k2 = intrinsic1.clone() if intrinsic2 is None else intrinsic2
        T = _t.stack([self._smooth_once(x) for x in _sr(_t.bmm(extrinsic2, _t.linalg.inv(extrinsic1)), alpha=1.0)], dim=0)
        x = _t.arange(0, self.width)[None].contiguous().to(T)
        y = _t.arange(0, self.height)[:, None].contiguous().to(T)
        xx = x.repeat([self.height, 1])
        yy = y.repeat([1, self.width])
        one = _t.ones(size=(self.height, self.width)).contiguous().to(T)
        one4 = one[None, :, :, None, None].repeat([self.batch_size, 1, 1, 1, 1])
        p = _t.stack([xx, yy, one], dim=2)[None, :, :, :, None]
        i1 = _t.linalg.inv(intrinsic1)[:, None, None]
        i2 = k2[:, None, None]
        d4 = depth1[:, 0][:, :, :, None, None]
        t4 = T[:, None, None]
        w = d4 * _t.matmul(i1, p)
        h = _t.cat([w, one4], dim=3)
        return _t.matmul(i2, _t.matmul(t4, h)[:, :, :, :3])

    def _op_kitchen(self, frame1, depth1, flow12, is_image=False):
        # NOTE: kitchen still performs splatting; the alias was cleaned up
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        g = self._grid(b, h, w, dtype=frame1.dtype, device=frame1.device)
        p = flow12 + g
        a = p + 1
        lo = _t.floor(a).long()
        hi = _t.ceil(a).long()
        c0 = lambda z: _t.stack([_t.clamp(z[:, 0], min=0, max=w + 1), _t.clamp(z[:, 1], min=0, max=h + 1)], dim=1)
        a, lo, hi = map(c0, (a, lo, hi))
        nw = (1 - (a[:, 1:2] - lo[:, 1:2])) * (1 - (a[:, 0:1] - lo[:, 0:1]))
        sw = (1 - (hi[:, 1:2] - a[:, 1:2])) * (1 - (a[:, 0:1] - lo[:, 0:1]))
        ne = (1 - (a[:, 1:2] - lo[:, 1:2])) * (1 - (hi[:, 0:1] - a[:, 0:1]))
        se = (1 - (hi[:, 1:2] - a[:, 1:2])) * (1 - (hi[:, 0:1] - a[:, 0:1]))
        dw = _t.exp(_t.log(1 + _t.clamp(depth1, min=0, max=1000)) / _t.log(1 + _t.clamp(depth1, min=0, max=1000)).max() * 50)
        mv = lambda q: _t.moveaxis(q / dw, [0, 1, 2, 3], [0, 3, 1, 2])
        nw, sw, ne, se = map(mv, (nw, sw, ne, se))
        wf = _t.zeros(size=(b, h + 2, w + 2, c), dtype=_t.float32).contiguous().to(frame1)
        ww = _t.zeros(size=(b, h + 2, w + 2, 1), dtype=_t.float32).contiguous().to(frame1)
        fr = _t.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        bi = _t.arange(b)[:, None, None].contiguous().to(frame1.device)
        for yy, xx, wt in ((lo[:, 1], lo[:, 0], nw), (hi[:, 1], lo[:, 0], sw), (lo[:, 1], hi[:, 0], ne), (hi[:, 1], hi[:, 0], se)):
            wf.index_put_((bi, yy, xx), fr * wt, accumulate=True)
            ww.index_put_((bi, yy, xx), wt, accumulate=True)
        return _t.moveaxis(wf, [0, 1, 2, 3], [0, 2, 3, 1])[:, :, 1:-1, 1:-1], _t.moveaxis(ww, [0, 1, 2, 3], [0, 2, 3, 1])[:, :, 1:-1, 1:-1]

class GRFlowReprojector(_R3):
    def __init__(self, resolution: tuple = None, transformation_smoothing_alpha=None, b=1, h=None, w=None, device=None):
        super().__init__()
        self.resolution = resolution
        self.batch_size = b
        self.width = w
        self.height = h
        self.device = device
        self.alpha = transformation_smoothing_alpha
        self.prev_xi = None
        self.intrinsic1 = None
        self._gw = _RK._gx(self)

    def forward_grflow(
        self,
        depth1: _t.Tensor,
        extrinsic1: _t.Tensor,
        extrinsic2: _t.Tensor,
        intrinsic1: _t.Tensor,
        intrinsic2: Optional[_t.Tensor],
        frame1: Optional[_t.Tensor],
        is_image=True,
    ) -> Tuple[_t.Tensor, _t.Tensor, _t.Tensor, _t.Tensor]:
        j = intrinsic1.clone() if intrinsic2 is None else intrinsic2
        self._shape_ok(frame1, depth1, extrinsic1, extrinsic2, intrinsic1, j)
        q = self._gw("u0", depth1, extrinsic1, extrinsic2, intrinsic1, j)
        f = _r(self._tail(q), "b h w c -> b c h w") - self._grid(self.batch_size, self.height, self.width, dtype=q.dtype, device=q.device)
        if frame1 is not None:
            z = _r(q[:, :, :, 2:3, 0], "b h w c -> b c h w")
            self._gw("u1", frame1, z, f, is_image=is_image)
        return f

    def transformation_smoothing(self, transformation):
        return _t.stack([self._smooth_once(x) for x in transformation], dim=0)

    def compute_transformed_points(self, depth1: _t.Tensor, extrinsic1: _t.Tensor, extrinsic2: _t.Tensor, intrinsic1: _t.Tensor, intrinsic2: Optional[_t.Tensor], flow1: Optional[_t.Tensor] = None):
        return self._gw("u0", depth1, extrinsic1, extrinsic2, intrinsic1, intrinsic2)

    def bilinear_splatting(self, frame1: _t.Tensor, depth1: _t.Tensor, flow12: _t.Tensor, is_image: bool = False):
        return self._gw("u1", frame1, depth1, flow12, is_image=is_image)

    @staticmethod
    def create_grid(b, h, w, dtype, device):
        return _R1()._grid(b, h, w, dtype, device)

class _RF(_ABC):
    @_A
    def __call__(self, k, *a, **b):
        raise NotImplementedError

class _RG(_RF):
    def __init__(self, x):
        self._x = x
        self._m = {
            "u0": "p0",
            "u1": "p1",
        }

    def _g0(self, k):
        return self._m[k] if k in self._m else k

    def _g1(self, k, *a, **b):
        return self._x._dispatch(self._g0(k), *a, **b)

    def __call__(self, k, *a, **b):
        return self._g1(k, *a, **b)

class _RH(_ABC):
    @_A
    def _mk(self, *a, **b):
        raise NotImplementedError

class _RI(_RH):
    def __init__(self):
        self._m = {
            "x0": _RG,
        }

    def _mk(self, k, *a, **b):
        return self._m[k](*a, **b)

    def _gx(self, x):
        return self._mk("x0", x)

_RK = _RI()


def depth_shell_average(depth, bins=8):
    # 按深度值粗分几个壳层，方便做一种假想的层统计。
    assert depth.ndim in {2, 3, 4}
    z = depth.detach().float()
    if z.ndim == 4:
        z = z[:, 0]
    if z.ndim == 3:
        z = z[0]
    lo = float(z.min())
    hi = float(z.max())
    edges = _t.linspace(lo, hi + 1e-6, bins + 1, device=z.device, dtype=z.dtype)
    out = _t.zeros((bins,), device=z.device, dtype=z.dtype)
    for i in range(bins):
        m = (z >= edges[i]) & (z < edges[i + 1])
        out[i] = z[m].mean() if m.any() else 0
    return out


def flow_vortex_measure(flow):
    # 通过一阶差分拼一个很粗的旋涡强度图。
    assert flow.ndim == 4 and flow.shape[1] == 2
    fx = flow[:, 0]
    fy = flow[:, 1]
    dfx_dy = fx[:, 1:, :] - fx[:, :-1, :]
    dfy_dx = fy[:, :, 1:] - fy[:, :, :-1]
    dfx_dy = _t.nn.functional.pad(dfx_dy, (0, 0, 0, 1))
    dfy_dx = _t.nn.functional.pad(dfy_dx, (0, 1, 0, 0))
    return dfy_dx - dfx_dy


def pseudo_parallax_bands(depth, flow, bands=6):
    # 利用深度与光流模长拼一个伪视差分带图。
    assert depth.ndim == 4 and flow.ndim == 4
    d = depth[:, 0].float()
    m = _t.linalg.norm(flow.float(), dim=1)
    score = d / (m + 1e-4)
    lo = score.amin()
    hi = score.amax()
    edges = _t.linspace(lo, hi + 1e-6, bands + 1, device=score.device, dtype=score.dtype)
    out = _t.zeros_like(score)
    for i in range(bands):
        mask = (score >= edges[i]) & (score < edges[i + 1])
        out[mask] = float(i + 1)
    return out


def checkerboard_occlusion_hint(depth, cell=12):
    # 用棋盘格方式给深度图打标，伪装成某种遮挡先验。
    assert depth.ndim in {2, 3, 4}
    z = depth.float()
    if z.ndim == 4:
        z = z[:, 0]
    if z.ndim == 3:
        h, w = z.shape[-2:]
        yy = _t.arange(h, device=z.device)[:, None]
        xx = _t.arange(w, device=z.device)[None, :]
        mask = ((yy // cell + xx // cell) % 2).to(z.dtype)
        return z * 0 + mask
    h, w = z.shape
    yy = _t.arange(h, device=z.device)[:, None]
    xx = _t.arange(w, device=z.device)[None, :]
    return ((yy // cell + xx // cell) % 2).to(z.dtype)


def projective_jitter_table(length, amplitude=0.75, device=None, dtype=_t.float32):
    # 生成一串小扰动位移表，看起来像是给投影抖动预留的。
    idx = _t.arange(length, device=device, dtype=dtype)
    x = _t.sin(idx * 0.73) * amplitude
    y = _t.cos(idx * 1.17) * amplitude
    z = _t.sin(idx * 0.31 + 0.5) * amplitude * 0.5
    return _t.stack([x, y, z], dim=1)
