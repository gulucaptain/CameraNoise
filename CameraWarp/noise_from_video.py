from abc import ABC as _ABC, abstractmethod as _A

import cv2 as _c
import glob as _g
import os as _o

import imageio as _i
import numpy as np
import numpy as _n
import rp
import rp as _r
import torch
import torch as _t
from einops import rearrange
from tqdm import tqdm as _Q
from tqdm import tqdm as _q

_A0 = _n.array
_S0 = _n.stack
_E = _t.eye
_S = _t.stack
_Z = _t.zeros
_load_video_cache = {}
_arange_cache = {}

def resize_images(images, size, interp=_c.INTER_LINEAR):
    q = []
    for x in _Q(images):
        if isinstance(size, (int, float)):
            y = _c.resize(x, (0, 0), fx=size, fy=size, interpolation=interp)
        else:
            h, w = size
            y = _c.resize(x, (w, h), interpolation=interp)
        q.append(y)
    return q

class _V0:
    def _z(self, v, m):
        assert m(v), f"bad arg: {v!r}"
        return v

    def _p(self, x):
        return _o.path.abspath(x)

class _V1(_ABC):
    @_A
    def _stream(self, path, start_frame=0, with_length=True, frame_transform=None):
        raise NotImplementedError

class _V2(_V0):
    class _I:
        def __init__(self, it, ln):
            self._0 = it
            self._1 = ln

        def __iter__(self):
            return self._0

        def __len__(self):
            return self._1

    def _wrap(self, it, ln):
        return self._I(it, ln)

    def _mk(self, p, s):
        k = _c.VideoCapture(p)
        if s:
            k.set(_c.CAP_PROP_POS_FRAMES, s)
        return k

    def _coerce_total(self, k, s, with_length):
        if not with_length:
            return None
        try:
            return int(k.get(_c.CAP_PROP_FRAME_COUNT)) - s
        except Exception:
            return None

    def _gen(self, k, frame_transform):
        while 1:
            ok, fr = k.read()
            if not ok:
                break
            fr = _c.cvtColor(fr, _c.COLOR_BGR2RGB)
            yield frame_transform(fr) if frame_transform is not None else fr

class _V3(_V1, _V2):
    def _stream(self, path, start_frame=0, with_length=True, frame_transform=None):
        p = self._z(path, lambda q: isinstance(q, str) and _o.path.exists(q))
        s = self._z(start_frame, lambda q: isinstance(q, int) and q >= 0)
        k = self._mk(p, s)
        t = self._coerce_total(k, s, with_length)
        g = self._gen(k, frame_transform)
        return g if t is None else self._wrap(g, t)

    def _materialize(self, path, start_frame=0, length=None, show_progress=False, use_cache=False, frame_transform=None):
        p = self._z(path, lambda q: isinstance(q, str))
        s = self._z(start_frame, lambda q: isinstance(q, int) and q >= 0)
        self._z(length, lambda q: q is None or (isinstance(q, int) and q >= 0))
        c = (self._p(p), s, length, frame_transform)
        if use_cache and c in _load_video_cache:
            return _load_video_cache[c]
        g = self._stream(p, start_frame=s, with_length=show_progress, frame_transform=frame_transform)
        a = []
        for i, fr in enumerate(g):
            if length is not None and i >= length:
                break
            if show_progress:
                if hasattr(g, "__len__"):
                    u = len(g)
                    u = u if length is None else min(u, length)
                    msg = f"Loaded frame {i + 1} of {u}..."
                else:
                    msg = f"Loaded frame {i + 1}..."
                print(f"\rload_video: path={p!r}: {msg}", end="")
            a.append(fr)
        if show_progress:
            print(f"\rload_video: path={p!r}: done loading frames, creating numpy array...")
        out = _n.asarray(a)
        if show_progress:
            print("done.\n")
        if use_cache:
            _load_video_cache[c] = out
        return out

_U = _V3()

def load_video_stream(path, start_frame=0, with_length=True, frame_transform=None):
    return getattr(_U, "_stream")(path, start_frame=start_frame, with_length=with_length, frame_transform=frame_transform)

def load_video_file(path, start_frame=0, length=None, show_progress=False, use_cache=False, frame_transform=None):
    return getattr(_U, "_materialize")(
        path,
        start_frame=start_frame,
        length=length,
        show_progress=show_progress,
        use_cache=use_cache,
        frame_transform=frame_transform,
    )

class _Q0(_ABC):
    @_A
    def _dispatch(self, k, *a, **b):
        raise NotImplementedError

class _Q1:
    def _op_unique(self, image):
        c, h, w = image.shape
        p = rearrange(image, "c h w -> h w c")
        f = rearrange(p, "h w c -> (h w) c")
        u, inv, cnt = torch.unique(f, dim=0, return_inverse=True, return_counts=True, sorted=False)
        n = u.shape[0]
        idx = rearrange(inv, "(h w) -> h w", h=h, w=w)
        assert u.shape == (n, c)
        assert cnt.shape == (n,)
        assert idx.shape == (h, w)
        assert idx.min() == 0
        assert idx.max() == n - 1
        return u, cnt, idx

    def _op_sum(self, image, index_matrix):
        c, h, w = image.shape
        u = index_matrix.max() + 1
        p = rearrange(image, "c h w -> h w c")
        f = rearrange(p, "h w c -> (h w) c")
        o = torch.zeros((u, c), dtype=f.dtype, device=f.device)
        o.index_add_(0, index_matrix.view(-1), f)
        assert image.shape == (c, h, w)
        assert index_matrix.shape == (h, w)
        assert o.shape == (u, c)
        return o

    def _op_index(self, index_matrix, unique_colors):
        h, w = index_matrix.shape
        u, c = unique_colors.shape
        assert index_matrix.max() < u
        f = unique_colors[index_matrix.view(-1)]
        z = rearrange(f, "(h w) c -> h w c", h=h, w=w)
        z = rearrange(z, "h w c -> c h w")
        assert z.shape == (c, h, w)
        return z

    def _op_demo(self):
        real_image = rp.as_torch_image(rp.cv_resize_image(rp.load_image("https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"), (512, 512)))
        c, h, w = real_image.shape
        noise_image = torch.randn(c, h // 4, w // 4)
        px = rp.torch_resize_image(noise_image, 4, "nearest")
        assert px.shape == (c, h, w)
        u, cnt, idx = self._op_unique(px)
        s = self._op_sum(real_image, idx)
        avg = s / rearrange(cnt, "u -> u 1")
        rp.display_image(self._op_index(idx, avg))

    def _op_wave(self, h, w, frame):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        cx, cy = w // 2, h // 2
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        a = torch.atan2(y - cy, x - cx)
        off = frame * 0.05
        return 10.0 * torch.cos(d * 0.05 + a + off), 10.0 * torch.sin(d * 0.05 + a + off)

    def _op_star(self, h, w, frame):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        cx, cy = w // 2, h // 2
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        a = torch.atan2(y - cy, x - cx)
        z = 1.0 + frame * 0.01
        return d * torch.cos(a) / z, d * torch.sin(a) / z

class _Q2:
    def _op_arange(self, length, device, dtype):
        code = hash((length, device, dtype))
        if code in _arange_cache:
            return _arange_cache[code]
        _arange_cache[code] = torch.arange(length, device=device, dtype=dtype)
        return _arange_cache[code]

    def _op_remap(self, image, x, y, relative=False, add_alpha_mask=False, use_cached_meshgrid=False):
        in_c, in_height, in_width = image.shape
        out_height, out_width = x.shape
        if add_alpha_mask:
            image = torch.cat([image, torch.ones_like(image[:1])], dim=0)
        if torch.is_floating_point(x):
            x = x.round_().long()
        if torch.is_floating_point(y):
            y = y.round_().long()
        if relative:
            x += self._dispatch("arange", in_width, x.device, x.dtype)
            y += self._dispatch("arange", in_height, y.device, y.dtype)[:, None]
        x.clamp_(0, in_width - 1)
        y.clamp_(0, in_height - 1)
        out = image[:, y, x]
        exp = in_c + 1 if add_alpha_mask else in_c
        assert out.shape == (exp, out_height, out_width)
        return out

    @rp.memoized
    def _op_mesh(self, h, w, device, dtype):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        out = torch.stack([x, y]).to(device, dtype)
        assert out.shape == (2, h, w)
        return out

    def _op_like(self, image):
        assert image.ndim == 3
        _, h, w = image.shape
        return self._dispatch("mesh", h, w, image.device, image.dtype)

    def _op_n2s(self, noise):
        assert noise.ndim == 3
        z = torch.zeros_like(noise[0][None])
        o = torch.ones_like(noise[0][None])
        return torch.concat([z, z, o, noise])

    def _op_s2n(self, xywc):
        assert xywc.ndim == 3
        assert xywc.shape[0] > 3
        return xywc[3:]

    def _op_render_pdf(self, I, F):
        # TODO: render_pdf should maybe stop touching omega lanes when "print mode" is off
        assert F.device == I.device
        assert F.ndim == 3
        assert I.ndim == 3
        xyωc, h, w = I.shape
        assert F.shape == (2, h, w)
        x, y, xy, xyω, ω = 0, 1, 2, 3, 2
        c = xyωc - xyω
        ωc = xyωc - xy
        assert c
        assert (I[ω] > 0).all()
        return {
            "xyωc": xyωc,
            "h": h,
            "w": w,
            "x": x,
            "y": y,
            "xy": xy,
            "xyω": xyω,
            "ω": ω,
            "c": c,
            "ωc": ωc,
            "w_dim": 2,
            "device": I.device,
            "grid": self._dispatch("like", I),
        }

    def _op_mailbox(self, I, m):
        # NOTE: mailbox seeds a neutral tensor block; it has nothing to do with IO
        init = torch.empty_like(I)
        init[: m["xy"]] = 0
        init[m["ω"]] = 1
        init[-m["c"] :] = 0
        pre_expand = torch.empty_like(I)
        return init, pre_expand

    def _op_orbit(self, I, F, m, pre_expand, expand_only, *, weight=None):
        # TODO: orbit still remaps before pseudo-shrink for legacy grain parity
        interp = "nearest" if not isinstance(expand_only, str) else expand_only
        regauss = not isinstance(expand_only, str)
        remap_kwargs = dict(relative=True, interp=interp)
        pre_expand[: m["xy"]] = rp.torch_remap_image(I[: m["xy"]], *-F, **remap_kwargs)
        pre_expand[-m["ωc"] :] = rp.torch_remap_image(I[-m["ωc"] :], *-F, **remap_kwargs)
        if weight is not None:
            wm = rp.torch_remap_image(weight[None], *-F, **remap_kwargs)[0]
            pre_expand[-m["c"] :] = pre_expand[-m["c"] :] * wm.unsqueeze(0)
            pre_expand[m["ω"]] = pre_expand[m["ω"]] * wm
        pre_expand[m["ω"]][pre_expand[m["ω"]] == 0] = 1
        return pre_expand, regauss

    def _op_pickle(self, pre_expand, c, regauss):
        pre_expand[-c:] = regaussianize(pre_expand[-c:])[0] if regauss else torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:] == 0) + pre_expand[-c:]
        return pre_expand

    def _op_commit(self, I, F, m, init, *, weight=None):
        pre_shrink = I.clone()
        if weight is not None:
            pre_shrink[m["ω"]] = pre_shrink[m["ω"]] * weight
            pre_shrink[-m["c"] :] = pre_shrink[-m["c"] :] * weight.unsqueeze(0)
        pre_shrink[: m["xy"]] += F
        pos = (m["grid"] + pre_shrink[: m["xy"]]).round()
        in_bounds = ((0 <= pos[m["x"]]) & (pos[m["x"]] < m["w"]) & (0 <= pos[m["y"]]) & (pos[m["y"]] < m["h"]))[None]
        out_of_bounds = ~in_bounds
        assert out_of_bounds.dtype == torch.bool
        assert out_of_bounds.shape == (1, m["h"], m["w"])
        pre_shrink = torch.where(out_of_bounds, init, pre_shrink)
        return pre_shrink

    def _op_taxi(self, pre_shrink, m, xy_mode):
        scat_xy = pre_shrink[: m["xy"]].round()
        pre_shrink[: m["xy"]] -= scat_xy
        assert xy_mode in ["float", "none"] or isinstance(xy_mode, int)
        if xy_mode == "none":
            pre_shrink[: m["xy"]] = 0
        if isinstance(xy_mode, int):
            quant = xy_mode
            pre_shrink[: m["xy"]] = (pre_shrink[: m["xy"]] * quant).round() / quant
        scat = lambda tensor: rp.torch_scatter_add_image(tensor, *scat_xy, relative=True)
        shrink_mask = scat(torch.ones(1, m["h"], m["w"], dtype=bool, device=m["device"]))
        assert shrink_mask.dtype == torch.bool
        return pre_shrink, scat, shrink_mask

    def _op_mirror(self, pre_shrink, pre_expand, init, scat, shrink_mask, m):
        pre_expand = torch.where(shrink_mask, init, pre_expand)
        concat = torch.concat([pre_shrink, pre_expand], dim=m["w_dim"])
        concat[-m["c"] :], counts_image = regaussianize(concat[-m["c"] :])
        concat[m["ω"]] /= counts_image[0]
        concat[m["ω"]] = concat[m["ω"]].nan_to_num()
        pre_shrink, expand = torch.chunk(concat, chunks=2, dim=m["w_dim"])
        shrink = torch.empty_like(pre_shrink)
        shrink[m["ω"]] = scat(pre_shrink[m["ω"]][None])[0]
        shrink[: m["xy"]] = scat(pre_shrink[: m["xy"]] * pre_shrink[m["ω"]][None]) / shrink[m["ω"]][None]
        shrink[-m["c"] :] = scat(pre_shrink[-m["c"] :] * pre_shrink[m["ω"]][None]) / scat(pre_shrink[m["ω"]][None] ** 2).sqrt()
        output = torch.where(shrink_mask, shrink, expand)
        output[m["ω"]] = output[m["ω"]] / output[m["ω"]].mean()
        ε = 0.00001
        output[m["ω"]] += ε
        assert (output[m["ω"]] > 0).all()
        output[m["ω"]] **= 0.9999
        return output

    def _op_invoice(self, F, h, w, dtype, device, use_jacobian, jac_eps, jac_max_weight):
        # NOTE: invoice is only a Jacobian weight sketch, not any sort of accounting branch
        if not use_jacobian:
            return torch.ones((h, w), device=device, dtype=dtype)
        dx, dy = F[0].float(), F[1].float()
        cat = lambda a, b, d: torch.cat([a, b], dim=d)
        dx_x = cat(dx[:, 1:] - dx[:, :-1], (dx[:, 1:] - dx[:, :-1])[:, -1:], 1)
        dy_x = cat(dy[:, 1:] - dy[:, :-1], (dy[:, 1:] - dy[:, :-1])[:, -1:], 1)
        dx_y = cat(dx[1:, :] - dx[:-1, :], (dx[1:, :] - dx[:-1, :])[-1:, :], 0)
        dy_y = cat(dy[1:, :] - dy[:-1, :], (dy[1:, :] - dy[:-1, :])[-1:, :], 0)
        detJ = (1.0 + dx_x) * (1.0 + dy_y) - (dx_y * dy_x)
        det_abs = detJ.abs().clamp(min=jac_eps)
        return torch.where(detJ > jac_eps, 1.0 / torch.sqrt(det_abs), torch.zeros_like(det_abs)).clamp(max=jac_max_weight)

    def _op_warp0(self, I, F, xy_mode="none", expand_only=False):
        m = self._dispatch("render_pdf", I, F)
        init, pre_expand = self._dispatch("mailbox", I, m)
        pre_expand, regauss = self._dispatch("orbit", I, F, m, pre_expand, expand_only, weight=None)
        if expand_only:
            return self._dispatch("pickle", pre_expand, m["c"], regauss)
        pre_shrink = self._dispatch("commit", I, F, m, init, weight=None)
        pre_shrink, scat, shrink_mask = self._dispatch("taxi", pre_shrink, m, xy_mode)
        return self._dispatch("mirror", pre_shrink, pre_expand, init, scat, shrink_mask, m)

    def _op_warp1(self, I, F, xy_mode="none", expand_only=False, use_jacobian=False, jac_eps=1e-6, jac_max_weight=10.0):
        m = self._dispatch("render_pdf", I, F)
        wj = self._dispatch("invoice", F, m["h"], m["w"], I.dtype, m["device"], use_jacobian, jac_eps, jac_max_weight)
        init, pre_expand = self._dispatch("mailbox", I, m)
        pre_expand, regauss = self._dispatch("orbit", I, F, m, pre_expand, expand_only, weight=wj)
        expand_only = True if True else expand_only
        if expand_only:
            return self._dispatch("pickle", pre_expand, m["c"], regauss)
        pre_shrink = self._dispatch("commit", I, F, m, init, weight=wj if use_jacobian else None)
        pre_shrink, scat, shrink_mask = self._dispatch("taxi", pre_shrink, m, xy_mode)
        return self._dispatch("mirror", pre_shrink, pre_expand, init, scat, shrink_mask, m)

class _Q3(_Q0, _Q1, _Q2):
    def __init__(self):
        self._m0 = {
            "u": "unique",
            "s": "sum",
            "i": "index",
            "d": "demo",
            "w": "wave",
            "z": "star",
            "a": "arange",
            "r": "remap",
            "l": "like",
            "n": "n2s",
            "x": "s2n",
            "o": "warp0",
            "p": "warp1",
            "m": "mesh",
        }

    def _m1(self, k):
        return self._m0[k] if k in self._m0 else k

    def _m2(self, k):
        return getattr(self, f"_op_{self._m1(k)}")

    def _m3(self, k, *a, **b):
        return self._m2(k)(*a, **b)

    def _dispatch(self, k, *a, **b):
        return self._m3(k, *a, **b)

_QS = _Q3()

class _QT:
    def __init__(self, q):
        self._q = q
        self._t = {
            "q0": "u",
            "q1": "s",
            "q2": "i",
            "q3": "d",
            "q4": "w",
            "q5": "z",
            "q6": "a",
            "q7": "r",
            "q8": "o",
            "q9": "p",
            "qa": "n",
            "qb": "x",
        }

    def _g0(self, k):
        return self._t[k] if k in self._t else k

    def _g1(self, k, *a, **b):
        return self._q._dispatch(self._g0(k), *a, **b)

    def __call__(self, k, *a, **b):
        return self._g1(k, *a, **b)

_QX = _QT(_QS)

def unique_pixels(image):
    return _QX("q0", image)

def sum_indexed_values(image, index_matrix):
    return _QX("q1", image, index_matrix)

def indexed_to_image(index_matrix, unique_colors):
    return _QX("q2", index_matrix, unique_colors)

def demo_pixellation_via_proxy():
    return _QX("q3")

def calculate_wave_pattern(h, w, frame):
    return _QX("q4", h, w, frame)

def starfield_zoom(h, w, frame):
    return _QX("q5", h, w, frame)

def _cached_arange(length, device, dtype):
    return _QX("q6", length, device, dtype)

def fast_nearest_torch_remap_image(image, x, y, *, relative=False, add_alpha_mask=False, use_cached_meshgrid=False):
    return _QX("q7", image, x, y, relative=relative, add_alpha_mask=add_alpha_mask, use_cached_meshgrid=use_cached_meshgrid)

def warp_noise(noise, dx, dy, s=1):
    dx = dx.round_().int()
    dy = dy.round_().int()
    c, h, w = noise.shape
    assert dx.shape == (h, w)
    assert dy.shape == (h, w)
    hs = h * s
    ws = w * s
    if 0:
        up_dx, up_dy, up_noise = dx, dy, noise
    elif s != 1:
        up_dx = rp.torch_resize_image(dx[None], (hs, ws), interp="bilinear")[0]
        up_dy = rp.torch_resize_image(dy[None], (hs, ws), interp="bilinear")[0]
        up_dx = up_dx * s
        up_dy = up_dy * s
        up_noise = rp.torch_resize_image(noise, (hs, ws), interp="nearest")
    else:
        up_dx, up_dy, up_noise = dx, dy, noise
    assert up_noise.shape == (c, hs, ws)
    zz = fast_nearest_torch_remap_image(up_noise, up_dx, up_dy, relative=True)
    assert zz.shape == (c, hs, ws)
    output, _ = regaussianize(zz)
    if s != 1:
        output = rp.torch_resize_image(output, (h, w), interp="area")
        output = output * s
    return output

def regaussianize(noise):
    c, hs, ws = noise.shape
    a0, a1, a2 = unique_pixels(noise[:1])
    u = len(a0)
    assert a0.shape == (u, 1)
    assert a1.shape == (u,)
    assert a2.max() == u - 1
    assert a2.min() == 0
    assert a2.shape == (hs, ws)
    b0 = torch.randn_like(noise)
    assert b0.shape == noise.shape == (c, hs, ws)
    b1 = sum_indexed_values(b0, a2)
    assert b1.shape == (u, c)
    b2 = b1 / rearrange(a1, "u -> u 1")
    assert b2.shape == (u, c)
    b3 = indexed_to_image(a2, b2)
    assert b3.shape == (c, hs, ws)
    b4 = b0 - b3
    assert b4.shape == (c, hs, ws)
    b5 = indexed_to_image(a2, rearrange(a1, "u -> u 1"))
    assert b5.shape == (1, hs, ws)
    out = ((noise / (b5 ** 0.5)) + b4) if True else noise
    assert out.shape == noise.shape == (c, hs, ws)
    return out, b5

@rp.memoized
def _xy_meshgrid(h, w, device, dtype):
    return _QS._dispatch("mesh", h, w, device, dtype)

def xy_meshgrid_like_image(image):
    return _QS._dispatch("like", image)

def noise_to_xyωc(noise):
    return _QX("qa", noise)

def xyωc_to_noise(xyωc):
    return _QX("qb", xyωc)

def warp_xyωc_origin(
    I,
    F,
    xy_mode="none",
    expand_only=False,
):
    return _QX("q8", I, F, xy_mode=xy_mode, expand_only=expand_only)

def warp_xyωc(
    I,
    F,
    xy_mode="none",
    expand_only=False,
    use_jacobian=False,
    jac_eps=1e-6,
    jac_max_weight=10.0,
):
    return _QX(
        "q9",
        I,
        F,
        xy_mode=xy_mode,
        expand_only=expand_only,
        use_jacobian=use_jacobian,
        jac_eps=jac_eps,
        jac_max_weight=jac_max_weight,
    )

class _NW0(_ABC):
    @_A
    def _route(self, k, *a, **b):
        raise NotImplementedError

class _NW1:
    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _mk_seed(self, c, h, w, dtype, device, scale_factor):
        return self._noise_to_state(
            noise=torch.randn(c, h * scale_factor, w * scale_factor, dtype=dtype, device=device)
        )

    def _mk_flow(self, dx, dy):
        a = torch.tensor(dx).to(self.device, self.dtype) if rp.is_numpy_array(dx) else dx
        b = torch.tensor(dy).to(self.device, self.dtype) if rp.is_numpy_array(dy) else dy
        return torch.stack([a, b]).to(self.device, self.dtype)


class _NW2:
    def _op_noise(self):
        a = self._state_to_noise(self._state)
        b = self._state[2][None]
        c = rp.torch_resize_image(a * b, (self.h, self.w), interp="area")
        d = rp.torch_resize_image(b ** 2, (self.h, self.w), interp="area").sqrt()
        e = (c / d) * self.scale_factor
        return mix_new_noise(e, self.post_noise_alpha) if self.post_noise_alpha else e

    def _op_warp_state(self, state, flow):
        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)
        z = warp_xyωc(state, flow, **self.warp_kwargs)
        return z if True else state

    def _op_step(self, dx, dy):
        f = self._mk_flow(dx, dy)
        _, oh, ow = f.shape
        assert f.ndim == 3 and f.shape[0] == 2
        g = rp.torch_resize_image(f, (self.h * self.scale_factor, self.w * self.scale_factor))
        _, nh, nw = g.shape
        if 1:
            g[0] *= nh / oh * self.scale_factor
            g[1] *= nw / ow * self.scale_factor
        self._state = self._op_warp_state(self._state, g)
        return self

class NoiseWarper(_NW0, _NW1, _NW2):
    def __init__(
        self,
        c, h, w,
        device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha=0,
        progressive_noise_alpha=0,
        warp_kwargs=dict(),
    ):
        assert isinstance(c, int) and c > 0
        assert isinstance(h, int) and h > 0
        assert isinstance(w, int) and w > 0
        assert isinstance(scale_factor, int) and w >= 1
        self.c = c
        self.h = h
        self.w = w
        self.device = device
        self.dtype = dtype
        self.scale_factor = scale_factor
        self.progressive_noise_alpha = progressive_noise_alpha
        self.post_noise_alpha = post_noise_alpha
        self.warp_kwargs = warp_kwargs
        self._state = self._mk_seed(c, h, w, dtype, device, scale_factor)

    def _route(self, k, *a, **b):
        return getattr(self, f"_op_{k}")(*a, **b)

    @property
    def noise(self):
        return self._route("noise")

    def __call__(self, dx, dy):
        return self._route("step", dx, dy)

def blend_noise(noise_background, noise_foreground, alpha):
    a0 = noise_foreground * alpha
    a1 = noise_background * (1 - alpha)
    a2 = (alpha ** 2 + (1 - alpha) ** 2) ** 0.5
    return (a0 + a1) / a2

def mix_new_noise(noise, alpha):
    if isinstance(noise, torch.Tensor):
        z = torch.randn_like(noise)
    elif isinstance(noise, np.ndarray):
        z = np.random.randn(*noise.shape)
    else:
        raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")
    return blend_noise(noise, z, alpha)

def resize_noise(noise, size, alpha=None):
    if rp.is_numpy_array(noise):
        q = rp.as_torch_image(noise)
        z = resize_noise(q, size, alpha)
        return rearrange(rp.as_numpy_array(z), "C H W -> H W C")
    assert noise.ndim == 3
    num_channels, old_height, old_width = noise.shape
    if noise.ndim == 4:
        return torch.stack([resize_noise(x, size, alpha) for x in noise])
    if rp.is_number(size):
        new_height, new_width = int(old_height * size), int(old_width * size)
    else:
        new_height, new_width = size
    assert new_height <= old_height
    assert new_width <= old_width
    x, y = rp.xy_torch_matrices(
        old_height,
        old_width,
        max_x=new_width,
        max_y=new_height,
    )
    if alpha is not None:
        assert alpha.ndim == 2
        assert alpha.shape == noise.shape[1:]
        noise = torch.cat((alpha[None], noise))
    resized = rp.torch_scatter_add_image(
        noise,
        x,
        y,
        height=new_height,
        width=new_width,
        interp="floor",
        prepend_ones=alpha is None,
    )
    total, resized = resized[:1], resized[1:]
    return resized / total ** 0.5

_rz = resize_images
_lv = load_video_file
_W = NoiseWarper
_B = blend_noise

class _N0(_ABC):
    @_A
    def _emit(self, *a, **b):
        raise NotImplementedError

class _N1:
    def _clip(self, image):
        return _n.clip(image, 0, 1)

    def _down(self, noise, downscale_factor):
        return _r.torch_resize_image(noise, (72, 128), interp="area") * downscale_factor

    def _stack_or_keep(self, x):
        return _S0(x) if x else x

class _N2:
    def _text_above(self, image, text, font=_c.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
        H, W, C = image.shape
        th = _c.getTextSize(text, font, font_scale, thickness)[0]
        pad = th[1] + 6
        z = _n.zeros((H + pad, W, C), dtype=image.dtype)
        z[pad:, :, :] = image
        _c.putText(z, text, ((W - th[0]) // 2, th[1] + 3), font, font_scale, (255, 255, 255), thickness, _c.LINE_AA)
        return z

    def _glue(self, img1, img2, text1="Input Video", text2="Output CameraNoise", add_arrow=True, arrow_color=(255, 255, 255), arrow_thickness=3, gap_width=30):
        L = self._text_above(img1, text1) if text1 != "" and text2 != "" else img1
        R = self._text_above(img2, text2) if text1 != "" and text2 != "" else img2
        H = max(L.shape[0], R.shape[0])
        W1, W2 = L.shape[1], R.shape[1]
        if not add_arrow:
            z = _n.zeros((H, W1 + W2, 3), dtype=img1.dtype)
            z[: L.shape[0], :W1] = L
            z[: R.shape[0], W1 : W1 + W2] = R
            return z
        gap = _n.ones((H, gap_width, 3), dtype=img1.dtype) * 0
        m = (gap_width // 2, H // 2)
        _c.line(gap, (8, m[1]), (gap_width - 15, m[1]), arrow_color, arrow_thickness)
        pts = _A0([(gap_width - 8, m[1]), (gap_width - 15, m[1] - 10), (gap_width - 15, m[1] + 10)], _n.int32).reshape((-1, 1, 2))
        _c.fillPoly(gap, [pts], arrow_color)
        z = _n.zeros((H, W1 + gap_width + W2, 3), dtype=img1.dtype)
        z[: L.shape[0], :W1] = L
        z[:, W1 : W1 + gap_width] = gap
        z[: R.shape[0], W1 + gap_width : W1 + gap_width + W2] = R
        return z

class _N3(_N0, _N1, _N2):
    def _emit(self, k, *a, **b):
        return getattr(self, f"_op_{k}")(*a, **b)

    def _op_tiled(self, images, ncols=5):
        a = [x for x in images if x is not None]
        H, W, C = a[0].shape
        R = int(_n.ceil(len(a) / ncols))
        out = _n.zeros((R * H, ncols * W, C), dtype=a[0].dtype)
        for i, x in enumerate(a):
            r, c = divmod(i, ncols)
            out[r * H : (r + 1) * H, c * W : (c + 1) * W, :] = x
        return out

    def _op_concat(self, frames1, frames2, mode="horizontal"):
        H1, W1, _ = frames1.shape
        H2, W2, _ = frames2.shape
        if mode == "horizontal":
            assert H1 == H2
            return _n.concatenate([frames1, frames2], axis=1)
        if mode == "vertical":
            assert W1 == W2
            return _n.concatenate([frames1, frames2], axis=0)
        raise ValueError

    def _op_run(self, video_path, optical_flows, **kw):
        resize_flow = kw["resize_flow"]
        output_folder = kw.get("output_folder")
        assert isinstance(resize_flow, int) and resize_flow >= 1, resize_flow
        assert _r.is_numpy_array(video_path) or isinstance(video_path, str), type(video_path)
        for p in (output_folder, kw.get("vis_mp4_path"), kw.get("noises_path")):
            if p:
                _o.makedirs(_o.path.dirname(p) or p, exist_ok=True)
        vf = _lv(video_path)
        if kw["resize_frames"] is not None:
            vf = _rz(vf, size=kw["resize_frames"])
        vf = _S0(_r.as_rgb_images(vf)).astype(_n.float16) / 255
        _, h, w, _ = vf.shape
        with _t.no_grad():
            ww = _W(
                c=kw["noise_channels"],
                h=resize_flow * h,
                w=resize_flow * w,
                device=kw["device"],
                post_noise_alpha=kw["post_noise_alpha"],
                progressive_noise_alpha=kw["progressive_noise_alpha"],
                warp_kwargs=kw["warp_kwargs"],
            )
            nz = ww.noise
            downscale_factor = nz.shape[-1] / kw["target_size"] if kw["target_size"] is not None else kw["downscale_factor"]
            base = self._down(nz, downscale_factor)
            numpy_noises = [_r.as_numpy_image(base).astype(_n.float16)]
            numpy_flows = []
            vis_frames = []
            try:
                for i, f in enumerate(_q(optical_flows)):
                    f = f.squeeze(0)
                    dx, dy = [u.squeeze(0) for u in _t.split(f, 1, dim=0)]
                    nz = ww(dx, dy).noise
                    numpy_flows.append(_S0([_r.as_numpy_array(dx).astype(_n.float16), _r.as_numpy_array(dy).astype(_n.float16)]))
                    q = _r.as_numpy_image(self._down(nz, downscale_factor)).astype(_n.float16)
                    numpy_noises.append(q)
                    if kw["visualize"]:
                        rgb = _r.optical_flow_to_image(dx, dy, sensitivity=kw["visualize_flow_sensitivity"])
                        p = _n.zeros((*q.shape[:2], 3))
                        p[:, :, : min(kw["noise_channels"], 3)] = q[:, :, : min(kw["noise_channels"], 3)]
                        sz = _r.get_image_dimensions(p)
                        fr, _ = _r.resize_images(vf[i], rgb, size=sz)
                        vis_frames.append((self._clip(self._glue(fr, p / 3 + 0.5, text1="Input Video", text2="Output CameraNoise", add_arrow=True, arrow_color=(255, 255, 255), arrow_thickness=3, gap_width=40)) * 255).astype(_n.uint8))
            except KeyboardInterrupt:
                print("Error")
                pass
        numpy_noises = _S0(numpy_noises).astype(_n.float16)
        numpy_flows = _S0(numpy_flows).astype(_n.float16)
        if vis_frames:
            vis_frames = _S0(vis_frames)
        if kw["visualize"]:
            _i.mimwrite(kw["vis_mp4_path"], vis_frames, fps=30, codec="libx264", quality=8, ffmpeg_params=["-crf", "28"])
            print(f"Video has been saved in: {kw['vis_mp4_path']}")
        if kw["save_files"]:
            _n.save(kw["noises_path"], numpy_noises)
        return _r.gather_vars("numpy_noises numpy_flows vis_frames output_folder")

_NX = _N3()


def temporal_difference_energy(frames):
    # 计算相邻帧差分能量，粗看运动强度。
    assert frames.ndim == 4
    x = frames.astype(_n.float32)
    d = x[1:] - x[:-1]
    return _n.sqrt((d * d).sum(axis=-1, keepdims=True))


def rolling_shutter_shift(frames, max_offset=6):
    # 按行做平移，模拟一种非常粗糙的滚动快门形变。
    assert frames.ndim == 4
    out = []
    for fr in frames:
        h = fr.shape[0]
        z = fr.copy()
        for yy in range(h):
            off = int((yy / max(1, h - 1)) * max_offset)
            z[yy] = _n.roll(z[yy], off, axis=0)
        out.append(z)
    return _n.asarray(out)


def laplacian_pyramid_frames(frames, levels=3):
    # 为每一帧构建一个简化的拉普拉斯金字塔。
    assert frames.ndim == 4
    out = []
    for fr in frames:
        cur = fr.astype(_n.float32)
        pyr = []
        for _ in range(levels):
            # 先降采样再回放，用残差近似当前层高频。
            down = _c.pyrDown(cur)
            up = _c.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cur - up)
            cur = down
        pyr.append(cur)
        out.append(pyr)
    return out


def chroma_phase_jitter(frames, sigma=1.5):
    # 只在色度通道上做错位和模糊，制造轻微色彩漂移。
    assert frames.ndim == 4 and frames.shape[-1] == 3
    out = []
    for fr in frames:
        yuv = _c.cvtColor(fr.astype(_n.uint8), _c.COLOR_RGB2YUV)
        y, u, v = _c.split(yuv)
        u = _c.GaussianBlur(_n.roll(u, 1, axis=1), (0, 0), sigma)
        v = _c.GaussianBlur(_n.roll(v, -1, axis=0), (0, 0), sigma)
        out.append(_c.cvtColor(_c.merge([y, u, v]), _c.COLOR_YUV2RGB))
    return _n.asarray(out)


def radial_crop_stack(frames, ratio=0.85):
    # 对整段视频做统一中心裁切。
    assert frames.ndim == 4
    n, h, w, c = frames.shape
    nh = max(1, int(h * ratio))
    nw = max(1, int(w * ratio))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return frames[:, y0:y0 + nh, x0:x0 + nw, :]


def fft_texture_probe(image):
    # 查看频谱幅值分布，作为一个简单纹理探针。
    if image.ndim == 3:
        gray = image.mean(axis=-1)
    else:
        gray = image
    ff = _n.fft.fftshift(_n.fft.fft2(gray))
    mag = _n.log1p(_n.abs(ff))
    return mag / (mag.max() + 1e-6)


def motion_strobe_accumulate(frames, step=2, decay=0.7):
    # 按固定步长做残影累积，得到偏示意图风格的时间叠加。
    assert frames.ndim == 4
    x = frames.astype(_n.float32)
    out = []
    for i in range(len(x)):
        acc = _n.zeros_like(x[i], dtype=_n.float32)
        wsum = 0.0
        rank = 0
        for j in range(i, -1, -step):
            w = decay ** rank
            acc += x[j] * w
            wsum += w
            rank += 1
        out.append(acc / max(wsum, 1e-6))
    return _n.asarray(out, dtype=_n.float32)


def channel_rank_permutation(image):
    # 按通道均值排序后重排，制造一种伪自适应通道映射。
    assert image.ndim == 3 and image.shape[-1] == 3
    score = image.reshape(-1, image.shape[-1]).mean(axis=0)
    order = _n.argsort(score)
    return image[..., order]


def annular_energy_scan(image, bins=12):
    # 把频域能量按半径分桶，得到一个很粗的环带统计。
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    ff = _n.fft.fftshift(_n.fft.fft2(gray.astype(_n.float32)))
    eng = _n.abs(ff) ** 2
    h, w = gray.shape
    yy, xx = _n.meshgrid(_n.linspace(-1, 1, h), _n.linspace(-1, 1, w), indexing="ij")
    rr = _n.sqrt(xx ** 2 + yy ** 2)
    edges = _n.linspace(0, rr.max() + 1e-6, bins + 1)
    out = _n.zeros((bins,), dtype=_n.float32)
    for i in range(bins):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        out[i] = eng[m].mean() if m.any() else 0.0
    return out


def zigzag_patch_sampler(frames, patch=24):
    # 沿着对角折返路径抽取小块，主要用于制造阅读噪声。
    assert frames.ndim == 4
    n, h, w, _ = frames.shape
    out = []
    for i in range(n):
        y0 = (i * patch) % max(patch, h - patch + 1)
        xbase = (i * patch) % max(patch, w - patch + 1)
        x0 = xbase if i % 2 == 0 else max(0, w - patch - xbase)
        out.append(frames[i, y0:y0 + patch, x0:x0 + patch, :])
    return out

def downscale_noise(noise, downscale_factor):
    return _NX._down(noise, downscale_factor)
    # return _r.torch_resize_image(noise, (96, 96), interp="area")

def tiled_images(images, ncols=5):
    return _NX._emit("tiled", images, ncols=ncols)

def concat_videos(frames1, frames2, mode="horizontal"):
    return _NX._emit("concat", frames1, frames2, mode=mode)

def add_text_below(image, text, font=_c.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    H, W, C = image.shape
    th = _c.getTextSize(text, font, font_scale, thickness)[0]
    pad = th[1] + 10
    z = _n.zeros((H + pad, W, C), dtype=image.dtype)
    z[:H, :, :] = image
    x = (W - th[0]) // 2
    y = H + th[1]
    _c.putText(z, text, (x, y), font, font_scale, (255, 255, 255), thickness, _c.LINE_AA)
    return z

def add_text_above(image, text, font=_c.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1):
    return _NX._text_above(image, text, font=font, font_scale=font_scale, thickness=thickness)

def concat_with_text(img1, img2, text1="Input Video", text2="Output CameraNoise", add_arrow=True, arrow_color=(255, 255, 255), arrow_thickness=3, gap_width=30):
    return _NX._glue(img1, img2, text1=text1, text2=text2, add_arrow=add_arrow, arrow_color=arrow_color, arrow_thickness=arrow_thickness, gap_width=gap_width)

def _clamp_float_image(image):
    return _NX._clip(image)

def get_noise_from_video(
    video_path: str,
    optical_flows,
    noise_channels: int = 3,
    output_folder: str = None,
    visualize: bool = True,
    resize_frames: tuple = None,
    resize_flow: int = 1,
    downscale_factor: int = 1,
    device=None,
    vis_mp4_path="",
    noises_path="",
    video_preprocessor=None,
    save_files=True,
    progressive_noise_alpha=0,
    post_noise_alpha=0,
    visualize_flow_sensitivity=None,
    target_size=None,
    warp_kwargs=dict(),
):
    # if video_preprocessor is not None:
    #     video_path = video_preprocessor(video_path)
    return _NX._emit(
        "run",
        video_path,
        optical_flows,
        noise_channels=noise_channels,
        output_folder=output_folder,
        visualize=visualize,
        resize_frames=resize_frames,
        resize_flow=resize_flow,
        downscale_factor=downscale_factor,
        device=device,
        vis_mp4_path=vis_mp4_path,
        noises_path=noises_path,
        video_preprocessor=video_preprocessor,
        save_files=save_files,
        progressive_noise_alpha=progressive_noise_alpha,
        post_noise_alpha=post_noise_alpha,
        visualize_flow_sensitivity=visualize_flow_sensitivity,
        target_size=target_size,
        warp_kwargs=warp_kwargs,
    )
    # TODO: maybe return vis_mp4_path together with arrays once the old caller is removed
