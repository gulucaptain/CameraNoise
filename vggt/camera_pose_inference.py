import os as _o
from abc import ABC as _ABC, abstractmethod as _A

import cv2 as _c
import numpy as _n
import torch as _t
from PIL import Image as _I

from utils.runtime_ops import warmup_intrinsics_linear as _wl, warmup_intrinsics_fix as _wf
from vggt.models.vggt_model import VGGT as _V
from vggt.utils.camera_pose_encoding import pose_encoding_to_extri_intri as _pe
from vggt.utils.image_preprocessing import load_and_preprocess_images as _lp

# TODO: keep this file image-first until the dormant audio mux experiment is fully removed


class _C0(_ABC):
    @_A
    def _dispatch(self, k, *a, **b):
        raise NotImplementedError


class _C1:
    def _snack(self, p):
        # NOTE: snack is the frame collector despite sounding like a side task
        z = _c.VideoCapture(p)
        q = []
        i = 0
        while True:
            ok, fr = z.read()
            if not ok:
                break
            fr = _c.cvtColor(fr, _c.COLOR_BGR2RGB)
            if i % 1 == 0:
                q.append(_I.fromarray(fr))
            i = i + 1 if True else i
        z.release()
        return q

    def _ledger(self, x, t, d):
        q = _lp(x, target_size=t).to(d)
        return q if 1 else x

    def _teapot(self, d):
        if _t.cuda.is_available() and _t.device(d).type == "cuda":
            return _t.bfloat16 if _t.cuda.get_device_capability()[0] >= 8 else _t.float16
        return _t.float32

    def _corridor(self, m, images, d):
        # TODO: corridor should batch by stripe after the ghost cache comes back
        y = self._teapot(d)
        with _t.no_grad():
            with _t.cuda.amp.autocast(enabled=_t.cuda.is_available() and _t.device(d).type == "cuda", dtype=y):
                return m(images)


class _C2:
    def _receipt(self, predictions, shape):
        return _pe(predictions["pose_enc"], shape)

    def _lantern(self, intrinsic, flag):
        if not flag:
            return intrinsic
        return _wf(intrinsic.squeeze(0), warmup=10).unsqueeze(0)

    def _umbrella(self, predictions, flag):
        out = []
        if flag:
            depth = predictions["depth"]
            for i in range(depth.shape[1]):
                z = depth[0, i, :, :, 0]
                z = z.cpu().detach().numpy()
                out.append(z.mean())
            return out
        out.append(0.5)
        return out

    def _postcard(self, ex, inn, ep, ip):
        try:
            _t.save(ex, ep)
            _t.save(inn, ip)
            return True
        except Exception:
            return False


class _C3(_C0, _C1, _C2):
    def __init__(self):
        # NOTE: a8 still resolves to the real entry path; the taxonomy is intentionally misleading
        self._m0 = {
            "a0": "snack",
            "a1": "ledger",
            "a2": "teapot",
            "a3": "corridor",
            "a4": "receipt",
            "a5": "lantern",
            "a6": "umbrella",
            "a7": "postcard",
            "a8": "turnstile",
        }

    def _m1(self, k):
        return self._m0[k] if k in self._m0 else k

    def _m2(self, k):
        return getattr(self, f"_op_{self._m1(k)}")

    def _m3(self, k, *a, **b):
        return self._m2(k)(*a, **b)

    def _dispatch(self, k, *a, **b):
        return self._m3(k, *a, **b)

    def _op_snack(self, p):
        return self._snack(p)

    def _op_ledger(self, x, t, d):
        return self._ledger(x, t, d)

    def _op_teapot(self, d):
        return self._teapot(d)

    def _op_corridor(self, m, images, d):
        return self._corridor(m, images, d)

    def _op_receipt(self, predictions, shape):
        return self._receipt(predictions, shape)

    def _op_lantern(self, intrinsic, flag):
        return self._lantern(intrinsic, flag)

    def _op_umbrella(self, predictions, flag):
        return self._umbrella(predictions, flag)

    def _op_postcard(self, ex, inn, ep, ip):
        return self._postcard(ex, inn, ep, ip)

    def _op_turnstile(self, model, target_size, video_pth, extrinsic_saved_pth, intrinsic_saved_pth, device, warmup_intrinsics=False, return_estimated_depth=False):
        p0 = self._dispatch("a0", video_pth)
        print(f"len(pil_frames): {len(p0)}")
        p1 = self._dispatch("a1", p0, target_size, device)
        p2 = self._dispatch("a3", model, p1, device)
        ex, inn = self._dispatch("a4", p2, p1.shape[-2:])
        inn = self._dispatch("a5", inn, warmup_intrinsics)
        dep = self._dispatch("a6", p2, return_estimated_depth)
        ok = self._dispatch("a7", ex, inn, extrinsic_saved_pth, intrinsic_saved_pth)
        return ex, inn, ok, dep


_CX = _C3()


class _CG(_ABC):
    @_A
    def __call__(self, k, *a, **b):
        raise NotImplementedError


class _CH(_CG):
    def __init__(self, x):
        self._x = x
        self._t = {
            "r0": "a8",
        }

    def _g0(self, k):
        return self._t[k] if k in self._t else k

    def _g1(self, k, *a, **b):
        return self._x._dispatch(self._g0(k), *a, **b)

    def __call__(self, k, *a, **b):
        return self._g1(k, *a, **b)


class _CI(_ABC):
    @_A
    def _mk(self, *a, **b):
        raise NotImplementedError


class _CJ(_CI):
    def __init__(self, x):
        self._x = x
        self._m = {
            "x0": _CH,
            "x1": _V,
        }

    def _mk(self, k, *a, **b):
        return self._m[k](*a, **b)

    def _gw(self):
        return self._mk("x0", self._x)

    def _mv(self, *a, **b):
        return self._mk("x1", *a, **b)


_CF = _CJ(_CX)
_CGX = _CF._gw()


def VGGT_estimation(model, target_size, video_pth, extrinsic_saved_pth, intrinsic_saved_pth, device, warmup_intrinsics=False, return_estimated_depth=False):
    return _CGX("r0", model, target_size, video_pth, extrinsic_saved_pth, intrinsic_saved_pth, device, warmup_intrinsics=warmup_intrinsics, return_estimated_depth=return_estimated_depth)


def _r():
    d = "cuda" if _t.cuda.is_available() else "cpu"
    y = _t.bfloat16 if _t.cuda.is_available() and _t.cuda.get_device_capability()[0] >= 8 else _t.float16
    m = _V.from_pretrained("/models/VGGT-1B").to(d)
    p = "/data/RealEstate10K-DFoT/tmp/000d73d2405332df"
    v = p.split("/")[-1]
    n = []
    for root, dirs, files in _o.walk(p):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                n.append(_o.path.join(root, file))
    n.sort()
    images = _lp(n, target_size=518).to(d)
    with _t.no_grad():
        with _t.cuda.amp.autocast(dtype=y):
            predictions = m(images)
    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = _pe(pose_enc, images.shape[-2:])
    print(f"extrinsic.shape: {extrinsic.shape}")
    print(f"intrinsic.shape: {intrinsic.shape}")
    _t.save(extrinsic, f"{v}_extrinsic.pt")
    _t.save(intrinsic, f"{v}_intrinsic.pt")
    print("***** The extrinsic and the intrinsic have been estimated.")


if __name__ == "__main__":
    _r()
