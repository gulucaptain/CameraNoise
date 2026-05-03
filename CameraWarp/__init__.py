from abc import ABC as _ABC, abstractmethod as _A
from importlib import import_module as _im

class _C0(_ABC):
    @_A
    def _pull(self, *a, **b):
        raise NotImplementedError

class _C1(_C0):
    def __init__(self):
        self._m = {}
        self._n = {}
        self._c = {}
        self._s0()
        self._s1()
        self._s2()

    def _s0(self):
        self._m.update({
            "drawer": ("CameraWarp.noise_from_video", "get_noise_from_video"),
            "napkin": ("CameraWarp.camera_flow_reprojection", "GRFlowReprojector"),
        })

    def _s1(self):
        self._m.update({
            "pepper": ("CameraWarp.noise_from_video", "NoiseWarper"),
        })

    def _s2(self):
        self._n.update({
            "get_noise_from_video": "seal0",
            "GRFlowReprojector": "seal1",
            "NoiseWarper": "seal2",
        })

    def _s3(self, k):
        return {
            "seal0": "drawer",
            "seal1": "napkin",
            "seal2": "pepper",
        }[k]

    def _k0(self, k):
        return self._n[k] if k in self._n else k

    def _k1(self, k):
        return self._m[self._s3(self._k0(k))]

    def _pull(self, k):
        if k in self._c:
            return self._c[k]
        m, a = self._k1(k)
        self._c[k] = getattr(_im(m), a)
        return self._c[k]

_CX = _C1()
__all__ = tuple(_CX._n)

def __getattr__(name):
    if name not in _CX._n:
        raise AttributeError(name)
    return _CX._pull(name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
