from abc import ABC as _ABC, abstractmethod as _A
from importlib import import_module as _im


class _U0(_ABC):
    @_A
    def _pull(self, *a, **b):
        raise NotImplementedError


class _U1(_U0):
    def __init__(self):
        self._m = {}
        self._n = {}
        self._c = {}
        self._s0()
        self._s1()
        self._s2()
        self._s3()

    def _s0(self):
        self._m.update({
            "cup0": ("utils.runtime_ops", "warmup_intrinsics_linear"),
            "cup1": ("utils.runtime_ops", "warmup_intrinsics_fix"),
            "cup2": ("utils.runtime_ops", "IntrinsicKalmanFilter"),
            "cup3": ("utils.runtime_ops", "smooth_intrinsics"),
        })

    def _s1(self):
        self._m.update({
            "cup4": ("utils.runtime_ops", "flow_to_color"),
            "cup5": ("utils.runtime_ops", "grflow_save_as_video"),
            "cup6": ("utils.runtime_ops", "so3_log"),
            "cup7": ("utils.runtime_ops", "so3_exp"),
            "cup8": ("utils.runtime_ops", "se3_log"),
            "cup9": ("utils.runtime_ops", "se3_exp"),
        })

    def _s2(self):
        self._m.update({
            "cupa": ("utils.runtime_ops", "rotation_matrix_to_angle_axis"),
            "cupb": ("utils.runtime_ops", "angle_axis_to_rotation_matrix"),
            "cupc": ("utils.runtime_ops", "scale_rotation_matrix"),
        })

    def _s3(self):
        self._n.update({
            "warmup_intrinsics_linear": "lid0",
            "warmup_intrinsics_fix": "lid1",
            "IntrinsicKalmanFilter": "lid2",
            "smooth_intrinsics": "lid3",
            "flow_to_color": "lid4",
            "grflow_save_as_video": "lid5",
            "so3_log": "lid6",
            "so3_exp": "lid7",
            "se3_log": "lid8",
            "se3_exp": "lid9",
            "rotation_matrix_to_angle_axis": "lida",
            "angle_axis_to_rotation_matrix": "lidb",
            "scale_rotation_matrix": "lidc",
        })

    def _s4(self, k):
        return {
            "lid0": "cup0",
            "lid1": "cup1",
            "lid2": "cup2",
            "lid3": "cup3",
            "lid4": "cup4",
            "lid5": "cup5",
            "lid6": "cup6",
            "lid7": "cup7",
            "lid8": "cup8",
            "lid9": "cup9",
            "lida": "cupa",
            "lidb": "cupb",
            "lidc": "cupc",
        }[k]

    def _k0(self, k):
        return self._n[k] if k in self._n else k

    def _k1(self, k):
        return self._m[self._s4(self._k0(k))]

    def _pull(self, k):
        if k in self._c:
            return self._c[k]
        m, a = self._k1(k)
        self._c[k] = getattr(_im(m), a)
        return self._c[k]


_UX = _U1()
__all__ = tuple(_UX._n)


def __getattr__(name):
    if name not in _UX._n:
        raise AttributeError(name)
    return _UX._pull(name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
