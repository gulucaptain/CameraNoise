import functools
import numbers

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def is_numpy_array(value):
    return isinstance(value, np.ndarray)


def is_number(value):
    return isinstance(value, numbers.Number)


def memoized(function):
    cache = {}

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cache[key]

    return wrapper


def get_image_dimensions(image):
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            return int(image.shape[-2]), int(image.shape[-1])
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])
    if isinstance(image, np.ndarray):
        return int(image.shape[0]), int(image.shape[1])
    raise TypeError(f"Unsupported image type: {type(image)}")


def _size_to_height_width(size, in_height, in_width):
    if isinstance(size, numbers.Number):
        return max(1, int(round(in_height * size))), max(1, int(round(in_width * size)))
    if len(size) != 2:
        raise ValueError(f"size must be a number or (height, width), got {size}")
    height = in_height if size[0] is None else int(size[0])
    width = in_width if size[1] is None else int(size[1])
    return height, width


def torch_resize_image(image, size, interp="auto", *, copy=True):
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError(f"image must be a CHW torch tensor, got {type(image)} {getattr(image, 'shape', None)}")

    in_height, in_width = get_image_dimensions(image)
    height, width = _size_to_height_width(size, in_height, in_width)
    if (height, width) == (in_height, in_width):
        return image.clone() if copy else image

    if interp == "auto":
        interp = "area" if height <= in_height and width <= in_width else "bilinear"

    valid = {"bilinear", "bicubic", "area", "nearest", "nearest-exact"}
    if interp not in valid:
        raise ValueError(f"Unsupported interpolation mode: {interp}")

    kwargs = {}
    if interp in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False

    return rearrange(
        F.interpolate(rearrange(image, "c h w -> 1 c h w"), size=(height, width), mode=interp, **kwargs),
        "1 c h w -> c h w",
    )


def as_numpy_array(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def as_numpy_image(image, *, copy=True):
    if isinstance(image, np.ndarray):
        return image.copy() if copy else image
    if isinstance(image, torch.Tensor):
        array = as_numpy_array(image)
        if array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        return array
    raise TypeError(f"Unsupported image type: {type(image)}")


def as_numpy_images(images, copy=True):
    if isinstance(images, np.ndarray):
        return images.copy() if copy else images
    if isinstance(images, torch.Tensor):
        array = as_numpy_array(images)
        if array.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got {images.shape}")
        return np.transpose(array, (0, 2, 3, 1))
    return [as_numpy_image(image, copy=copy) for image in images]


def as_torch_image(image, *, device=None, dtype=None, copy=False):
    if isinstance(image, torch.Tensor):
        tensor = image.clone() if copy else image
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor
    if isinstance(image, np.ndarray):
        array = image.copy() if copy else image
        if array.ndim == 3:
            array = np.transpose(array, (2, 0, 1))
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        tensor = torch.tensor(array, device=device, dtype=dtype)
        return tensor
    raise TypeError(f"Unsupported image type: {type(image)}")


def as_rgb_images(images, *, copy=True):
    output = []
    for image in images:
        if isinstance(image, torch.Tensor):
            image = as_numpy_image(image, copy=copy)
        elif copy:
            image = image.copy()

        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] != 3:
            raise ValueError(f"Expected grayscale, RGB, or RGBA image, got {image.shape}")
        output.append(image)
    return output


def torch_remap_image(image, x, y, *, relative=False, interp="bilinear", add_alpha_mask=False):
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError(f"image must be a CHW torch tensor, got {type(image)} {getattr(image, 'shape', None)}")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        image = torch.cat([image, torch.ones_like(image[:1])], dim=0)

    if relative:
        if (in_height, in_width) != (out_height, out_width):
            raise ValueError("relative remap requires input and output sizes to match")
        x = x + torch.arange(in_width, device=x.device, dtype=x.dtype)
        y = y + torch.arange(in_height, device=y.device, dtype=y.dtype)[:, None]

    x_norm = (x / max(in_width - 1, 1)) * 2 - 1
    y_norm = (y / max(in_height - 1, 1)) * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).to(image.dtype)

    mode = {"bilinear": "bilinear", "bicubic": "bicubic", "nearest": "nearest"}[interp]
    out = F.grid_sample(image.unsqueeze(0), grid, mode=mode, align_corners=True)
    expected_c = in_c + 1 if add_alpha_mask else in_c
    return out.squeeze(0).reshape(expected_c, out_height, out_width)


def _bilinear_weights(x, y):
    floor_x = x.floor()
    floor_y = y.floor()
    ceil_x = floor_x + 1
    ceil_y = floor_y + 1

    a = (ceil_x - x) * (ceil_y - y)
    b = (x - floor_x) * (ceil_y - y)
    c = (ceil_x - x) * (y - floor_y)
    d = (x - floor_x) * (y - floor_y)

    return (
        (floor_x, floor_y, a),
        (ceil_x, floor_y, b),
        (floor_x, ceil_y, c),
        (ceil_x, ceil_y, d),
    )


def torch_scatter_add_image(image, x, y, *, relative=False, interp="floor", height=None, width=None, prepend_ones=False):
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError(f"image must be a CHW torch tensor, got {type(image)} {getattr(image, 'shape', None)}")

    if prepend_ones:
        image = torch.cat([torch.ones_like(image[:1]), image], dim=0)

    in_c, in_height, in_width = image.shape
    out_height = int(height) if height is not None else in_height
    out_width = int(width) if width is not None else in_width

    if interp == "bilinear":
        parts = []
        for x_part, y_part, weight in _bilinear_weights(x, y):
            parts.append(
                torch_scatter_add_image(
                    image * weight[None],
                    x_part,
                    y_part,
                    relative=relative,
                    interp="floor",
                    height=out_height,
                    width=out_width,
                )
            )
        return sum(parts)

    if interp == "round":
        x = x.round()
        y = y.round()
    elif interp == "ceil":
        x = x.ceil()
        y = y.ceil()
    elif interp != "floor":
        raise ValueError(f"Unsupported scatter interpolation mode: {interp}")

    x = x.long()
    y = y.long()

    if relative:
        if (in_height, in_width) != tuple(x.shape):
            raise ValueError("relative scatter requires x/y shape to match input size")
        x = x + torch.arange(in_width, device=x.device, dtype=x.dtype)
        y = y + torch.arange(in_height, device=y.device, dtype=y.dtype)[:, None]

    indices = y * out_width + x
    valid = (x >= 0) & (x < out_width) & (y >= 0) & (y < out_height)
    flat_indices = indices.reshape(-1)[valid.reshape(-1)]
    flat_values = rearrange(image, "c h w -> (h w) c")[valid.reshape(-1)]

    out = torch.zeros((out_height * out_width, in_c), dtype=image.dtype, device=image.device)
    out.index_add_(0, flat_indices, flat_values)
    return rearrange(out, "(h w) c -> c h w", h=out_height, w=out_width)


def optical_flow_to_image(dx, dy, *, mode="saturation", sensitivity=None):
    if isinstance(dx, torch.Tensor):
        dx = as_numpy_array(dx)
    if isinstance(dy, torch.Tensor):
        dy = as_numpy_array(dy)

    dx = dx.astype(float)
    dy = dy.astype(float)
    hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
    hsv[:] = 255
    mag, ang = cv2.cartToPolar(dx, dy)

    if sensitivity is None:
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    elif is_number(sensitivity):
        norm_mag = np.clip(np.tanh(sensitivity * mag) * 255, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Invalid sensitivity: {sensitivity}")

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., {"brightness": 2, "saturation": 1}[mode]] = norm_mag
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
