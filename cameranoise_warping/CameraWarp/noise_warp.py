import numpy as np
import torch
from einops import rearrange
from utils.image_tensor_utils import (
    is_numpy_array,
    memoized,
    torch_remap_image,
    torch_resize_image,
    torch_scatter_add_image,
)

def unique_pixels(image):
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True, sorted=False)
    # unique_colors, inverse_indices, counts = torch.unique_consecutive(flattened_pixels, dim=0, return_inverse=True, return_counts=True)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output

def indexed_to_image(index_matrix, unique_colors):
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image


_arange_cache={}
def _cached_arange(length, device, dtype):
    code=hash((length,device,dtype))
    if code in _arange_cache:
        return _arange_cache[code]

    
    _arange_cache[code]= torch.arange(length , device=device, dtype=dtype)
    return _arange_cache[code]

def fast_nearest_torch_remap_image(image, x, y, *, relative=False, add_alpha_mask=False, use_cached_meshgrid=False):
    import torch

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        alpha_mask = torch.ones_like(image[:1])
        image = torch.cat([image, alpha_mask], dim=0)

    if torch.is_floating_point(x): x = x.round_().long()
    if torch.is_floating_point(y): y = y.round_().long()

    if relative:
        # assert in_height == out_height, "For relative warping, input and output heights must match, but got in_height={} and out_height={}".format(in_height, out_height)
        # assert in_width  == out_width , "For relative warping, input and output widths must match, but got in_width={} and out_width={}".format(in_width, out_width)
        x += _cached_arange(in_width , device=x.device, dtype=x.dtype)
        y += _cached_arange(in_height, device=y.device, dtype=y.dtype)[:,None]

    x.clamp_(0, in_width - 1)
    y.clamp_(0,in_height-1)
    out = image[:, y, x]

    expected_c = in_c+1 if add_alpha_mask else in_c
    assert out.shape == (expected_c, out_height, out_width), "Expected output shape: ({}, {}, {}), but got: {}".format(expected_c, out_height, out_width, out.shape)

    return out


def warp_noise(noise, dx, dy, s=1):
    #This is *certainly* imperfect. We need to have particle swarm in addition to this.

    dx=dx.round_().int()
    dy=dy.round_().int()

    c, h, w = noise.shape
    assert dx.shape==(h,w)
    assert dy.shape==(h,w)

    #s is scaling factor
    hs = h * s
    ws = w * s
    
    #Upscale the warping with linear interpolation. Also scale it appropriately.
    if s!=1:
        up_dx = torch_resize_image(dx[None], (hs, ws), interp="bilinear")[0]
        up_dy = torch_resize_image(dy[None], (hs, ws), interp="bilinear")[0]
        up_dx *= s
        up_dy *= s

        up_noise = torch_resize_image(noise, (hs, ws), interp="nearest")
    else:
        up_dx = dx
        up_dy = dy
        up_noise = noise
    assert up_noise.shape == (c, hs, ws)

    # Warp the noise - and put 0 where it lands out-of-bounds
    # up_noise = torch_remap_image(up_noise, up_dx, up_dy, relative=True, interp="nearest")
    up_noise = fast_nearest_torch_remap_image(up_noise, up_dx, up_dy, relative=True)
    assert up_noise.shape == (c, hs, ws)
    
    # Regaussianize the noise
    output, _ = regaussianize(up_noise)

    #Now we resample the noise back down again
    if s!=1:
        output = torch_resize_image(output, (h, w), interp='area')
        output = output * s #Adjust variance by multiplying by sqrt of area, aka sqrt(s*s)=s

    return output


def regaussianize(noise):
    c, hs, ws = noise.shape

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    assert unique_colors.shape == (u, 1)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(noise)
    assert foreign_noise.shape == noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    output = noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    assert output.shape == noise.shape == (c, hs, ws)

    return output, counts_image
    

@memoized
def _xy_meshgrid(h,w,device,dtype):
    y, x = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
    )

    output = torch.stack(
        [x, y],
    ).to(device, dtype)

    assert output.shape == (2, h, w)
    return output

def xy_meshgrid_like_image(image):
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape
    return _xy_meshgrid(h,w,image.device,image.dtype)

def noise_to_xyωc(noise):
    assert noise.ndim == 3, "noise is in CHW form"
    zeros=torch.zeros_like(noise[0][None])
    ones =torch.ones_like (noise[0][None])

    #Prepend [dx=0, dy=0, weights=1] channels
    output=torch.concat([zeros, zeros, ones, noise])
    return output

def xyωc_to_noise(xyωc):
    assert xyωc.ndim == 3, "xyωc is in [ω x y c]·h·w form"
    assert xyωc.shape[0]>3, 'xyωc should have at least one noise channel'
    noise=xyωc[3:]
    return noise

def warp_xyωc_origin(
    I,
    F,
    xy_mode="none",
    # USED FOR ABLATIONS:
    expand_only=False,
):
    """
    For ablations, set:
        - expand_only=True #No contraction
        - expand_only='bilinear' #Bilinear Interpolation
        - expand_only='nearest' #Nearest Neighbors Warping
    """
    #Input assertions
    assert F.device==I.device
    assert F.ndim==3, str(F.shape)+' F stands for flow, and its in [x y]·h·w form'
    assert I.ndim==3, str(I.shape)+' I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape==(2,h,w) # Should be [x y]·h·w
    device=I.device
    
    #How I'm going to address the different channels:
    x   = 0        #          // index of Δx channel
    y   = 1        #          // index of Δy channel
    xy  = 2        # I[:xy]
    xyω = 3        # I[:xyω]
    ω   = 2        # I[ω]     // index of weight channel
    c   = xyωc-xyω # I[-c:]   // num noise channels
    ωc  = xyωc-xy  # I[-ωc:]
    # h_dim = 1
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω]>0).all(), 'All weights should be greater than 0'

    #Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[-c:]=0

    #Caluclate initial pre-expand
    pre_expand = torch.empty_like(I)

    #ABLATION STUFF IN THIS PARAGRAPH
    #Using F_index instead of F so we can use ablations like bilinear, bicubic etc
    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)
    F_index = F
    if interp=='nearest':
        #Default behaviour, ablations or not
        F_index=F_index.round()

    pre_expand[:xy] = torch_remap_image(I[:xy], * -F, relative=True, interp=interp)# <---- Last minute change
    pre_expand[-ωc:] = torch_remap_image(I[-ωc:], * -F, relative=True, interp=interp)
    pre_expand[ω][pre_expand[ω]==0]=1 #Give new noise regions a weight of 1 - effectively setting it to init there

    if expand_only:
        if regauss:
            #This is an ablation option - simple warp + regaussianize
            #Enable to preview expansion-only noise warping
            #The default behaviour! My algo!
            pre_expand[-c:]=regaussianize(pre_expand[-c:])[0]
        else:
            #Turn zeroes to noise
            pre_expand[-c:]=torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:]==0) + pre_expand[-c:]
        return pre_expand

    #Calculate initial pre-shrink
    pre_shrink = I.clone()
    pre_shrink[:xy] += F

    #Pre-Shrink mask - discard out-of-bounds pixels
    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0<= pos[x]) & (pos[x] < w) & (0<= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None] #Match the shape of the input
    out_of_bounds = ~in_bounds
    assert out_of_bounds.dtype==torch.bool
    assert out_of_bounds.shape==(1,h,w)
    assert pre_shrink.shape == init.shape
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    #FLOATING POINT POSITIONS: I will disable this for now. It does in fact increase sensitivity! But it also makes it less long-term coherent
    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode=='none':
        pre_shrink[:xy] = 0

    if isinstance(xy_mode, int):
        # XY quantization: best to use odd numbers!
        quant = xy_mode
        pre_shrink[:xy] = (
            pre_shrink[:xy] * quant
        ).round() / quant  

    scat = lambda tensor: torch_scatter_add_image(tensor, *scat_xy, relative=True)

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1,h,w,dtype=bool,device=device) #The purpose is to get zeroes where no element is used
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    #Remove the expansion points where we'll use shrink
    pre_expand = torch.where(shrink_mask, init, pre_expand)
    # Debug preview can be added here if needed.

    #Horizontally Concat
    concat_dim = w_dim
    concat     = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    #Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert  counts_image.shape == (1, h, 2*w)
    #Distribute Weights
    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num() #We shouldn't need this, this is a crutch. Final mask should take care of this.

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == expand.shape == (3+c, h, w)
 
    shrink = torch.empty_like(pre_shrink)
    shrink[ω]   = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy]*pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:]*pre_shrink[ω][None]) / scat(pre_shrink[ω][None]**2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean() #Don't let them get too big or too small
    ε = .00001
    output[ω] += ε #Don't let it go too low
    
    assert (output[ω]>0).all()

    output[ω] **= .9999 #Make it tend towards 1


    return output


def warp_xyωc(
    I,
    F,
    xy_mode="none",
    # USED FOR ABLATIONS:
    expand_only=False,
    # --- new params for jacobian ---
    use_jacobian=False,
    jac_eps=1e-6,
    jac_max_weight=10.0,
):
    """
    ... original docstring ...
    New behaviour:
      - If use_jacobian=True, compute weight map w = 1/sqrt(|det(I + grad F)|)
        and apply it to both expansion and contraction paths.
      - jac_eps avoids div/NaN; jac_max_weight clamps huge weights.
    """
    #Input assertions
    assert F.device==I.device
    assert F.ndim==3, str(F.shape)+' F stands for flow, and its in [x y]·h·w form'
    assert I.ndim==3, str(I.shape)+' I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape==(2,h,w) # Should be [x y]·h·w
    device=I.device
    
    # ---------- NEW: compute Jacobian-based weight map ----------
    # F: [2, h, w]
    if use_jacobian:
        # use float flow (not rounded) for gradients
        dx = F[0].float()
        dy = F[1].float()

        # finite differences -> ∂dx/∂x, ∂dx/∂y, ∂dy/∂x, ∂dy/∂y
        # axis: x corresponds to width (dim=1 for these dx arrays)
        # compute ∂/∂x (width): diff along last dim
        dx_x = dx[:, 1:] - dx[:, :-1]
        # dx_x = torch.nn.functional.pad(dx_x, (0, 1, 0, 0), mode='replicate')   # shape (h,w)
        last_col = dx_x[:, -1:]   # 取最后一列，复制一次
        dx_x = torch.cat([dx_x, last_col], dim=1)  # 在宽度方向补一列

        dy_x = dy[:, 1:] - dy[:, :-1]
        # dy_x = torch.nn.functional.pad(dy_x, (0, 1, 0, 0), mode='replicate')
        last_col = dy_x[:, -1:]
        dy_x = torch.cat([dy_x, last_col], dim=1)

        # compute ∂/∂y (height): diff along first dim
        dx_y = dx[1:, :] - dx[:-1, :]
        # dx_y = torch.nn.functional.pad(dx_y, (0, 0, 0, 1), mode='replicate')
        last_row = dx_y[-1:, :]
        dx_y = torch.cat([dx_y, last_row], dim=0)

        dy_y = dy[1:, :] - dy[:-1, :]
        # dy_y = torch.nn.functional.pad(dy_y, (0, 0, 0, 1), mode='replicate')
        last_row = dy_y[-1:, :]
        dy_y = torch.cat([dy_y, last_row], dim=0)

        # Jacobian determinant: det( I + grad d )
        detJ = (1.0 + dx_x) * (1.0 + dy_y) - (dx_y * dy_x)   # (h,w)

        # Make weight: amplitude scale = 1/sqrt(|detJ|)
        det_abs = detJ.abs().clamp(min=jac_eps)
        weight = 1.0 / torch.sqrt(det_abs)

        # clamp and treat invalid detJ (negative or extremely small) as 0 (unreliable)
        # Negative detJ often means folding/occlusion; set weight 0 to mark unreliable sources.
        weight = torch.where(detJ > jac_eps, weight, torch.zeros_like(weight))
        weight = weight.clamp(max=jac_max_weight)

    else:
        weight = torch.ones((h,w), device=device, dtype=I.dtype)
    # ------------------------------------------------------------

    #How I'm going to address the different channels:
    x   = 0        # index of Δx channel
    y   = 1        # index of Δy channel
    xy  = 2        # I[:xy]
    xyω = 3        # I[:xyω]
    ω   = 2        # I[ω]
    c   = xyωc-xyω # num noise channels
    ωc  = xyωc-xy  # I[-ωc:]
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω]>0).all(), 'All weights should be greater than 0'

    #Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[-c:]=0

    #Caluclate initial pre-expand
    pre_expand = torch.empty_like(I)

    #ABLATION STUFF IN THIS PARAGRAPH
    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)
    F_index = F
    if interp=='nearest':
        #Default behaviour, ablations or not
        F_index=F_index.round()

    # --- REMAP: when remapping, also remap the weight map so that weights align with the expanded pixels ---
    # note: torch_remap_image accepts channels-first; weight[None] -> (1,h,w)
    remap_kwargs = dict(relative=True, interp=interp)
    pre_expand[:xy] = torch_remap_image(I[:xy], * -F, **remap_kwargs) # source offsets used
    pre_expand[-ωc:] = torch_remap_image(I[-ωc:], * -F, **remap_kwargs)

    if use_jacobian:
        # remap the scalar weight map into target coordinates (same remap as channels)
        weight_mapped = torch_remap_image(weight[None], * -F, **remap_kwargs)[0]  # (h,w)
        # apply amplitude scaling to noise channels that got remapped
        # multiply noise channels by weight; multiply ω (weights) by weight as well so splatting considers jacobian
        pre_expand[-c:] = pre_expand[-c:] * weight_mapped.unsqueeze(0)   # (c,h,w) * (h,w)
        pre_expand[ω]    = pre_expand[ω] * weight_mapped
    # ensure new areas (where remap produced zeros) have weight 1 (your original logic)
    pre_expand[ω][pre_expand[ω]==0]=1

    expand_only = True
    if expand_only:
        if regauss:
            pre_expand[-c:]=regaussianize(pre_expand[-c:])[0]
        else:
            pre_expand[-c:]=torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:]==0) + pre_expand[-c:]
        return pre_expand

    #Calculate initial pre-shrink
    pre_shrink = I.clone()
    # --- APPLY jacobian weight to source ω for contraction path ---
    if use_jacobian:
        # multiply source ω by weight (so that when we scatter, source's contribution is scaled)
        pre_shrink[ω] = pre_shrink[ω] * weight
        # also scale source noise amplitudes (so splatting numerator uses weighted amplitudes)
        pre_shrink[-c:] = pre_shrink[-c:] * weight.unsqueeze(0)

    pre_shrink[:xy] += F

    #Pre-Shrink mask - discard out-of-bounds pixels
    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0<= pos[x]) & (pos[x] < w) & (0<= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None] #Match the shape of the input
    out_of_bounds = ~in_bounds
    assert out_of_bounds.dtype==torch.bool
    assert out_of_bounds.shape==(1,h,w)
    assert pre_shrink.shape == init.shape
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    #FLOATING POINT POSITIONS handling...
    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode=='none':
        pre_shrink[:xy] = 0

    if isinstance(xy_mode, int):
        quant = xy_mode
        pre_shrink[:xy] = (pre_shrink[:xy] * quant).round() / quant  

    scat = lambda tensor: torch_scatter_add_image(tensor, *scat_xy, relative=True)

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1,h,w,dtype=bool,device=device)
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    #Remove the expansion points where we'll use shrink
    pre_expand = torch.where(shrink_mask, init, pre_expand)

    #Horizontally Concat
    concat_dim = w_dim
    concat     = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    #Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert  counts_image.shape == (1, h, 2*w)

    #Distribute Weights
    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num()
    
    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == expand.shape == (3+c, h, w)
    
    shrink = torch.empty_like(pre_shrink)
    shrink[ω]   = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy]*pre_shrink[ω][None]) / shrink[ω][None]
    # note: the sqrt in denominator follows your previous variance combination rule
    shrink[-c:] = scat(pre_shrink[-c:]*pre_shrink[ω][None]) / scat(pre_shrink[ω][None]**2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean()
    ε = .00001
    output[ω] += ε
    assert (output[ω]>0).all()
    output[ω] **= .9999

    return output



class NoiseWarper:
    def __init__(
        self,
        c, h, w,
        device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha = 0,
        progressive_noise_alpha = 0,
        warp_kwargs=dict(),
    ):

        #Some non-exhaustive input assertions
        assert isinstance(c,int) and c>0
        assert isinstance(h,int) and h>0
        assert isinstance(w,int) and w>0
        assert isinstance(scale_factor,int) and w>=1

        #Record arguments
        self.c=c
        self.h=h
        self.w=w
        self.device=device
        self.dtype=dtype
        self.scale_factor=scale_factor
        self.progressive_noise_alpha=progressive_noise_alpha
        self.post_noise_alpha=post_noise_alpha
        self.warp_kwargs=warp_kwargs

        #Initialize the state
        self._state = self._noise_to_state(
            noise=torch.randn(
                c,
                h * scale_factor,
                w * scale_factor,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def noise(self):
        noise = self._state_to_noise(self._state)
        weights = self._state[2][None] #xyωc
        noise = (
              torch_resize_image(noise * weights, (self.h, self.w), interp="area")
            / torch_resize_image(weights**2     , (self.h, self.w), interp="area").sqrt()
        )
        noise = noise * self.scale_factor

        if self.post_noise_alpha:
            noise = mix_new_noise(noise, self.post_noise_alpha)

        return noise
    
    def __call__(self, dx, dy):

        if is_numpy_array(dx): dx = torch.tensor(dx).to(self.device, self.dtype)
        if is_numpy_array(dy): dy = torch.tensor(dy).to(self.device, self.dtype)

        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape #Original height and width of the flow
        
        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow is in [x y]·h·w form"
        flow = torch_resize_image(
            flow,
            (
                self.h * self.scale_factor,
                self.w * self.scale_factor,
            ),
        ) 

        _, flowh, floww = flow.shape

        #Multiply the flow values by the size change
        flow[0] *= flowh / oflowh * self.scale_factor
        flow[1] *= floww / ofloww * self.scale_factor

        self._state = self._warp_state(self._state, flow)
        return self

    #The following three methods can be overridden in subclasses:

    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _warp_state(self, state, flow):
        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)

        return warp_xyωc(state, flow, **self.warp_kwargs)
    
def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5

def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise)      , alpha)
    elif isinstance(noise, np.ndarray): return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else: raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")
