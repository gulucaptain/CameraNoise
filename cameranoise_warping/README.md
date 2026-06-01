# CameraNoise Warping

[中文文档](README_zh.md) | English

📷 **CameraNoise Warping** generates temporally coherent CameraNoise from input videos.

Given one or more videos, the pipeline estimates camera motion with VGGT, converts adjacent camera poses into GRFlow, and uses that motion field to warp Gaussian noise frame by frame.

## 🚀 Quick Start

Install dependencies in the target environment:

```bash
pip install -r cameranoise_warping/requirements.txt
```

Preview the selected videos without loading VGGT:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --dry-run
```

Run CameraNoise generation:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml
```

## 🧭 Pipeline

For each input video:

1. Load video frames.
2. Estimate per-frame camera intrinsics and extrinsics with VGGT.
3. Optionally smooth intrinsics.
4. Convert adjacent camera poses into GRFlow.
5. Warp Gaussian noise with GRFlow.
6. Save CameraNoise as `.npy`.
7. Optionally save visualization video and flow arrays.

## 📥 Inputs

Input videos can be provided in the experiment yaml or through command-line overrides.

Use an inline list for small runs:

```yaml
videos:
  - /path/to/video_001.mp4
  - /path/to/video_002.mp4
```

Use a text file for large datasets:

```yaml
videos_txt: /path/to/videos.txt
```

`videos.txt` should contain one video path per line.

Command-line overrides:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --videos /path/to/a.mp4 /path/to/b.mp4
```

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --videos-txt /path/to/videos.txt
```

## 📤 Outputs

All outputs are written under `data_saved_root`.

For a video named `example.mp4`:

```text
data_saved_root/
  camerapose/
    example/
      extrinsic.pt
      intrinsic.pt
  noises/
    example_noises.npy
    example_visualization.mp4      # only when cameranoise_visualize: true
  flows/
    example_flows.npy              # only when saved_flow: true
  grflows/
    example.mp4                    # only when grflow_visualization: true
  errors.txt                       # failed videos and tracebacks
```

CameraNoise shape is validated after generation:

```text
[T, H, W, C]
```

Where:

- `T`: number of estimated camera frames.
- `H, W`: final CameraNoise spatial size.
- `C`: `noise_channels`, default `16`.

If the output shape does not match the expected definition, `cameranoise.py` raises an error and writes the failure to `errors.txt`.

## ⚙️ Configuration

The entry script loads:

1. `configs/default.yaml`
2. the experiment yaml passed by `--config`

Recommended layout:

```text
cameranoise_warping/
  configs/
    default.yaml
    experiments/
      demo.yaml
      dynpose.yaml
      your_dataset.yaml
```

Minimal experiment file:

```yaml
vggt_ckpt_pth: /path/to/VGGT-1B
data_saved_root: /path/to/output/cameranoise

videos_txt: /path/to/videos.txt

cameranoise_std_reference_size: 96
cameranoise_downscale_size: [72, 128]
cameranoise_visualize: false
saved_flow: false
```

## 🔧 Important Parameters

Camera and VGGT:

```yaml
vggt_ckpt_pth: /path/to/VGGT-1B
vggt_estimation_target_size: 518
warmup_intrinsics: false
intrinsic_smoothing: true
```

Noise resolution:

```yaml
FRAME: 0.5
FLOW: 4
LATENT: 8
noise_channels: 16
```

The base `downscale_factor` is computed in `cameranoise.py`:

```python
downscale_factor = round(FRAME * FLOW) * LATENT
```

With the default values:

```text
round(0.5 * 4) * 8 = 16
```

CameraNoise amplitude and output sizing:

```yaml
cameranoise_std_reference_size: 96
cameranoise_downscale_size: [72, 128]
```

`cameranoise_std_reference_size` controls the amplitude/std scaling. When this value is set, it replaces the base `downscale_factor` with:

```python
downscale_factor = raw_noise_width / cameranoise_std_reference_size
```

Use `96` for the CameraNoise setup used by this project.

`cameranoise_downscale_size` controls the saved CameraNoise spatial resolution. It should match the latent resolution used by inference:

```yaml
# 576x1024 inference video
cameranoise_std_reference_size: 96
cameranoise_downscale_size: [72, 128]

# 768x768 inference video
cameranoise_std_reference_size: 96
cameranoise_downscale_size: [96, 96]
```

If `cameranoise_downscale_size` is `null`, the noise is resized by:

```python
torch_resize_image(noise, 1 / downscale_factor, interp="area")
```

Saving options:

```yaml
cameranoise_save_files: true
cameranoise_visualize: false
cameranoise_visualize_match_video_size: true
saved_flow: false
grflow_visualization: false
overwrite: false
```

Visualization size:

```yaml
cameranoise_visualize_match_video_size: true
```

When this option is `true`, the visualization mp4 resizes the displayed noise image to match the video frame size. This only affects the `.mp4` visualization; the saved `.npy` file keeps the real CameraNoise size.

Set it to `false` if you want the visualization to use the raw CameraNoise spatial size.

## 🧪 Shape Check

The generated `example_noises.npy` is expected to be:

```text
[T, H, W, C]
```

This matches the downstream sanity-check convention:

```python
noise = np.load("example_noises.npy")      # [T,H,W,C]
noise = torch.tensor(noise)
noise = einops.rearrange(noise, "T H W C -> T C H W")
noise = noise.unsqueeze(0)                 # [1,T,C,H,W]
```

The main script checks `[T,H,W,C]` immediately after generation before reporting success.

## 🧩 Sharding

For multi-process or multi-GPU dataset generation, use explicit sharding:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/dynpose.yaml \
  --shard-id 0 \
  --num-shards 8
```

`--num-shards` is the total number of shards. `--shard-id` is the current shard index, starting from `0`. It must satisfy:

```text
0 <= shard-id < num-shards
```

The script assigns videos by list index:

```python
index % num_shards == shard_id
```

For example, if the input list has 10 videos and `--num-shards 3`:

| command | processed indices |
| --- | --- |
| `--shard-id 0 --num-shards 3` | `0, 3, 6, 9` |
| `--shard-id 1 --num-shards 3` | `1, 4, 7` |
| `--shard-id 2 --num-shards 3` | `2, 5, 8` |

A typical 4-GPU launch is:

```bash
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:0 --shard-id 0 --num-shards 4
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:1 --shard-id 1 --num-shards 4
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:2 --shard-id 2 --num-shards 4
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:3 --shard-id 3 --num-shards 4
```

Each process writes to the same `data_saved_root`, but each video has its own output filename, so shards do not overlap. Existing noise files are skipped unless `--overwrite` is set.

Use a quick slice for debugging:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/dynpose.yaml \
  --start 0 \
  --limit 100
```

`--start` and `--limit` are applied after optional sharding. For example, `--shard-id 0 --num-shards 4 --start 10 --limit 20` first selects shard `0`, then takes 20 videos starting from the 10th item inside that shard.

## 🖼 Result Gallery

`cameranoise.py` builds a static HTML gallery automatically after processing. By default, it is saved to:

```text
data_saved_root/index.html
```

Useful gallery overrides:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --gallery-max-items 24 \
  --gallery-sample even \
  --gallery-output /path/to/output/cameranoise/gallery.html
```

Use `--no-gallery` to disable automatic HTML generation.

You can also build or rebuild the gallery separately:

```bash
python cameranoise_warping/build_gallery.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --max-items 24 \
  --sample even
```

The gallery shows:

- result id and sample index
- CameraNoise visualization mp4
- CameraNoise shape `[T,H,W,C]`
- links to `.npy`, `.mp4`, flow, and camera pose files

Sampling options:

```bash
--max-items 24       # maximum cards to show, 0 means all
--sample first       # first N results
--sample even        # evenly spaced samples
--sample random      # random samples
--seed 0             # random seed
```

You can also pass the output root directly:

```bash
python cameranoise_warping/build_gallery.py \
  --data-root /path/to/output/cameranoise \
  --output /path/to/output/cameranoise/gallery.html \
  --max-items 32
```

## 🛠 Command-Line Overrides

Useful overrides:

```bash
--ckpt /path/to/VGGT-1B
--output-root /path/to/output
--device cuda:0
--overwrite
--dry-run
```

Example:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --ckpt /path/to/VGGT-1B \
  --output-root /path/to/output \
  --device cuda:0
```

## ✅ Recommended Workflow

1. Create a new yaml file under `configs/experiments/`.
2. Set `vggt_ckpt_pth`, `data_saved_root`, and either `videos` or `videos_txt`.
3. Run once with `--dry-run` to confirm the selected videos.
4. Run without `--dry-run` to generate CameraNoise.
5. Check `data_saved_root/noises/` for the generated `.npy` files.

Example:

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/your_dataset.yaml \
  --dry-run

python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/your_dataset.yaml
```
