# CameraNoise Warping

中文 | [English](README.md)

📷 **CameraNoise Warping** 用于从输入视频生成时序一致的 CameraNoise。

给定一个或多个输入视频，我们的算法首先利用相机参数估计模型（如VGGT）估计视频中的相机运动；随后，将相邻帧之间的相机位姿变化转换为GRFlow，并基于该运动场逐帧对高斯噪声进行warp，从而生成与相机运动时序一致的CameraNoise。

## 🚀 快速开始

先在目标环境中安装依赖：

```bash
pip install -r cameranoise_warping/requirements.txt
```

正式运行前，建议先预览本次会处理哪些视频。`--dry-run` 不会加载 VGGT，也不会生成文件：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --dry-run
```

确认视频列表无误后，开始生成 CameraNoise：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml
```

## 🧭 处理流程

对每个输入视频，代码会执行：

1. 读取视频帧。
2. 使用 VGGT 估计每帧相机内参和外参。
3. 可选地对相机内参做平滑。
4. 将相邻帧相机位姿转换为 GRFlow。
5. 使用 GRFlow warp 高斯噪声。
6. 将 CameraNoise 保存为 `.npy` 文件。
7. 可选保存可视化视频和 flow 数组。

## 📥 输入

输入视频可以写在实验 yaml 中，也可以通过命令行覆盖。

少量视频可以直接写成列表：

```yaml
videos:
  - /path/to/video_001.mp4
  - /path/to/video_002.mp4
```

大量视频建议使用文本文件：

```yaml
videos_txt: /path/to/videos.txt
```

`videos.txt` 每行写一个视频路径。

也可以通过命令行临时指定视频：

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

## 📤 输出

所有输出都会写到 `data_saved_root` 下。

假设输入视频名为 `example.mp4`，输出结构如下：

```text
data_saved_root/
  camerapose/
    example/
      extrinsic.pt
      intrinsic.pt
  noises/
    example_noises.npy
    example_visualization.mp4      # 仅当 cameranoise_visualize: true 时生成
  flows/
    example_flows.npy              # 仅当 saved_flow: true 时生成
  grflows/
    example.mp4                    # 仅当 grflow_visualization: true 时生成
  errors.txt                       # 失败视频和 traceback
```

CameraNoise 会在生成后进行维度检查：

```text
[T, H, W, C]
```

其中：

- `T`：估计到的相机帧数。
- `H, W`：最终 CameraNoise 的空间分辨率。
- `C`：噪声通道数，默认为 `16`。

如果输出维度与预期不一致，`cameranoise.py` 会抛出错误，并把失败信息写入 `errors.txt`。

## ⚙️ 配置

入口脚本会依次读取：

1. `configs/default.yaml`
2. `--config` 指定的实验 yaml

推荐目录结构：

```text
cameranoise_warping/
  configs/
    default.yaml
    experiments/
      demo.yaml
      dynpose.yaml
      your_dataset.yaml
```

最小实验配置示例：

```yaml
vggt_ckpt_pth: /path/to/VGGT-1B
data_saved_root: /path/to/output/cameranoise

videos_txt: /path/to/videos.txt

cameranoise_std_reference_size: 96
cameranoise_downscale_size: [72, 128]
cameranoise_visualize: false
saved_flow: false
```

## 🔧 关键参数

相机和 VGGT：

```yaml
vggt_ckpt_pth: /path/to/VGGT-1B
vggt_estimation_target_size: 518
warmup_intrinsics: false
intrinsic_smoothing: true
```

噪声分辨率相关参数：

```yaml
FRAME: 0.5
FLOW: 4
LATENT: 8
noise_channels: 16
```

基础 `downscale_factor` 在 `cameranoise.py` 中计算：

```python
downscale_factor = round(FRAME * FLOW) * LATENT
```

使用默认值时：

```text
round(0.5 * 4) * 8 = 16
```

CameraNoise 幅值和输出尺寸：

```yaml
cameranoise_std_reference_size: 96
cameranoise_downscale_size: [72, 128]
```

`cameranoise_std_reference_size` 控制幅值/std 缩放。设置该值后，会用下面的计算覆盖基础 `downscale_factor`：

```python
downscale_factor = raw_noise_width / cameranoise_std_reference_size
```

本项目的 CameraNoise 设置建议保持为 `96`。

`cameranoise_downscale_size` 控制最终保存的 CameraNoise 空间分辨率。它应该和推理时使用的 latent 分辨率一致：

```yaml
# 576x1024 推理视频
cameranoise_std_reference_size: 96
cameranoise_downscale_size: [72, 128]

# 768x768 推理视频
cameranoise_std_reference_size: 96
cameranoise_downscale_size: [96, 96]
```

如果 `cameranoise_downscale_size` 为 `null`，则默认按如下比例下采样：

```python
torch_resize_image(noise, 1 / downscale_factor, interp="area")
```

保存相关选项：

```yaml
cameranoise_save_files: true
cameranoise_visualize: false
cameranoise_visualize_match_video_size: true
saved_flow: false
grflow_visualization: false
overwrite: false
```

可视化尺寸：

```yaml
cameranoise_visualize_match_video_size: true
```

当该选项为 `true` 时，可视化 mp4 会把用于显示的 noise 图像 resize 到视频帧大小，方便观察。这个设置只影响 `.mp4` 可视化文件，不会改变保存到 `.npy` 中的真实 CameraNoise 尺寸。

如果希望可视化直接使用 CameraNoise 的原始空间尺寸，可以设置为 `false`。

## 🧪 维度检查

生成的 `example_noises.npy` 应当是：

```text
[T, H, W, C]
```

这与下游 sanity check 的使用方式一致：

```python
noise = np.load("example_noises.npy")      # [T,H,W,C]
noise = torch.tensor(noise)
noise = einops.rearrange(noise, "T H W C -> T C H W")
noise = noise.unsqueeze(0)                 # [1,T,C,H,W]
```

主脚本会在生成后立刻检查 `[T,H,W,C]`，检查通过后才报告成功。

## 🧩 数据分片

如果需要多进程或多 GPU 生成数据，可以使用显式分片：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/dynpose.yaml \
  --shard-id 0 \
  --num-shards 8
```

`--num-shards` 表示总共分成多少片。`--shard-id` 表示当前进程处理第几片，从 `0` 开始计数，必须满足：

```text
0 <= shard-id < num-shards
```

脚本会按视频列表下标取模来分配：

```python
index % num_shards == shard_id
```

例如输入列表有 10 个视频，并设置 `--num-shards 3`：

| 命令 | 处理的视频下标 |
| --- | --- |
| `--shard-id 0 --num-shards 3` | `0, 3, 6, 9` |
| `--shard-id 1 --num-shards 3` | `1, 4, 7` |
| `--shard-id 2 --num-shards 3` | `2, 5, 8` |

一个典型的 4 GPU 启动方式是：

```bash
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:0 --shard-id 0 --num-shards 4
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:1 --shard-id 1 --num-shards 4
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:2 --shard-id 2 --num-shards 4
python cameranoise_warping/cameranoise.py --config config.yaml --device cuda:3 --shard-id 3 --num-shards 4
```

这些进程可以写入同一个 `data_saved_root`。每个视频会使用自己的文件名保存结果，所以不同 shard 不会重复处理同一个视频。已存在的 noise 文件默认会跳过；如果需要重新生成，使用 `--overwrite`。

调试时可以只处理一小段数据：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/dynpose.yaml \
  --start 0 \
  --limit 100
```

`--start` 和 `--limit` 会在分片之后再生效。例如 `--shard-id 0 --num-shards 4 --start 10 --limit 20` 会先选出第 `0` 片，然后从这片内部的第 10 个视频开始，取 20 个视频处理。

## 🖼 结果网页

`cameranoise.py` 在处理完成后会自动生成静态 HTML 结果页。默认保存到：

```text
data_saved_root/index.html
```

常用网页参数：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --gallery-max-items 24 \
  --gallery-sample even \
  --gallery-output /path/to/output/cameranoise/gallery.html
```

如果不想自动生成网页，可以使用 `--no-gallery`。

也可以单独生成或重新生成结果网页：

```bash
python cameranoise_warping/build_gallery.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --max-items 24 \
  --sample even
```

网页会展示：

- 结果 id 和样本序号
- CameraNoise 可视化 mp4
- CameraNoise shape：`[T,H,W,C]`
- `.npy`、`.mp4`、flow、相机位姿文件链接

抽样参数：

```bash
--max-items 24       # 最多展示多少条，0 表示全部展示
--sample first       # 取前 N 条
--sample even        # 均匀抽样
--sample random      # 随机抽样
--seed 0             # 随机种子
```

也可以直接指定输出根目录：

```bash
python cameranoise_warping/build_gallery.py \
  --data-root /path/to/output/cameranoise \
  --output /path/to/output/cameranoise/gallery.html \
  --max-items 32
```

## 🛠 命令行覆盖参数

常用覆盖参数：

```bash
--ckpt /path/to/VGGT-1B
--output-root /path/to/output
--device cuda:0
--overwrite
--dry-run
```

示例：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/demo.yaml \
  --ckpt /path/to/VGGT-1B \
  --output-root /path/to/output \
  --device cuda:0
```

## ✅ 推荐使用流程

1. 在 `configs/experiments/` 下创建一个新的实验 yaml。
2. 设置 `vggt_ckpt_pth`、`data_saved_root`，以及 `videos` 或 `videos_txt`。
3. 先运行一次 `--dry-run`，确认视频列表正确。
4. 去掉 `--dry-run`，正式生成 CameraNoise。
5. 在 `data_saved_root/noises/` 中检查生成的 `.npy` 文件。

示例：

```bash
python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/your_dataset.yaml \
  --dry-run

python cameranoise_warping/cameranoise.py \
  --config cameranoise_warping/configs/experiments/your_dataset.yaml
```
