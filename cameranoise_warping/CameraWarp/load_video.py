import cv2
import os
import numpy as np

def load_video_stream(path, start_frame=0, with_length=True, frame_transform=None):
    """
    Load video frames as an iterator.
    
    Args:
        path (str): Path to the video file (local or downloaded).
        start_frame (int): Frame index to start reading from.
        with_length (bool): If True, try to provide __len__ for the iterator.
        frame_transform (callable): Optional function applied to each frame.
    
    Yields:
        frame (ndarray): Video frame in RGB format.
    """
    assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
    assert isinstance(start_frame, int) and start_frame >= 0, f"start_frame must be >= 0, got {start_frame}"
    assert os.path.exists(path), f"Video path does not exist: {path}"

    cap = cv2.VideoCapture(path)

    # 跳到指定的起始帧
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 计算视频长度（可选）
    total_frames = None
    if with_length:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            total_frames = max(0, frame_count - start_frame)

    def frame_generator():
        while True:
            success, frame = cap.read()
            if not success:
                break
            # OpenCV 默认是 BGR，这里转成 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_transform is not None:
                frame = frame_transform(frame)
            yield frame

    # 如果需要长度信息，返回一个带 __len__ 的生成器
    if total_frames is not None:
        class IteratorWithLen:
            def __init__(self, iterator, length):
                self.iterator = iterator
                self.length = length
            def __iter__(self):
                return self.iterator
            def __len__(self):
                return self.length
        return IteratorWithLen(frame_generator(), total_frames)
    else:
        return frame_generator()


# 缓存字典
_load_video_cache = {}
def load_video_file(path, start_frame=0, length=None, show_progress=False, use_cache=False, frame_transform=None):
    """
    Load a full video into memory as a numpy array (frames in RGB format).
    
    Args:
        path (str): Path to video file.
        start_frame (int): Index of the first frame to read.
        length (int | None): Number of frames to read. None means read until end.
        show_progress (bool): If True, print loading progress.
        use_cache (bool): If True, cache results for repeated calls.
        frame_transform (callable | None): Optional transform applied to each frame.

    Returns:
        np.ndarray: Video frames with shape (num_frames, H, W, 3).
    """
    assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
    assert isinstance(start_frame, int) and start_frame >= 0, f"start_frame must be >= 0, got {start_frame}"
    assert length is None or (isinstance(length, int) and length >= 0), f"length must be None or non-negative int, got {length}"

    # 生成唯一缓存 key
    cache_id = (os.path.abspath(path), start_frame, length, frame_transform)

    # 如果有缓存直接返回
    if use_cache and cache_id in _load_video_cache:
        return _load_video_cache[cache_id]

    # 从流式接口读取
    stream = load_video_stream(
        path,
        start_frame=start_frame,
        with_length=show_progress,
        frame_transform=frame_transform
    )

    frames = []
    for i, frame in enumerate(stream):
        if length is not None and i >= length:
            break

        # 打印进度
        if show_progress:
            if hasattr(stream, "__len__"):
                total = len(stream) if length is None else min(len(stream), length)
                msg = f"Loaded frame {i+1} of {total}..."
            else:
                msg = f"Loaded frame {i+1}..."
            print(f"\rload_video: path={path!r}: {msg}", end="")

        frames.append(frame)

    if show_progress:
        print(f"\rload_video: path={path!r}: done loading frames, creating numpy array...")

    # 转为 numpy 数组
    frames = np.asarray(frames)

    if show_progress:
        print("done.\n")

    # 缓存结果
    if use_cache:
        _load_video_cache[cache_id] = frames

    return frames
