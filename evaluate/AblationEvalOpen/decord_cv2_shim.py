"""decord's cv2 backend implementation, only providing the API used by ask_vllm.py. Why
it's needed: decord only released the x86_64 wheel, and aarch64 machines (and pypi
mirrors without eva-decord environment) cannot install it. The image generation process
100% depends on it to decode. cv2 works on the local machine and can open mkv files and
seek by frame number (3 random seeks take ~0.55s), sufficient to replace. Must strictly
match decord's API: otherwise the output images will be wrong: -
get_batch(...).asnumpy() returns RGB (cv2 native is BGR, explicitly converted here);
callers will do RGB2BGR again before saving; if this step is skipped, colors will be
inverted. - len(vr) returns total frame count, get_avg_fps() returns average fps. Usage
(before importing decord): import decord_cv2_shim; decord_cv2_shim.install() or injected
via sitecustomize / environment variables, see gen_images_arm.sh."""
import os
import sys
import types

import cv2
import numpy as np


class _NDArrayLike:
    "Wrap numpy arrays to provide decord's .asnumpy() interface."

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, i):
        return self._arr[i]

    @property
    def shape(self):
        return self._arr.shape


class VideoReader:
    """Minimal implementation based on cv2.VideoCapture. num_threads is passed to cv2:
    cv2's ffmpeg backend defaults to open decoding threads based on CPU cores. On a
    machine with 192 cores, single process tests show 128 threads, and 12 processes
    running concurrently is ~1500 threads competing for 192 cores, resulting in load
    reaching 300+ but low output (same issue as DECORD_NUM_THREADS in ask_vllm.py).
    Therefore, explicit convergence is required."""

    def __init__(self, path, ctx=None, num_threads=None, **kwargs):
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2 无法打开视频: {self._path}")
        # This attribute requires OpenCV 4.5.4+; older versions silently fail, but does
        # not affect correctness.
        if num_threads:
            try:
                self._cap.set(cv2.CAP_PROP_N_THREADS, int(num_threads))
            except Exception:
                pass
        self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if self._n <= 0:
            raise RuntimeError(f"cv2 读到的帧数非法({self._n}): {self._path}")

    def __len__(self):
        return self._n

    def get_avg_fps(self):
        return self._fps

    def _read_at(self, idx):
        idx = max(0, min(int(idx), self._n - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Metadata near the end of a clip often has errors; fallback to one frame
            # before the end to avoid failure due to inability to read the last frame.
            for back in (1, 2, 5, 10):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - back))
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    break
        if not ok or frame is None:
            raise RuntimeError(f"cv2 读帧失败 idx={idx}: {self._path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        return _NDArrayLike(self._read_at(idx))

    def get_batch(self, indices):
        # Sequential reading is much faster than random seek, but results must be
        # returned in the order specified by the caller; otherwise frame numbers and
        # images will be out of sync. Therefore, decode in order and then restore the
        # original sequence.
        idx_list = [int(i) for i in indices]
        order = sorted(range(len(idx_list)), key=lambda k: idx_list[k])
        frames = [None] * len(idx_list)
        for k in order:
            frames[k] = self._read_at(idx_list[k])
        return _NDArrayLike(np.stack(frames, axis=0))

    def release(self):
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class _Context:
    def __init__(self, device_id=0):
        self.device_id = device_id


def cpu(device_id=0):
    return _Context(device_id)


def gpu(device_id=0):
    return _Context(device_id)


def bridge_noop(*_args, **_kwargs):
    return None


def install(force=False):
    """Register this module as `decord`, so that `from decord import VideoReader, cpu`
    works. When a valid decord is already available, it will not be overwritten
    (force=True can be used to force overwrite), ensuring the original implementation is
    used on x86 machines."""
    # cv2's internal parallel operations (resize/cvtColor, etc.) default to using thread
    # pools based on the number of cores. When generating images with multiprocessing,
    # this layer will be amplified proportionally; therefore, it should be converged
    # with the decoding threads.
    try:
        cv2.setNumThreads(int(os.environ.get("DECORD_NUM_THREADS", "2")))
    except Exception:
        pass

    if not force:
        try:
            import decord  # noqa: F401
            return False
        except Exception:
            pass

    mod = types.ModuleType("decord")
    mod.VideoReader = VideoReader
    mod.cpu = cpu
    mod.gpu = gpu
    mod.__version__ = "0.0.0+cv2shim"

    bridge = types.ModuleType("decord.bridge")
    bridge.set_bridge = bridge_noop
    mod.bridge = bridge

    sys.modules["decord"] = mod
    sys.modules["decord.bridge"] = bridge
    return True
