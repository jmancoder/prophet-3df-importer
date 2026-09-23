import numpy as np
import numpy.typing as npt
import texture2ddecoder


def rgba_to_floats(pixels: npt.NDArray) -> npt.NDArray:
    return (pixels.astype(np.float32) / 255.0).ravel()


def dxt1_to_rgba(data: bytes, width: int, height: int) -> npt.NDArray:
    bgra = texture2ddecoder.decode_bc1(data, width, height)
    rgba = np.frombuffer(bgra, np.uint8).reshape(-1, 4)
    return rgba_to_floats(rgba[:, [2, 1, 0, 3]])
