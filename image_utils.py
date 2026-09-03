import numpy as np
import numpy.typing as npt
import texture2ddecoder


def rgba_to_floats(pixels: npt.NDArray) -> npt.NDArray:
    return pixels.astype(np.float32) / 255.0


def dxt1_to_rgba(data: bytes, width: int, height: int) -> npt.NDArray:
    rgba_raw = texture2ddecoder.decode_bc1(data, width, height)
    rgba_array = np.frombuffer(rgba_raw, np.uint8)
    return rgba_to_floats(rgba_array)
