from io import BytesIO
from typing import Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


def _open_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(raw))
        img.verify()
        return Image.open(BytesIO(raw))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Invalid image file: {exc}") from exc


def preprocess_image_for_emotion_model(
    raw: bytes,
    image_size: Tuple[int, int] = (128, 128),
    channels: int = 1,
) -> np.ndarray:
    """Replica el preprocesamiento usado en entrenamiento (ver Untitled0.ipynb):
    - Convertir a grayscale (canal único)
    - Redimensionar a IMG_SIZE
    - Normalizar /255.0
    - Devolver tensor shape (1, H, W, C)
    """
    img = _open_image(raw)
    mode = "L" if channels == 1 else "RGB"
    img = img.convert(mode).resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)
