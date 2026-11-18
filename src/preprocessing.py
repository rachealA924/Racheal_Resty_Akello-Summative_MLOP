"""
preprocessing.py: everything that turns raw image files into model inputs.

The same functions are used in the notebook, at prediction time and during
retraining, so an image is always processed exactly the same way.

Steps for one image:
  1. open the file and check it is a real, readable image
  2. convert to RGB (removes transparency, turns grayscale into 3 channels)
  3. resize to 224 x 224 (the size MobileNetV2 expects)
  4. keep pixel values as 0..255 floats; the model scales them to -1..1 itself
     (a Rescaling layer inside the model), so callers never have to remember it
"""

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import ALLOWED_EXTENSIONS, CLASS_NAMES, IMG_SIZE


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def load_image(source, size=IMG_SIZE):
    """
    Load an image from a path, raw bytes or a file object.
    Returns a float32 array of shape (size, size, 3) with values 0..255.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img = ImageOps.exif_transpose(img)   # respect phone camera rotation
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError("The file is not a readable JPG or PNG image.") from e
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def load_batch(paths, size=IMG_SIZE):
    """Load many images into one array of shape (n, size, size, 3). Unreadable files are skipped."""
    arrays, kept = [], []
    for p in paths:
        try:
            arrays.append(load_image(p, size))
            kept.append(p)
        except InvalidImageError:
            continue
    if not arrays:
        return np.zeros((0, size, size, 3), dtype=np.float32), []
    return np.stack(arrays), kept


def list_labelled_images(root):
    """
    Walk a folder laid out as root/<class_name>/<image files>.
    Returns two lists: file paths and integer labels (index in CLASS_NAMES).
    """
    root = Path(root)
    paths, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        folder = root / name
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in ALLOWED_EXTENSIONS:
                paths.append(f)
                labels.append(label)
    return paths, np.array(labels, dtype=np.int64)


def save_upload(data: bytes, label: str, upload_dir, original_name="image.jpg"):
    """
    Validate an uploaded image and save it as a clean JPEG in upload_dir/<label>/.
    Saving a re-encoded copy (not the raw bytes) means corrupt or disguised files
    never reach the training data. Returns the saved path.
    """
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown label '{label}'. Use one of {CLASS_NAMES}")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(f"{original_name} is not a readable image") from e
    # keep a reasonable size on disk; training resizes to 224 anyway
    img.thumbnail((800, 800))
    folder = Path(upload_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex[:12]}.jpg"
    img.save(path, "JPEG", quality=90)
    return path


#  SIMPLE IMAGE FEATURES, USED FOR THE DATA INSIGHT CHARTS (NOT FOR THE MODEL).
def image_statistics(path):
    """
    Three easy-to-explain features of a leaf photo:
      brightness  : mean pixel value of the grayscale image (0 dark .. 255 bright)
      greenness   : share of green in the colour, G / (R + G + B) (0.33 means neutral)
      yellow_ratio: share of pixels that look yellow or brown (R and G high, B low),
                    a rough sign of discoloured, diseased tissue
      contrast    : standard deviation of grayscale pixels (texture / spots)
    """
    arr = load_image(path, size=128)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    total = r + g + b + 1e-6
    yellow = (r > 100) & (g > 100) & (b < 0.6 * g) & (np.abs(r - g) < 60)
    return {
        "brightness": float(gray.mean()),
        "greenness": float((g / total).mean()),
        "yellow_ratio": float(yellow.mean()),
        "contrast": float(gray.std()),
    }
