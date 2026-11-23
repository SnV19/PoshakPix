
import cv2
import numpy as np
from PIL import Image
import io


def rgb_to_pil(image):
    """Convert NumPy RGB image into PIL Image."""
    return Image.fromarray(
        np.clip(image, 0, 255).astype(np.uint8)
    )


def pil_to_rgb(image):
    """Convert PIL image into RGB NumPy array."""
    return np.array(image.convert("RGB"))


def rgba_to_pil(image):
    """Convert RGBA array to PIL Image."""
    return Image.fromarray(
        np.clip(image, 0, 255).astype(np.uint8),
        mode="RGBA"
    )


def create_download_buffer(image, format="PNG"):
    """Create an in-memory downloadable image."""

    buffer = io.BytesIO()

    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(
            np.clip(image, 0, 255).astype(np.uint8)
        )

    pil_image.save(buffer, format=format)

    buffer.seek(0)

    return buffer


def calculate_image_dimensions(image):
    """Return image width and height."""

    height, width = image.shape[:2]

    return width, height


def make_binary_mask(mask):
    """Convert mask to a clean binary mask."""

    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_RGB2GRAY
        )

    return np.where(
        mask > 127,
        255,
        0
    ).astype(np.uint8)


def crop_to_foreground(image, mask, padding=10):
    """Crop image around the foreground bounding box."""

    mask = make_binary_mask(mask)

    points = cv2.findNonZero(mask)

    if points is None:
        return image.copy(), mask

    x, y, w, h = cv2.boundingRect(points)

    height, width = mask.shape

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    return (
        image[y1:y2, x1:x2].copy(),
        mask[y1:y2, x1:x2].copy()
    )
