
import numpy as np
import cv2
from PIL import Image


def load_image(image_file):
    """Load an uploaded image and return RGB and BGR arrays."""
    pil_image = Image.open(image_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_rgb, image_bgr


def resize_image(image, max_size=768):
    """Resize an image while preserving its aspect ratio."""
    height, width = image.shape[:2]

    if max(height, width) <= max_size:
        return image.copy()

    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def convert_to_rgb(image):
    """Convert a BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gaussian_denoising(image, kernel_size=3):
    """Apply Gaussian denoising."""
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def normalize_image(image):
    """Normalize image pixel values to the range 0–255."""
    return cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)



def preprocess_image(image, max_size=768):
    """Run resizing, denoising, and normalization."""

    resized = resize_image(
        image,
        max_size=max_size
    )

    denoised = gaussian_denoising(resized)

    normalized = normalize_image(denoised)

    return {
        "resized": resized,
        "denoised": denoised,
        "normalized": normalized
    }



def get_image_information(image):
    """Return basic image information."""
    height, width = image.shape[:2]

    return {
        "width": width,
        "height": height,
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "shape": image.shape,
        "dtype": str(image.dtype)
    }

