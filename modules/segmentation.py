
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


# Load U2Net model only once
_session = None


def get_rembg_session():
    global _session

    if _session is None:
        _session = new_session("u2net")

    return _session


def grabcut_segmentation(image, iterations=5):
    """
    Classical GrabCut segmentation.
    Input: RGB image
    Output: Binary mask (0 background, 255 foreground)
    """

    image = image.copy()

    height, width = image.shape[:2]

    if height < 10 or width < 10:
        raise ValueError("Image is too small for GrabCut.")

    margin_x = max(1, int(width * 0.05))
    margin_y = max(1, int(height * 0.05))

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    # GrabCut requires BGR input
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = np.zeros((height, width), np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            bgr,
            mask,
            rect,
            bg_model,
            fg_model,
            iterations,
            cv2.GC_INIT_WITH_RECT
        )

        binary_mask = np.where(
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        return binary_mask

    except cv2.error as error:
        raise RuntimeError(
            f"GrabCut segmentation failed: {error}"
        )


def rembg_segmentation(image):
    """
    Pretrained U2Net segmentation using rembg.

    Input: RGB image
    Returns:
        binary_mask: Grayscale binary mask
        rgba: Original image with transparent background
    """

    try:
        pil_image = Image.fromarray(
            image.astype(np.uint8)
        ).convert("RGB")

        session = get_rembg_session()

        result = remove(
            pil_image,
            session=session
        )

        result = result.convert("RGBA")

        rgba = np.array(result)

        alpha_mask = rgba[:, :, 3]

        binary_mask = np.where(
            alpha_mask > 127,
            255,
            0
        ).astype(np.uint8)

        return binary_mask, rgba

    except Exception as error:
        raise RuntimeError(
            f"U2Net segmentation failed: {error}"
        )


def apply_mask(image, mask):
    """Apply binary mask to RGB image and return RGBA."""

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    rgba = np.dstack([
        image,
        mask
    ])

    return rgba


def get_segmentation_statistics(mask):
    """Calculate foreground and background percentages."""

    total_pixels = mask.size

    foreground_pixels = np.count_nonzero(mask)
    background_pixels = total_pixels - foreground_pixels

    return {
        "Foreground Percentage": round(
            foreground_pixels / total_pixels * 100, 2
        ),
        "Background Percentage": round(
            background_pixels / total_pixels * 100, 2
        )
    }
