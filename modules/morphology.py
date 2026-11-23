
import cv2
import numpy as np


def create_kernel(kernel_size=3):
    """Create a rectangular morphological kernel."""

    kernel_size = max(1, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )


def ensure_binary_mask(mask):
    """Convert input mask into a binary uint8 mask."""

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


def erosion(mask, kernel_size=3):
    mask = ensure_binary_mask(mask)

    return cv2.erode(
        mask,
        create_kernel(kernel_size),
        iterations=1
    )


def dilation(mask, kernel_size=3):
    mask = ensure_binary_mask(mask)

    return cv2.dilate(
        mask,
        create_kernel(kernel_size),
        iterations=1
    )


def opening(mask, kernel_size=3):
    mask = ensure_binary_mask(mask)

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        create_kernel(kernel_size)
    )


def closing(mask, kernel_size=3):
    mask = ensure_binary_mask(mask)

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        create_kernel(kernel_size)
    )


def remove_small_components(mask, min_area=100):
    """Remove small foreground connected components."""

    mask = ensure_binary_mask(mask)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    result = np.zeros_like(mask)

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            result[labels == i] = 255

    return result


def refine_mask(mask, kernel_size=3):
    """
    Perform opening followed by closing
    to remove noise and fill small gaps.
    """

    original = ensure_binary_mask(mask)

    opened = opening(
        original,
        kernel_size
    )

    refined = closing(
        opened,
        kernel_size
    )

    return {
        "original_mask": original,
        "opened_mask": opened,
        "refined_mask": refined
    }


def draw_contours(image, mask):
    """Draw segmentation boundaries on an RGB image."""

    image = image.copy()
    mask = ensure_binary_mask(mask)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        (255, 0, 0),
        2
    )

    return image

