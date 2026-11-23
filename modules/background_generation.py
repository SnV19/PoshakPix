
import cv2
import numpy as np


BACKGROUND_PRESETS = {
    "White": (255, 255, 255),
    "Light Grey": (235, 235, 235),
    "Pastel Pink": (255, 220, 230),
    "Pastel Blue": (210, 230, 255),
    "Pastel Green": (220, 245, 225),
    "Cream": (250, 240, 220),
    "Black": (30, 30, 30)
}


def create_solid_background(width, height, color):
    """Create a solid RGB background."""
    background = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    background[:] = color

    return background


def create_gradient_background(width, height, color1, color2):
    """Create a vertical RGB gradient."""
    background = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for y in range(height):
        alpha = y / max(height - 1, 1)

        color = (
            np.array(color1) * (1 - alpha)
            + np.array(color2) * alpha
        )

        background[y, :] = color.astype(np.uint8)

    return background


def create_background(width, height, background_type="White",
                      custom_color=(255, 255, 255)):

    """Generate the selected background."""

    if background_type == "Gradient":
        return create_gradient_background(
            width,
            height,
            (255, 230, 240),
            (210, 230, 255)
        )

    if background_type == "Custom":
        color = custom_color

    else:
        color = BACKGROUND_PRESETS.get(
            background_type,
            (255, 255, 255)
        )

    return create_solid_background(
        width,
        height,
        color
    )


def resize_foreground(image, mask, target_width, target_height,
                      scale=0.75):
    """Resize and center the garment while preserving aspect ratio."""

    h, w = image.shape[:2]

    scale_factor = min(
        target_width * scale / w,
        target_height * scale / h
    )

    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))

    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    resized_mask = cv2.resize(
        mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    alpha = np.zeros(
        (target_height, target_width),
        dtype=np.uint8
    )

    x = (target_width - new_w) // 2
    y = (target_height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized_image
    alpha[y:y + new_h, x:x + new_w] = resized_mask

    return canvas, alpha


def create_shadow(mask, strength=0.25, blur_size=31):
    """Generate a soft shadow from the garment mask."""

    shadow = np.zeros_like(mask, dtype=np.uint8)

    h, w = mask.shape

    matrix = np.float32([
        [1, 0, 0],
        [0, 1, max(1, int(h * 0.025))]
    ])

    shifted = cv2.warpAffine(
        mask,
        matrix,
        (w, h)
    )

    if blur_size % 2 == 0:
        blur_size += 1

    shadow = cv2.GaussianBlur(
        shifted,
        (blur_size, blur_size),
        0
    )

    return np.clip(
        shadow.astype(np.float32) * strength,
        0,
        255
    ).astype(np.uint8)


def composite_garment(image, mask, background,
                      add_shadow=True):
    """Place a garment on a new background using alpha compositing."""

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    if background.shape[:2] != image.shape[:2]:
        background = cv2.resize(
            background,
            (image.shape[1], image.shape[0])
        )

    alpha = mask.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0, 1)

    if add_shadow:
        shadow = create_shadow(mask)

        shadow_alpha = shadow.astype(np.float32) / 255.0

        background = (
            background.astype(np.float32) *
            (1 - shadow_alpha[:, :, None] * 0.35)
        )

    result = (
        image.astype(np.float32) * alpha[:, :, None]
        + background.astype(np.float32) *
        (1 - alpha[:, :, None])
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def inpaint_image(image, mask, radius=5,
                  method="TELEA"):
    """Demonstrate OpenCV inpainting on masked regions."""

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    flag = (
        cv2.INPAINT_TELEA
        if method == "TELEA"
        else cv2.INPAINT_NS
    )

    return cv2.inpaint(
        image,
        mask,
        radius,
        flag
    )
