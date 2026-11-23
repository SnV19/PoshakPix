
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab


def prepare_comparison(image1, image2, mask=None,
                       size=(256, 256)):
    """Align images and mask to a common resolution."""

    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)

    if mask is not None:
        mask = cv2.resize(
            mask,
            size,
            interpolation=cv2.INTER_NEAREST
        )

        if mask.ndim == 3:
            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_RGB2GRAY
            )

        mask = (mask > 127).astype(np.uint8)

    return image1, image2, mask


def calculate_ssim(image1, image2, mask=None):
    """Calculate SSIM over the garment region."""

    image1, image2, mask = prepare_comparison(
        image1, image2, mask
    )

    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    score, similarity_map = structural_similarity(
        gray1,
        gray2,
        data_range=255,
        full=True
    )

    if mask is not None and np.any(mask):
        return float(
            similarity_map[mask > 0].mean()
        )

    return float(score)


def calculate_mae(image1, image2, mask=None):
    """Calculate mean absolute pixel error."""

    image1, image2, mask = prepare_comparison(
        image1, image2, mask
    )

    difference = np.abs(
        image1.astype(np.float32)
        - image2.astype(np.float32)
    )

    if mask is not None:
        valid = mask > 0

        if not np.any(valid):
            return 0.0

        return float(difference[valid].mean())

    return float(difference.mean())


def calculate_histogram_similarity(image1, image2,
                                   mask=None):
    """Compare RGB histograms using Bhattacharyya similarity."""

    image1, image2, mask = prepare_comparison(
        image1, image2, mask
    )

    mask_cv = None

    if mask is not None:
        mask_cv = mask * 255

        if not np.any(mask_cv):
            return 0.0

    scores = []

    for channel in range(3):

        hist1 = cv2.calcHist(
            [image1],
            [channel],
            mask_cv,
            [256],
            [0, 256]
        )

        hist2 = cv2.calcHist(
            [image2],
            [channel],
            mask_cv,
            [256],
            [0, 256]
        )

        cv2.normalize(hist1, hist1, alpha=1,
                      norm_type=cv2.NORM_L1)

        cv2.normalize(hist2, hist2, alpha=1,
                      norm_type=cv2.NORM_L1)

        distance = cv2.compareHist(
            hist1,
            hist2,
            cv2.HISTCMP_BHATTACHARYYA
        )

        scores.append(max(0.0, 1.0 - distance))

    return float(np.mean(scores))


def calculate_colour_difference(image1, image2,
                                 mask=None):
    """Calculate mean CIE Delta E 76 over garment pixels."""

    image1, image2, mask = prepare_comparison(
        image1, image2, mask
    )

    lab1 = rgb2lab(
        image1.astype(np.float32) / 255.0
    )

    lab2 = rgb2lab(
        image2.astype(np.float32) / 255.0
    )

    delta = np.sqrt(
        np.sum((lab1 - lab2) ** 2, axis=2)
    )

    if mask is not None:
        valid = mask > 0

        if not np.any(valid):
            return 0.0

        return float(delta[valid].mean())

    return float(delta.mean())


def calculate_clip_similarity(image1, image2):
    """Optional semantic similarity using pretrained CLIP."""

    try:
        import torch
        from transformers import (
            CLIPModel,
            CLIPProcessor
        )
        from PIL import Image

        model_name = "openai/clip-vit-base-patch32"

        processor = CLIPProcessor.from_pretrained(
            model_name
        )

        model = CLIPModel.from_pretrained(
            model_name
        )

        model.eval()

        images = [
            Image.fromarray(image1.astype(np.uint8)),
            Image.fromarray(image2.astype(np.uint8))
        ]

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        similarity = torch.sum(
            features[0] * features[1]
        ).item()

        return float(similarity)

    except Exception as error:
        print(f"CLIP evaluation unavailable: {error}")
        return None


def evaluate_garment(original, processed, mask=None):
    """Calculate the main garment consistency metrics."""

    return {
        "SSIM": round(
            calculate_ssim(original, processed, mask), 4
        ),
        "MAE": round(
            calculate_mae(original, processed, mask), 4
        ),
        "Histogram Similarity": round(
            calculate_histogram_similarity(
                original, processed, mask
            ), 4
        ),
        "Mean Delta E 76": round(
            calculate_colour_difference(
                original, processed, mask
            ), 4
        )
    }

