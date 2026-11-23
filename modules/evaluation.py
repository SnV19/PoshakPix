
import cv2
import numpy as np
import pandas as pd
import time


def calculate_iou(predicted_mask, ground_truth):
    """Calculate Intersection over Union."""

    pred = predicted_mask > 127
    truth = ground_truth > 127

    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def calculate_dice(predicted_mask, ground_truth):
    """Calculate Dice coefficient."""

    pred = predicted_mask > 127
    truth = ground_truth > 127

    intersection = np.logical_and(pred, truth).sum()

    denominator = pred.sum() + truth.sum()

    if denominator == 0:
        return 1.0

    return float(2 * intersection / denominator)


def calculate_foreground_percentage(mask):
    """Calculate foreground percentage."""

    return float(np.mean(mask > 127) * 100)


def measure_execution_time(function, *args, **kwargs):
    """Measure actual function execution time."""

    start = time.perf_counter()

    result = function(*args, **kwargs)

    elapsed = time.perf_counter() - start

    return result, round(elapsed, 4)


def create_comparison_table(results):
    """Convert experimental results into a Pandas DataFrame."""

    return pd.DataFrame(results)


def save_results_csv(dataframe, filepath):
    """Save experimental results as CSV."""

    dataframe.to_csv(
        filepath,
        index=False
    )

    return filepath


def compare_segmentation_methods(
    grabcut_mask,
    rembg_mask,
    ground_truth=None
):
    """
    Compare two segmentation masks.

    IoU and Dice are calculated only when
    a ground-truth mask is supplied.
    """

    results = []

    methods = {
        "GrabCut": grabcut_mask,
        "rembg U2Net": rembg_mask
    }

    for method_name, mask in methods.items():

        row = {
            "Method": method_name,
            "Foreground Percentage":
                round(calculate_foreground_percentage(mask), 2)
        }

        if ground_truth is not None:

            row["IoU"] = round(
                calculate_iou(mask, ground_truth), 4
            )

            row["Dice"] = round(
                calculate_dice(mask, ground_truth), 4
            )

        else:

            row["IoU"] = "Not available"
            row["Dice"] = "Not available"

        results.append(row)

    return pd.DataFrame(results)


def create_experiment_summary(experiment_name, metrics):
    """Create a standard experiment result dictionary."""

    return {
        "Experiment": experiment_name,
        **metrics
    }
