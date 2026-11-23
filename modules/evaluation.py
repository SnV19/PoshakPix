
import cv2
import numpy as np
import pandas as pd
import time

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
    rembg_mask
):
    """
    Compare two segmentation masks using
    foreground percentage.
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

        results.append(row)

    return pd.DataFrame(results)


def create_experiment_summary(experiment_name, metrics):
    """Create a standard experiment result dictionary."""

    return {
        "Experiment": experiment_name,
        **metrics
    }

