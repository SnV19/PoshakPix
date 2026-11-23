
import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from modules.segmentation import (
    grabcut_segmentation,
    rembg_segmentation
)

from modules.morphology import refine_mask

from modules.evaluation import (
    calculate_iou,
    calculate_dice
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(ROOT, "data", "sample_images")
GT_DIR = os.path.join(ROOT, "data", "ground_truth")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_rgb(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_experiments():

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        print("No sample images found in data/sample_images.")
        return

    results = []

    for filename in files:

        print(f"\nProcessing: {filename}")

        path = os.path.join(INPUT_DIR, filename)

        image = read_rgb(path)

        # GrabCut
        start = time.perf_counter()
        grabcut_mask = grabcut_segmentation(image)
        grabcut_time = time.perf_counter() - start

        # rembg
        start = time.perf_counter()
        rembg_mask, _ = rembg_segmentation(image)
        rembg_time = time.perf_counter() - start

        # Morphological refinement
        start = time.perf_counter()
        refined = refine_mask(
            rembg_mask,
            kernel_size=3
        )["refined_mask"]
        morphology_time = time.perf_counter() - start

        gt_path = os.path.join(
            GT_DIR,
            os.path.splitext(filename)[0] + ".png"
        )

        has_ground_truth = os.path.exists(gt_path)

        if has_ground_truth:

            gt = cv2.imread(
                gt_path,
                cv2.IMREAD_GRAYSCALE
            )

            gt = cv2.resize(
                gt,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            gt = np.where(gt > 127, 255, 0).astype(np.uint8)

        methods = [
            ("GrabCut", grabcut_mask, grabcut_time),
            ("rembg", rembg_mask, rembg_time),
            ("rembg + Morphology", refined,
             rembg_time + morphology_time)
        ]

        for name, mask, elapsed in methods:

            row = {
                "Image": filename,
                "Method": name,
                "Runtime (seconds)": round(elapsed, 4),
                "Foreground Percentage": round(
                    np.mean(mask > 127) * 100,
                    2
                )
            }

            if has_ground_truth:

                row["IoU"] = round(
                    calculate_iou(mask, gt),
                    4
                )

                row["Dice"] = round(
                    calculate_dice(mask, gt),
                    4
                )

            else:

                row["IoU"] = np.nan
                row["Dice"] = np.nan

            results.append(row)

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        OUTPUT_DIR,
        "experiment_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\nExperiment Results:")
    print(df.to_string(index=False))

    # Average runtime
    runtime = df.groupby(
        "Method"
    )["Runtime (seconds)"].mean()

    plt.figure(figsize=(9, 5))

    runtime.plot(kind="bar")

    plt.title("Average Segmentation Runtime")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Method")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "runtime_comparison.png"
        ),
        dpi=200
    )

    plt.close()

    # Foreground percentage
    foreground = df.groupby(
        "Method"
    )["Foreground Percentage"].mean()

    plt.figure(figsize=(9, 5))

    foreground.plot(kind="bar")

    plt.title("Average Foreground Percentage")
    plt.ylabel("Foreground (%)")
    plt.xlabel("Method")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "foreground_comparison.png"
        ),
        dpi=200
    )

    plt.close()

    # Accuracy comparison only if ground truth exists
    valid = df.dropna(subset=["IoU", "Dice"])

    if not valid.empty:

        grouped = valid.groupby("Method")[["IoU", "Dice"]].mean()

        grouped.plot(
            kind="bar",
            figsize=(9, 5)
        )

        plt.title("Segmentation Accuracy Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        plt.xticks(rotation=15)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "accuracy_comparison.png"
            ),
            dpi=200
        )

        plt.close()

    print("\nExperiments completed.")

    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiments()
