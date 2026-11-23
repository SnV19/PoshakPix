
import matplotlib.pyplot as plt
import numpy as np


def plot_image_comparison(images, titles, figsize=(15, 5)):
    """Display multiple images side by side."""

    count = len(images)

    fig, axes = plt.subplots(
        1,
        count,
        figsize=figsize
    )

    if count == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):

        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    return fig


def plot_metric_comparison(names, values, title,
                           ylabel="Score"):
    """Generate a bar chart comparing numerical metrics."""

    fig, ax = plt.subplots(figsize=(8, 5))

    numeric_values = [
        float(value) if isinstance(
            value, (int, float, np.integer, np.floating)
        ) else 0
        for value in values
    ]

    bars = ax.bar(
        names,
        numeric_values
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)

    ax.bar_label(
        bars,
        fmt="%.3f"
    )

    plt.xticks(rotation=15)
    plt.tight_layout()

    return fig


def plot_processing_times(methods, times):
    """Compare execution time across methods."""

    return plot_metric_comparison(
        methods,
        times,
        "Processing Time Comparison",
        "Time (seconds)"
    )


def plot_segmentation_metrics(methods, iou, dice):
    """Compare IoU and Dice scores."""

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        iou,
        width,
        label="IoU"
    )

    ax.bar(
        x + width / 2,
        dice,
        width,
        label="Dice"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Segmentation Performance")

    ax.legend()

    plt.tight_layout()

    return fig
