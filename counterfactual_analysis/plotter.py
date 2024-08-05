import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data_from_file(file_path: str) -> list[tuple[float, int, float, float]]:
    with open(file_path, "r") as f:
        data: list[tuple[float, int, float, float]] = json.load(f)
    return data


def plot_data(output_filename: str) -> None:
    data = load_data_from_file(output_filename)

    font_size = 14
    fig_size = (19, 6)

    # thresholds for changing text color
    threshold_token_rate = 0.5
    threshold_score = 90

    plt.close("all")

    # Convert data to DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "Rejection Rate",
            "Decision Token",
            "Average Token Rate",
            "Average Score",
        ],
    )

    # Pivot the data for plotting
    pivot_token_rate = df.pivot(
        index="Rejection Rate", columns="Decision Token", values="Average Token Rate"
    )
    pivot_score = df.pivot(
        index="Rejection Rate", columns="Decision Token", values="Average Score"
    )

    # Function to determine text color based on background color
    def get_text_color(value, threshold):
        return "white" if value < threshold else "black"

    # Plot the heatmaps
    fig, axes = plt.subplots(ncols=2, figsize=fig_size)

    # Get the current figure manager
    manager = plt.get_current_fig_manager()

    # Set the position and size of the figure (widthxheight+x+y)
    manager.window.wm_geometry("+0+0")

    # Set font size
    plt.rcParams.update({"font.size": font_size})

    # Plot the Average Token Rate heatmap
    im1 = axes[0].imshow(pivot_token_rate, aspect="auto", cmap="viridis")
    axes[0].set_title("Average Token Rate", fontsize=font_size)
    axes[0].set_xlabel("Decision Token", fontsize=font_size)
    axes[0].set_ylabel("Rejection Rate", fontsize=font_size)
    axes[0].set_xticks(np.arange(len(pivot_token_rate.columns)))
    axes[0].set_xticklabels(pivot_token_rate.columns, fontsize=font_size)
    axes[0].set_yticks(np.arange(len(pivot_token_rate.index)))
    axes[0].set_yticklabels(pivot_token_rate.index, fontsize=font_size)

    # Add text annotations
    for i in range(len(pivot_token_rate.index)):
        for j in range(len(pivot_token_rate.columns)):
            color = get_text_color(pivot_token_rate.iloc[i, j], threshold_token_rate)
            text = axes[0].text(
                j,
                i,
                f"{pivot_token_rate.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color=color,
            )

    # Remove black border
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    # Remove colorbar border
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.outline.set_visible(False)

    # Plot the Average Score heatmap
    im2 = axes[1].imshow(pivot_score, aspect="auto", cmap="inferno")
    axes[1].set_title("Average Score", fontsize=font_size)
    axes[1].set_xlabel("Decision Token", fontsize=font_size)
    axes[1].set_ylabel("Rejection Rate", fontsize=font_size)
    axes[1].set_xticks(np.arange(len(pivot_score.columns)))
    axes[1].set_xticklabels(pivot_score.columns, fontsize=font_size)
    axes[1].set_yticks(np.arange(len(pivot_score.index)))
    axes[1].set_yticklabels(pivot_score.index, fontsize=font_size)

    # Add text annotations
    for i in range(len(pivot_score.index)):
        for j in range(len(pivot_score.columns)):
            color = get_text_color(pivot_score.iloc[i, j], threshold_score)
            text = axes[1].text(
                j,
                i,
                f"{pivot_score.iloc[i, j]:.1f}",
                ha="center",
                va="center",
                color=color,
            )

    # Remove black border
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    # Remove colorbar border
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_filename[:-5]}.png", dpi=300)
    # plt.show()
