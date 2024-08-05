import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(data: list[tuple[float, int, float, float]]) -> None:
    # Convert data to DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "Rejection rate",
            "Decision token",
            "Average token rate",
            "Average score",
        ],
    )

    # Pivot the data for plotting
    pivot_token_rate = df.pivot(
        index="Rejection rate", columns="Decision token", values="Average token rate"
    )
    pivot_score = df.pivot(
        index="Rejection rate", columns="Decision token", values="Average score"
    )

    # Plot the heatmaps
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

    # Plot the Average token rate heatmap
    im1 = axes[0].imshow(pivot_token_rate, aspect="auto", cmap="viridis")
    axes[0].set_title("Average token rate")
    axes[0].set_xlabel("Decision token")
    axes[0].set_ylabel("Rejection rate")
    axes[0].set_xticks(np.arange(len(pivot_token_rate.columns)))
    axes[0].set_xticklabels(pivot_token_rate.columns)
    axes[0].set_yticks(np.arange(len(pivot_token_rate.index)))
    axes[0].set_yticklabels(pivot_token_rate.index)

    # Add text annotations
    for i in range(len(pivot_token_rate.index)):
        for j in range(len(pivot_token_rate.columns)):
            text = axes[0].text(
                j,
                i,
                f"{pivot_token_rate.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im1, ax=axes[0])

    # Plot the Average score heatmap
    im2 = axes[1].imshow(pivot_score, aspect="auto", cmap="inferno")
    axes[1].set_title("Average score")
    axes[1].set_xlabel("Decision token")
    axes[1].set_ylabel("Rejection rate")
    axes[1].set_xticks(np.arange(len(pivot_score.columns)))
    axes[1].set_xticklabels(pivot_score.columns)
    axes[1].set_yticks(np.arange(len(pivot_score.index)))
    axes[1].set_yticklabels(pivot_score.index)

    # Add text annotations
    for i in range(len(pivot_score.index)):
        for j in range(len(pivot_score.columns)):
            text = axes[1].text(
                j,
                i,
                f"{pivot_score.iloc[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()
