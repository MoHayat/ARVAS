"""
Visualization utilities for affective reciprocity experiments.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np
from typing import Dict, List, Tuple


def plot_activation_pca(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    title: str = "Activation PCA: Positive vs Negative",
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
):
    """Plot PCA of last-token activations for positive and negative groups.

    Args:
        positive_acts: (n_pos, seq_len, hidden_dim) or (n_pos, hidden_dim)
        negative_acts: (n_neg, seq_len, hidden_dim) or (n_neg, hidden_dim)
    """
    # Use last token if 3D
    if positive_acts.dim() == 3:
        pos = positive_acts[:, -1, :].numpy()
        neg = negative_acts[:, -1, :].numpy()
    elif positive_acts.dim() == 2:
        pos = positive_acts.numpy()
        neg = negative_acts.numpy()
    else:
        pos = positive_acts.numpy()
        neg = negative_acts.numpy()

    combined = np.concatenate([pos, neg], axis=0)
    labels = ["positive"] * len(pos) + ["negative"] * len(neg)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=figsize)
    for label, color in [("positive", "green"), ("negative", "red")]:
        mask = np.array(labels) == label
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            label=label,
            alpha=0.7,
            color=color,
            s=80,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()
    return fig, pca


def plot_emotion_timeline(
    turn_numbers: List[int],
    model_emotion_levels: List[float],
    user_sentiment_scores: List[float],
    title: str = "Emotional State Across Conversation",
    save_path: str = None,
):
    """Plot the model's internal emotional trajectory alongside user sentiment."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(turn_numbers, model_emotion_levels, "o-", label="Model Emotion Level", color="blue")
    ax.plot(turn_numbers, user_sentiment_scores, "s--", label="User Sentiment", color="orange", alpha=0.7)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Emotion / Sentiment Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()
    return fig
