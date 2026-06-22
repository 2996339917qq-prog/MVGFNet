import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return x


def get_first_batch(x):
    """
    支持:
    score: [B, T, N, N] 或 [T, N, N]
    emb:   [B, T, N, C] 或 [T, N, C]
    graph: [N, N] 或 [B, T, N, C]
    """
    x = to_numpy(x)
    if x.ndim == 4:
        x = x[0]
    return x


def plot_attention_grid(score03, score06, save_path):
    """
    score03: [B, T, N, N] 或 [T, N, N]
    score06: [B, T, N, N] 或 [T, N, N]
    """

    score03 = get_first_batch(score03)  # [T, N, N]
    score06 = get_first_batch(score06)  # [T, N, N]

    steps = [3, 6, 12]

    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    for col, step in enumerate(steps):
        t = step - 1

        im = axes[0, col].imshow(score03[t], aspect="auto")
        axes[0, col].set_title(f"conv1 at step{step}", fontsize=16)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(score06[t], aspect="auto")
        axes[1, col].set_title(f"conv2 at step{step}", fontsize=16)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

        im = axes[2, col].imshow(score03[t], aspect="auto")
        axes[2, col].set_title(f"conv1 at step{step}", fontsize=16)
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
        plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

        im = axes[3, col].imshow(score06[t], aspect="auto")
        axes[3, col].set_title(f"conv2 at step{step}", fontsize=16)
        axes[3, col].set_xticks([])
        axes[3, col].set_yticks([])
        plt.colorbar(im, ax=axes[3, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def tsne_2d(x):
    """
    x: [N, C] 或 [N, N]
    """
    x = to_numpy(x)

    if x.ndim == 3:
        x = x[0]  # [N, C]

    n = x.shape[0]
    perplexity = max(5, min(30, n // 3))

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    return model.fit_transform(x)


def plot_graph_embedding_grid(
    node_forward1,
    node_backward1,
    node_forward2,
    node_backward2,
    save_path,
    time_index=0
):
    """
    node_forward1:  [N, N] 或 [B, T, N, C]
    node_backward1: [N, N] 或 [B, T, N, C]
    node_forward2:  [B, T, N, C] 或 [T, N, C]
    node_backward2: [B, T, N, C] 或 [T, N, C]
    """

    node_forward1 = to_numpy(node_forward1)
    node_backward1 = to_numpy(node_backward1)
    node_forward2 = get_first_batch(node_forward2)      # [T, N, C]
    node_backward2 = get_first_batch(node_backward2)    # [T, N, C]

    if node_forward1.ndim == 4:
        node_forward1 = node_forward1[0, time_index]
    if node_backward1.ndim == 4:
        node_backward1 = node_backward1[0, time_index]

    forward_predefined = tsne_2d(node_forward1)
    backward_predefined = tsne_2d(node_backward1)

    forward_learned = tsne_2d(node_forward2[time_index])
    backward_learned = tsne_2d(node_backward2[time_index])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    data_list = [
        (forward_predefined, "Forward Predefined Graph Embedding"),
        (backward_predefined, "Backward Predefined Graph Embedding"),
        (forward_learned, "Forward Learned Graph Embedding"),
        (backward_learned, "Backward Learned Graph Embedding"),
    ]

    for ax, (data, title) in zip(axes.flat, data_list):
        ax.scatter(data[:, 0], data[:, 1], s=35, edgecolors="black", linewidths=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    mkdir("./figures")

    score03 = np.load("./visual_results/score03.npy")
    score06 = np.load("./visual_results/score06.npy")

    node_forward1 = np.load("./visual_results/node_forward1.npy")
    node_backward1 = np.load("./visual_results/node_backward1.npy")
    node_forward2 = np.load("./visual_results/node_forward2.npy")
    node_backward2 = np.load("./visual_results/node_backward2.npy")

    plot_attention_grid(
        score03=score03,
        score06=score06,
        save_path="./figures/attention_visualization.png"
    )

    plot_graph_embedding_grid(
        node_forward1=node_forward1,
        node_backward1=node_backward1,
        node_forward2=node_forward2,
        node_backward2=node_backward2,
        save_path="./figures/graph_embedding_visualization.png",
        time_index=0
    )

    print("Finished.")
