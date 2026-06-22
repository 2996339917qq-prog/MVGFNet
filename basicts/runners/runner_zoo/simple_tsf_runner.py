import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

class SimpleTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # curriculum learning
        prediction_data= self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train)

        # if n == 2:
        #     import matplotlib.pyplot as plt
        #     import matplotlib as mpl
        #
        #     titles = [
        #         "conv1 at step3", "conv1 at step6", "conv1 at step12",
        #         "conv2 at step3", "conv2 at step6", "conv2 at step12",
        #     ]
        #
        #     # 调整figsize：更宽，稍矮
        #     fig, axes = plt.subplots(2, 3, figsize=(9, 5))  # 宽度拉长，竖直压短
        #
        #     # 调整布局，使横向空间略宽，竖向更紧凑
        #     plt.subplots_adjust(
        #         left=0.05, right=0.95,
        #         top=0.90, bottom=0.10,
        #         wspace=0.1, hspace=0.15
        #     )
        #
        #     mpl.rcParams['xtick.labelsize'] = 6
        #     mpl.rcParams['ytick.labelsize'] = 6
        #
        #     for a, ax in enumerate(axes.flat):
        #         if a < 3:
        #             visualize_attention_single_time_step(ax, score03, t=[2, 5, 11][a], selected_B=5,
        #                                                  title=titles[a])
        #         else:
        #             visualize_attention_single_time_step(ax, score06, t=[2, 5, 11][a - 3], selected_B=5,
        #                                                  title=titles[a])
        #
        #         ax.set_title(titles[a], fontsize=16, pad=2)
        #         ax.tick_params(
        #             labelsize=6,
        #             width=0.5, length=1.5,
        #             pad=1.5
        #         )
        #
        #     plt.savefig("s614", dpi=450, bbox_inches='tight', pad_inches=0.01)
        #     plt.show()

        # import matplotlib.pyplot as plt
        # from sklearn.decomposition import PCA
        # from sklearn.manifold import TSNE
        #
        # def reduce_to_2d(embedding: torch.Tensor, method='pca'):
        #     embedding_np = embedding.detach().cpu().numpy()
        #     if method == 'pca':
        #         reducer = PCA(n_components=2)
        #     elif method == 'tsne':
        #         reducer = TSNE(n_components=2, perplexity=5, random_state=42)
        #     else:
        #         raise NotImplementedError("Only 'pca' and 'tsne' are supported.")
        #     return reducer.fit_transform(embedding_np)
        #
        # def plot_embedding(ax, embedding_2d, title, color='blue'):
        #     ax.scatter(
        #         embedding_2d[:, 0], embedding_2d[:, 1],
        #         c=color,
        #         s=30,  # ✅ 减小点的大小（推荐 15~30）
        #         edgecolors='k',
        #         alpha=0.7
        #     )
        #     ax.set_title(title, fontsize=9)  # ✅ 子图标题字体减小
        #     ax.grid(True)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #
        # def visualize_all_node_embeddings(node_forward1, node_backward1, node_forward2, node_backward2, method='pca'):
        #     """
        #     node_forward1, node_backward1: [N, C] 原始图嵌入
        #     node_forward2, node_backward2: [B, T, N, C] 动态图嵌入
        #     """
        #     # 使用最后一帧动态嵌入，再对 batch 维平均，得到 [N, C]
        #     node_forward2_mean = node_forward2[:, -1].mean(dim=0)
        #     node_backward2_mean = node_backward2[:, -1].mean(dim=0)
        #
        #     # 降维
        #     emb_fwd1_2d = reduce_to_2d(node_forward1, method)
        #     emb_bwd1_2d = reduce_to_2d(node_backward1, method)
        #     emb_fwd2_2d = reduce_to_2d(node_forward2_mean, method)
        #     emb_bwd2_2d = reduce_to_2d(node_backward2_mean, method)
        #
        #     # 可视化布局
        #     fig, axs = plt.subplots(2, 2, figsize=(6, 5))
        #
        #     plot_embedding(axs[0, 0], emb_fwd1_2d, "Forward Predefined Graph Embedding", color='#007ACC')  # 深蓝
        #     plot_embedding(axs[0, 1], emb_bwd1_2d, "Backward Predefined Graph Embedding", color='#007ACC')
        #     plot_embedding(axs[1, 0], emb_fwd2_2d, "Forward Learned Graph Embedding", color='#00B3B3')  # 青色
        #     plot_embedding(axs[1, 1], emb_bwd2_2d, "Backward Learned Graph Embedding", color='#00B3B3')
        #
        #     plt.tight_layout()
        #     plt.savefig("graph64.pdf", format='pdf')  # 保存为 PDF 文件
        #     plt.show()
        #
        # # ✅ 调用可视化函数
        # visualize_all_node_embeddings(
        #     node_forward1=node_forward1,
        #     node_backward1=node_backward1,
        #     node_forward2=node_forward2,
        #     node_backward2=node_backward2,
        #     method='tsne'  # 可选 'pca' 或 'tsne'
        # )


        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value

def visualize_attention_single_time_step(ax, score, t=0, selected_B=0, title="Attention Score Heatmap"):
        """
        可视化某个时间步的注意力分数热力图。

        score: (B, T, N, N) 的注意力矩阵
        t: 选择的时间步，默认为第一个时间步 (t=0)
        selected_B: 选择的 batch，默认为第一个 batch
        """
        B, T, N, N = score.shape
        # 获取指定 batch 和时间步的注意力分数矩阵
        score_at_t = score[selected_B, t].detach().cpu().numpy()
        # 确保提取的数据是二维的
        if score_at_t.ndim != 2:
            raise ValueError(f"Expected 2D data for time step {t}, but got {score_at_t.ndim}D data.")
        # 在指定的子图中绘制热力图
        sns.heatmap(score_at_t, cmap="viridis", annot=False, ax=ax)
        ax.set_title(title)
        ax.set_xticks([])  # 隐藏x轴刻度
        ax.set_yticks([])  # 隐藏y轴刻度  这是可视化函数








