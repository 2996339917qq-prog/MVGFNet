import torch.nn as nn
import torch
from .mlp import MultiLayerPerceptron, GraphMLP
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import random
class GraphAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels, timestep_max, time_filter, dropout,show_scores):
        super().__init__()
        self.show_scores = show_scores
        # Keys, values and queries
        self.k = torch.nn.Conv2d(in_channels=in_channels,out_channels=timestep_max,kernel_size=(1, time_filter),stride=1)
        self.q = torch.nn.Conv2d(in_channels=in_channels,out_channels=timestep_max,kernel_size=(1, time_filter),stride=1)
        self.v = torch.nn.Conv2d(in_channels=in_channels,out_channels=timestep_max,kernel_size=(1, time_filter),stride=1)
        # Additional layers
        self.fc_res = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),stride=1)
        self.fc_out = torch.nn.Conv2d(in_channels=(timestep_max - time_filter) + 1,out_channels=out_channels,kernel_size=(1, 1), stride=1)
        # Activation, Normalization and Dropout
        self.act = torch.nn.Softmax(dim=-1)
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.dropout =torch.nn.Dropout(dropout)
        # To remove
        self.score_style = True
    def forward(self, x):
        k = self.k(x)  # B, T', N, (T - t + 1)
        q = self.q(x)  # B, T', N, (T - t + 1)
        v = self.v(x)  # B, T', N, (T - t + 1)
        score0 = torch.einsum("BTNC, BTnC -> BTNn", k, q).contiguous()  # B, T', N, N
        # if A is not None:
        #     score = score + A.unsqueeze(1)
        score = self.act(score0)
        if self.score_style:
            out = torch.einsum("BTnN, BTNC -> BCnT", score, v).contiguous()  # B, (T - t + 1), N, T'
        else:
            out = torch.einsum("BTNC, BTNn -> BCnT", v, score).contiguous()  # B, (T - t + 1), N, T'
        out = self.fc_out(out)  # B, C', N, T'
        res = self.fc_res(x)  # B, C', N, T'
        out = self.norm((out + res))  # B, C', N, T'
        out = self.dropout(out)  # B, C', N, T'
        if self.show_scores:
            return out, score0
        return out  # B, C', N, T'










class BLSTF(nn.Module):
    """
    Paper: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
    Link: https://arxiv.org/abs/2308.10425
    Official Code: https://github.com/XDZhelheim/STAEformer
    """

    def __init__(
            self,
            num_nodes,
            adj_mx,
            in_steps,
            out_steps,
            steps_per_day,
            input_dim,
            output_dim,
            input_embedding_dim,
            tod_embedding_dim,
            ts_embedding_dim,
            dow_embedding_dim,
            time_embedding_dim,
            adaptive_embedding_dim,
            node_dim,
            feed_forward_dim,
            out_feed_forward_dim,
            num_heads,
            num_layers,
            num_layers_m,
            mlp_num_layers,
            dropout,
            use_mixed_proj,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj_mx = adj_mx
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.ts_embedding_dim = ts_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.node_dim = node_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                # + adaptive_embedding_dim
                # + ts_embedding_dim
                + time_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.num_layers_m = num_layers_m
        self.dropout =  dropout
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(7 * steps_per_day, self.time_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        self.adj_mx_forward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )

        self.adj_mx_backward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )


        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)


        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps, out_channels=self.ts_embedding_dim, kernel_size=(1, 1),
                bias=True)

        self.fusion_model1 = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.model_dim *2,
                                   hidden_dim=self.model_dim*2 ,
                                   dropout=0.2)
            ],
        )
        self.fusion_model2 = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.model_dim * 4,
                                   hidden_dim=self.model_dim * 4,
                                   dropout=0.2)
              ],
            nn.Linear(in_features=self.model_dim * 4, out_features=self.model_dim, bias=True)
        )

        self.fusion_graph_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.model_dim+self.node_dim,
                                   hidden_dim=self.model_dim+self.node_dim,
                                   dropout=self.dropout)
              for _ in range(2)],
        )
        self.fusion_forward_linear = nn.Linear(in_features=self.model_dim+self.node_dim, out_features=self.model_dim,
                                                   bias=True)
        self.fusion_backward_linear = nn.Linear(in_features=self.model_dim+self.node_dim, out_features=self.model_dim,
                                                    bias=True)
        self.st1 = DSTTB(self.model_dim, self.model_dim, in_steps, dropout)
        self.fusion = nn.Linear(in_features=self.model_dim*2, out_features=self.model_dim,
                                               bias=True)
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)

        x = history_data
        batch_size, _, num_nodes, _ = x.shape

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        if self.time_embedding_dim > 0:
            tod = x[..., 1]
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        if self.ts_embedding_dim > 0:
            input_data = x.transpose(1, 2).contiguous()
            input_data = input_data.view(
                batch_size, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            # B L*3 N 1
            time_series_emb = self.time_series_emb_layer(input_data)
            time_series_emb = time_series_emb.transpose(1, -1).expand(batch_size, self.in_steps, self.num_nodes,
                                                                      self.ts_embedding_dim)
        # B ts_embedding_dim N 1

        # x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = []

        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.time_embedding_dim > 0:
            time_emb = self.time_embedding(
                ((tod + dow * 7) * self.steps_per_day).long()
            )
            features.append(time_emb)
        # if self.adaptive_embedding_dim > 0:
        #     adp_emb = self.adaptive_embedding.expand(
        #         size=(batch_size, *self.adaptive_embedding.shape)
        #     )
        #     features.append(adp_emb)
        temporal_x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        out,score03,score06 = self.st1(temporal_x)
        out = temporal_x + out


        if self.node_dim > 0:
            node_forward1 = self.adj_mx[0].to(device)
            node_forward2 =  self.adj_mx_forward_encoder(node_forward1.unsqueeze(0)).expand(batch_size, self.in_steps, -1,
                                                                                         -1)
            node_backward1 = self.adj_mx[1].to(device)
            node_backward2 = self.adj_mx_backward_encoder(node_backward1.unsqueeze(0)).expand(batch_size, self.in_steps,
                                                                                           -1,
                                                                                            -1)
            hidden_forward= torch.cat([out, node_forward2], dim=-1)
            hidden_forward = self.fusion_graph_model(hidden_forward)
            hidden_forward = self.fusion_forward_linear(hidden_forward)

            hidden_backward = torch.cat([out, node_backward2], dim=-1)
            hidden_backward = self.fusion_graph_model(hidden_backward)
            hidden_backward = self.fusion_backward_linear(hidden_backward)

        hidden = torch.cat([ hidden_backward,   hidden_forward ], dim=-1)

        x1 = self.fusion_model1( hidden)
        hidden=torch.cat([  hidden, x1], dim=-1)
        x = self.fusion_model2(hidden)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out,score03,score06
