import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch
from .mlp import MultiLayerPerceptron, GraphMLP
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FastAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, qkv_bias=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.fusion_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.model_dim * 2,
                                   hidden_dim=self.model_dim * 2,
                                   dropout=0.2)
              ],
        )
        self.out_proj = nn.Linear( model_dim * 2, model_dim)
        # self.proj_in = nn.Conv2d(model_dim, model_dim, (1, kernel), 1, 0) if kernel > 1 else nn.Identity()
        self.fast = 1

    def forward(self, x, edge_index=None, dim=0):
        # x = self.proj_in(x.transpose(1, 3)).transpose(1, 3)
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=dim, end_dim=dim + 1
        )
        if self.fast:
            out_s = self.fast_attention(x, qs, ks, vs, dim=dim)
        else:
            out_s = self.normal_attention(x, qs, ks, vs, dim=dim)
        if x.size(1) > 1:
            qs = torch.stack(
                torch.split(query.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            ks = torch.stack(
                torch.split(key.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            vs = torch.stack(
                torch.split(value.transpose(1, 2), self.head_dim, dim=-1), dim=-2
            ).flatten(start_dim=dim, end_dim=dim + 1)
            if self.fast:
                out_t = self.fast_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            else:
                out_t = self.normal_attention(
                    x.transpose(1, 2), qs, ks, vs, dim=dim
                ).transpose(1, 2)
            out = torch.concat([out_s, out_t], -1)
            out = self.fusion_model(out)
            out = self.out_proj(out)
        else:
            out = self.out_proj(out_s)

        return out

    def fast_attention(self, x, qs, ks, vs, dim=0):
        qs = nn.functional.normalize(qs, dim=-1)
        ks = nn.functional.normalize(ks, dim=-1)
        N = qs.shape[1]
        b, l = x.shape[dim : dim + 2]

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        out = attention_num / attention_normalizer  # [N, H, D]
        out = torch.unflatten(out, dim, (b, l)).flatten(start_dim=3)
        return out

    def normal_attention(self, x, qs, ks, vs, dim=0):
        b, l = x.shape[dim : dim + 2]
        qs, ks, vs = qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2)
        x = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = torch.unflatten(x, dim, (b, l)).flatten(start_dim=3)
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class FastTemporalAttentionLayer(nn.Module):
#     """轻量时间快速注意力，仅在时间维 T 上做注意力"""
#     def __init__(self, model_dim, num_heads=8, qkv_bias=False):
#         super().__init__()
#         assert model_dim % num_heads == 0
#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.head_dim = model_dim // num_heads
#         self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
#         # self.out_proj = nn.Linear(model_dim, model_dim)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(model_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, model_dim),
#         )
#
#     def forward(self, x):
#         # x: [B, T, N, C]
#         B, T, N, C = x.shape
#
#         # 将时间维放到中间，方便对时间做注意力
#         x_t = x.permute(0, 2, 1, 3)  # [B, N, T, C]
#
#         qkv = self.qkv(x_t)
#         q, k, v = qkv.chunk(3, dim=-1)
#
#         # 分头
#         q = q.view(B, N, T, self.num_heads, self.head_dim)
#         k = k.view(B, N, T, self.num_heads, self.head_dim)
#         v = v.view(B, N, T, self.num_heads, self.head_dim)
#
#         out = self.fast_attention(q, k, v)
#
#         out = out.reshape(B, N, T, C)
#         out = self.feed_forward(out)
#         return out.permute(0, 2, 1, 3).contiguous()  # [B, T, N, C]
#
#     def fast_attention(self, q, k, v, eps=1e-6):
#         """
#         基于点积的快速时间注意力
#         """
#         B, N, T, H, D = q.shape
#
#         # L2 normalize
#         q = F.normalize(q, dim=-1)
#         k = F.normalize(k, dim=-1)
#
#         # kv: [B, N, H, D, D]
#         kv = torch.einsum("bnthd,bnthe->bnhde", k, v)
#         num = torch.einsum("bnthd,bnhde->bnth e", q, kv)
#
#         k_sum = k.sum(dim=2)  # [B, N, H, D]
#         denom = torch.einsum("bnthd,bnhd->bnth", q, k_sum).unsqueeze(-1)
#
#         return num / (denom + eps)




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
                # input_embedding_dim
                 tod_embedding_dim
                + dow_embedding_dim
                # + adaptive_embedding_dim
                + ts_embedding_dim
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

        # if adaptive_embedding_dim > 0:
        #     self.adaptive_embedding = nn.init.xavier_uniform_(
        #         nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
        #     )
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
        # if adaptive_embedding_dim > 0:
        #     self.node_emb = nn.Parameter(
        #         torch.empty(self.num_nodes, self.adaptive_embedding_dim)
        #     )
        #     nn.init.xavier_uniform_(self.node_emb)

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
        self.fastattention = FastAttentionLayer(
            model_dim=self.model_dim,
            num_heads=self.num_heads,

        )

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
        #     spatial_emb = self.node_emb.expand(
        #         batch_size, self.in_steps, *self.node_emb.shape
        #     )
        #     features.append(spatial_emb)
        # if self.adaptive_embedding_dim > 0:
        #     adp_emb = self.adaptive_embedding.expand(
        #         size=(batch_size, *self.adaptive_embedding.shape)
        #     )
        #     features.append(adp_emb)
        temporal_x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        out = self.fastattention(temporal_x)
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

        return out,self.tod_embedding,self.dow_embedding
