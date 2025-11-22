import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from src.models.input_embedding import InputEmbedding, InputEncoding


class TemporalConvolution(nn.Module):
    def __init__(self, k, d_model):
        super().__init__()
        self.conv = nn.Conv2d(d_model, d_model, (k, 1), padding=((k - 1) // 2, 0))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.conv(x)
        return out.permute(0, 2, 3, 1)


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, V, D = x.shape
        z = x
        x = self.norm(x)
        q = self.q(x).view(B, T, V, self.h, self.d)
        k = self.k(x).view(B, T, V, self.h, self.d)
        v = self.v(x).view(B, T, V, self.h, self.d)

        q = q.permute(0,2,3,1,4)  # (B,V,H,T,d)
        k = k.permute(0,2,3,1,4)
        v = v.permute(0,2,3,1,4)
        attn = torch.einsum("b v h t d, b v h s d -> b v h t s", q, k) / math.sqrt(self.d)
        attn = self.dropout(self.softmax(attn))
        y = torch.einsum("b v h t s, b v h s d -> b v h t d", attn, v)
        y = y.permute(0,1,3,2,4).contiguous().view(B, V, T, D).permute(0,2,1,3)  # (B,T,V,D)
        out = self.out_proj(y)
        return out + z


class SpatialSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, V, D = x.shape
        z = x
        x = self.norm(x)
        q = self.q(x).view(B, T, V, self.h, self.d)
        k = self.k(x).view(B, T, V, self.h, self.d)
        v = self.v(x).view(B, T, V, self.h, self.d)

        q = q.permute(0,1,3,2,4)  # (B,T,H,V,d)
        k = k.permute(0,1,3,2,4)
        v = v.permute(0,1,3,2,4)
        attn = torch.einsum("b t h v d, b t h u d -> b t h v u", q, k) / math.sqrt(self.d)
        attn = self.dropout(self.softmax(attn))
        y = torch.einsum("b t h v u, b t h u d -> b t h v d", attn, v)
        y = y.permute(0,1,3,2,4).contiguous().view(B, T, V, D)
        out = self.out_proj(y)
        return out + z


class Graph:
    def __init__(self):
        self.num_node = 25
        self.edge = [
            (1, 2), (1, 13), (1, 17),
            (2, 21),
            (3, 4), (3, 21),
            (5, 6), (5, 21),
            (6, 7),
            (7, 8),
            (8, 22), (8, 23),
            (9, 10), (9, 21),
            (10, 11),
            (11, 12),
            (12, 24), (12, 25),
            (13, 14),
            (14, 15),
            (15, 16),
            (17, 18),
            (18, 19),
            (19, 20)
        ]
        self.A = self.normalize_adj_matrix(self.get_adjacency_matrix())

    def get_adjacency_matrix(self):
        A = np.zeros((self.num_node, self.num_node))
        for e in self.edge:
            A[e[0]-1, e[1]-1] = 1
            A[e[1]-1, e[0]-1] = 1
        return A

    def normalize_adj_matrix(self, A):
        A = A + np.eye(self.num_node)
        D = np.sum(A, axis=1)
        D_inv = np.diag(1.0 / np.sqrt(D))
        return (D_inv @ A @ D_inv).astype(np.float32)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        graph = Graph()
        self.register_buffer('A', torch.tensor(graph.A))  # (V,V), follows device
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        B, T, V, D = x.shape
        xw = torch.einsum("btvd, df -> btvf", x, self.weight)   # (B,T,V,F)
        out = torch.einsum("vv, btvf -> btvf", self.A, xw)      # GCN: A * XW
        if self.bias is not None:
            out = out + self.bias
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, k=9):
        super().__init__()
        self.tsa = TemporalSelfAttention(d_model, num_heads, dropout)
        self.ssa = SpatialSelfAttention(d_model, num_heads, dropout)
        self.tcn = TemporalConvolution(k, d_model)
        self.gcn = GraphConvolution(d_model, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        self.norm_s = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_s, x_t):
        # Spatial stream: SSA + TCN
        xs_res = x_s
        x_s = self.ssa(self.norm_s(x_s))
        x_s = self.tcn(x_s)
        x_s = x_s + xs_res

        # Temporal stream: GCN + TSA
        xt_res = x_t
        x_t = self.gcn(self.norm_t(x_t))
        x_t = self.tsa(x_t)
        x_t = x_t + xt_res

        # Fusion + FFN
        x = self.dropout(self.ffn(x_s + x_t))
        x = self.norm_out(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, k=9):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff, dropout, k) for _ in range(num_layers)])

    def forward(self, x_s, x_t):
        for layer in self.layers:
            x = layer(x_s, x_t)
            x_s, x_t = x, x
        return x


class STTransformer(nn.Module):
    def __init__(self, num_classes=120, d_model=64, num_heads=8, d_ff=64, num_layers=6, dropout=0.1, k=9, num_node=25, T_max=300):
        super().__init__()
        self.input_embedding = InputEmbedding(d_model=d_model)
        self.input_encoding = InputEncoding(num_node=num_node, d_model=d_model, T_max=T_max)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout, k)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B,T,V,3)
        B, T, V, D = x.shape
        x = self.input_embedding(x)      # (B,T,V,D)

        pos_enc = self.input_encoding.temporal_encoding(T)  # (T,D)
        spa_enc = self.input_encoding.spatial_encoding(V)   # (V,D)

        x_pos = x + pos_enc.unsqueeze(0).unsqueeze(2)  # (1,T,1,D)
        x_spa = x + spa_enc.unsqueeze(0).unsqueeze(1)  # (1,1,V,D)


        x = self.encoder(x_pos, x_spa)           # fused stream
        x = x.mean(dim=[1, 2])           # global average over (T,V): (B,D)
        logits = self.head(x)            # (B,C)
        return logits
