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
        self.bn = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = x

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        out = x.permute(0, 2, 3, 1)

        return out + x_res


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

    def forward(self, x):
        B, T, V, D = x.shape
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
        return out


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

    def forward(self, x):
        B, T, V, D = x.shape
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
        return out


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
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.bn = nn.BatchNorm2d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        B, T, V, D = x.shape
        xw = torch.einsum("btvd, df -> btvf", x, self.weight)   # (B,T,V,F)
        out = torch.einsum("vw, btvf -> btvf", self.A, xw)      # GCN: A * XW
        if self.bias is not None:
            out = out + self.bias

        out = out.permute(0, 3, 1, 2)
        out = self.bn(out)
        out = F.relu(out)
        out = out.permute(0, 2, 3, 1)
        return out


class FeatureExtraction(nn.Module):
    def __init__(self, d_model, k):
        super().__init__()
        self.gcn = GraphConvolution(d_model, d_model)
        self.tcn = TemporalConvolution(k, d_model)

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, k=9):
        super().__init__()
        self.tsa = TemporalSelfAttention(d_model, num_heads, dropout)
        self.ssa = SpatialSelfAttention(d_model, num_heads, dropout)

        self.norm1_s = nn.LayerNorm(d_model)
        self.norm1_t = nn.LayerNorm(d_model)

        self.tcn = TemporalConvolution(k, d_model)
        self.gcn = GraphConvolution(d_model, d_model)


        self.norm2_s = nn.LayerNorm(d_model)
        self.norm2_t = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Spatial stream: SSA + TCN
        xs = x
        y = self.norm1_s(xs)
        y = self.ssa(y)
        xs = xs + y

        y = self.norm2_s(xs)
        y = self.tcn(y)
        xs = xs + y

        # Temporal stream: GCN + TSA
        xt = x
        y = self.norm1_t(xt)
        y = self.gcn(y)
        y = self.tsa(y)
        xt = xt + y

        # Fusion + FFN
        fused = xs + xt
        z = self.ffn(fused)
        z = self.dropout(z)
        out = fused + z
        out = self.norm_out(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, k=9):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff, dropout, k) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class STTransformer(nn.Module):
    def __init__(self, num_classes=120, d_model=64, num_heads=8, d_ff=256, num_layers=6, dropout=0.1, k=9, num_node=25, T_max=100):
        super().__init__()
        self.input_embedding = InputEmbedding(d_model=d_model)
        self.input_encoding = InputEncoding(num_node=num_node, d_model=d_model, T_max=T_max)
        self.extractor = FeatureExtraction(d_model, k=k)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout, k)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B,T,V,3)
        B, T, V, D_in = x.shape
        x = self.input_embedding(x)      # (B,T,V,d_model)

        pos_enc = self.input_encoding.temporal_encoding(T).to(x.device)  # (T,D)
        spa_enc = self.input_encoding.spatial_encoding(V).to(x.device)   # (V,D)

        x = x + pos_enc.unsqueeze(0).unsqueeze(2)  # (1,T,1,D)
        x = x + spa_enc.unsqueeze(0).unsqueeze(1)  # (1,1,V,D)

        x = self.extractor(x) #(B,T,V,D)


        x = self.encoder(x)           # fused stream

        x = x.mean(dim=[1, 2])           # global average over (T,V): (B,D)
        logits = self.head(x)            # (B,C)
        return logits
