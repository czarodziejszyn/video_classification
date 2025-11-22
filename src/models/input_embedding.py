import torch
import torch.nn as nn
import numpy as np


class InputEmbedding(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)

    def forward(self, x):
        return self.embedding(x)


class InputEncoding(nn.Module):
    def __init__(self, num_node=25, d_model=64, T_max=300):
        super().__init__()
        self.num_node = num_node
        self.d_model = d_model
        self.T_max = T_max

        # positional (temporal) encoding
        pe = torch.zeros(T_max, d_model)
        position = torch.arange(0, T_max, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe)  # (T, D)

        # spatial encoding
        self.spatial_embedding = nn.Embedding(num_node, d_model)

    def temporal_encoding(self, T):
        return self.pos_encoding[:T, :]

    def spatial_encoding(self, V):
        device = self.spatial_embedding.weight.device
        node_idx = torch.arange(V, device=device)
        return self.spatial_embedding(node_idx)  # (V, D)
