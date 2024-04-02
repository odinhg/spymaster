import torch
import torch.nn as nn
import torch.nn.functional as F

class PermEquivLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm = torch.mean(x, dim=-2, keepdim=True)
        out = self.linear(x - xm)
        return out

class Set2Vec(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            PermEquivLayer(in_dim, hidden_dim),
            nn.GELU(),
            PermEquivLayer(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        out = torch.mean(out, dim=-2)
        return out

class SpyMasterModel(nn.Module):
    
    def __init__(self, emb_dim: int, expand_factor: int = 1):
        super().__init__()
        hidden_dim = emb_dim * expand_factor
        self.set_nn_t = Set2Vec(emb_dim, hidden_dim, emb_dim)

    def forward(
        self, x_t: torch.Tensor,
    ) -> torch.Tensor:
        h_t = self.set_nn_t(x_t)
        return h_t 

class SpyMaster(nn.Module):
    
    def __init__(self, emb_dim: int, hidden_dim: int=50):
        super().__init__()
        self.set_nn_positive = Set2Vec(emb_dim, hidden_dim, hidden_dim)
        self.set_nn_negative = Set2Vec(emb_dim, hidden_dim, hidden_dim)
        self.fcnn = nn.Sequential(nn.Linear(2 * hidden_dim, emb_dim))

    def forward(
            self, x_positive: torch.Tensor, x_negative: torch.Tensor,
    ) -> torch.Tensor:
        h_positive = self.set_nn_positive(x_positive)
        h_negative = self.set_nn_negative(x_negative)
        h_combined = torch.cat([h_positive, h_negative], dim=-1)
        out = self.fcnn(h_combined)
        return out