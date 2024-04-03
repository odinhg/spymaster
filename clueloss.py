import torch
import torch.nn as nn
import torch.nn.functional as F

class ClueLoss(nn.Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = torch.tensor(margin)

    def forward(self, clues, positives, negatives):
        dist_to_positives = 1 - F.cosine_similarity(
            positives, clues.unsqueeze(1), dim=2
        )
        dist_to_negatives = 1 - F.cosine_similarity(
            negatives, clues.unsqueeze(1), dim=2
        )
        max_dist_to_positives, _ = dist_to_positives.max(-1)
        min_dist_to_negatives, _ = dist_to_negatives.min(-1)
        loss = dist_to_positives.mean() + F.relu(
            max_dist_to_positives.mean() - min_dist_to_negatives.mean() + self.margin
        )
        return loss
