import torch
import torch.nn as nn


class L2PairwiceObjectiveFunction(nn.Module):
    def __init__(self, n_common_points: int = 1000):
        super().__init__()
        self.n_common_points = n_common_points

    def forward(
            self,
            x: torch.Tensor,
            y1: torch.Tensor,
            y2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x_min = x[:, 0]
        x_max = x[:, -1]
        x_common = torch.linspace(0, 1, self.n_common_points, device=x.device, dtype=x.dtype)
        x_common = x_min.unsqueeze(1) + x_common.unsqueeze(0) * (x_max - x_min).unsqueeze(1)

        y1_common = self._interpolate_batch(x, y1, x_common)
        y2_common = self._interpolate_batch(x, y2, x_common)

        diff = torch.mean(
            (y1_common.unsqueeze(-2) - y2_common.unsqueeze(-3)) ** 2, dim=-1
        )
        diff_baseline_y1 = torch.mean(
            y1_common.unsqueeze(-2) ** 2, dim=-1
        )
        diff_baseline_y2 = torch.mean(
            y2_common.unsqueeze(-2) ** 2, dim=-1
        )
        loss = 2 * diff / (diff_baseline_y1 + diff_baseline_y2)

        return loss

    def _interpolate_batch(
            self,
            x_original: torch.Tensor,
            y_original: torch.Tensor,
            x_new: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_original = x_original.shape
        m_new = x_new.shape[1]

        indices = torch.searchsorted(x_original, x_new)

        indices_lower = torch.clamp(indices - 1, min=0, max=n_original - 2)
        indices_upper = torch.clamp(indices, min=0, max=n_original - 1)

        batch_indices = torch.arange(batch_size, device=x_original.device).view(batch_size, 1)
        batch_indices = batch_indices.expand(batch_size, m_new)

        x_lower = x_original[batch_indices, indices_lower]
        x_upper = x_original[batch_indices, indices_upper]
        y_lower = y_original[batch_indices, indices_lower]
        y_upper = y_original[batch_indices, indices_upper]

        denom = x_upper - x_lower
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        weights = (x_new - x_lower) / denom
        weights = torch.clamp(weights, min=0.0, max=1.0)

        y_interp = y_lower + weights * (y_upper - y_lower)

        x_min = x_original[:, 0].unsqueeze(1)
        x_max = x_original[:, -1].unsqueeze(1)
        mask = (x_new >= x_min) & (x_new <= x_max)
        y_interp = torch.where(mask, y_interp, torch.zeros_like(y_interp))

        return y_interp


class SpectraMatchingObjective(nn.CosineSimilarity):
    def forward(self, y1_feature, y2_featrue):
        cos_objective = super().forward(y1_feature.unsqueeze(-2), y2_featrue.unsqueeze(-3))
        return (1 - cos_objective) / 2


class CosSpectraLoss(nn.Module):
    def __init__(self, n_common_points: int = 2000, eps: float = 1e-3):
        super().__init__()
        self.l2_objective = L2PairwiceObjectiveFunction(n_common_points=2000)
        self.cos_objective = SpectraMatchingObjective(dim=-1)
        self.eps = eps

    def forward(self, g_values, spec, spec_corrapted, spin_syste_feature, spec_feature):
        l2_pair_objective = self.l2_objective(g_values, spec, spec_corrapted)
        cos_objective = self.cos_objective(spin_syste_feature, spec_feature)
        loss = (cos_objective - l2_pair_objective) / (l2_pair_objective + self.eps)
        loss = torch.mean(loss ** 2)
        return loss