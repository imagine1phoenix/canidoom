"""Random Network Distillation (RND) + Egocentric Grid-Novelty intrinsic curiosity.

Includes:
- `RunningMeanStd`: batch Welford online normalizer safe for B=1 (`ddof=0`) and constant inputs.
- `compute_grid_novelty`: rewards expanding spatial coverage in the 32x32 egocentric grid.
- `RNDModule` (alias `RND`): frozen random target MLP + trainable predictor MLP.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    """Welford online mean & variance estimator safe for B=1 and zero-variance batches."""

    def __init__(self, shape: Sequence[int] | tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(tuple(shape), dtype=np.float64)
        self.var = np.ones(tuple(shape), dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray | torch.Tensor) -> None:
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return
        if self.mean.shape == ():
            flat = arr.reshape(-1)
            batch_mean = np.mean(flat)
            batch_var = np.var(flat, ddof=0)
            batch_count = float(flat.size)
        else:
            if arr.ndim == len(self.mean.shape):
                arr = arr[np.newaxis, ...]
            batch_mean = np.mean(arr, axis=0)
            batch_var = np.var(arr, axis=0, ddof=0)
            batch_count = float(arr.shape[0])

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray | float,
        batch_var: np.ndarray | float,
        batch_count: float,
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = np.maximum(m2 / tot_count, 1e-8)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        mean_t = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std_t = torch.sqrt(torch.as_tensor(self.var, dtype=x.dtype, device=x.device) + 1e-8)
        return ((x - mean_t) / std_t).clamp(-clip, clip)


def compute_grid_novelty(grid: torch.Tensor) -> torch.Tensor:
    """Compute per-sample egocentric grid exploration novelty bonus `(B,)` from `(B, 6, 32, 32)`.

    Rewards newly explored/visited cells (`ch 0` active spread + fresh frontier hits) while remaining
    strictly finite for all-zero and saturated-one inputs.
    """
    if grid.ndim != 4 or grid.shape[1] < 1:
        raise ValueError(f"Expected grid shape (B, 6, 32, 32), got {tuple(grid.shape)}")
    g = torch.nan_to_num(grid.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    visited = g[:, 0]  # (B, 32, 32)
    # Count newly visited cells (0.05..0.995) + total spatial spread excluding center tile
    spread = torch.sqrt(visited.mean(dim=(-2, -1)) + 1e-6) - 1e-3
    fresh_frontier = ((visited > 0.05) & (visited < 0.999)).float().mean(dim=(-2, -1))
    # Encourage discovering walls/doors/items in surrounding grid
    geom_spread = g[:, 1:5].mean(dim=(-3, -2, -1))
    bonus = (spread + 0.5 * fresh_frontier + 0.25 * geom_spread).clamp(min=0.0)
    return bonus


class RNDModule(nn.Module):
    """Random Network Distillation curiosity module with frozen target network."""

    def __init__(
        self,
        feature_dim: int = 100,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.out_dim = int(out_dim)

        self.target = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

        # Freeze target network permanently
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        self.obs_rms = RunningMeanStd(shape=(self.feature_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    def grid_novelty(self, grid: torch.Tensor) -> torch.Tensor:
        return compute_grid_novelty(grid)

    def compute_intrinsic_reward(
        self,
        features: torch.Tensor,
        update_stats: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(normalized_intrinsic_reward (B,), differentiable_predictor_loss ())`."""
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected features shape (B, {self.feature_dim}), got {tuple(features.shape)}"
            )

        clean = torch.nan_to_num(features.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if update_stats:
            self.obs_rms.update(clean)

        norm_x = self.obs_rms.normalize(clean, clip=5.0)
        with torch.no_grad():
            tgt_feat = self.target(norm_x)
        pred_feat = self.predictor(norm_x)

        per_sample_err = 0.5 * (pred_feat - tgt_feat).pow(2).mean(dim=-1)
        loss = per_sample_err.mean()

        raw_rew = per_sample_err.detach()
        if update_stats and raw_rew.numel() > 0:
            self.rew_rms.update(raw_rew)
        rew_std = float(np.sqrt(self.rew_rms.var + 1e-8))
        norm_rew = (raw_rew / max(1e-4, rew_std)).clamp(min=0.0, max=10.0)
        return norm_rew, loss

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compute_intrinsic_reward(features)


RND = RNDModule
