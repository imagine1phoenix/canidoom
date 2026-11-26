"""Hybrid Neural Architecture (HybridDoomNet) for Zero-Shot DOOM Navigation & Combat.

Implements the specification in Design.md §2:
- Circular 1D CNN over 64 360° rays + hit-type embeddings.
- Masked Multi-Head Attention set encoder over <= 32 visible entities (NaN-safe for 0 entities).
- Vitals and action-context MLP.
- Compact 64x48 RGB+depth CNN with training-time modality dropout.
- Egocentric Occupancy Grid 2D CNN encoder.
- Recurrent core: GRU(256, 256).
- 4 Objective Memory Slots with explicit 2D odometry rotation (-d_theta).
- 9 Bernoulli action heads with observation action masking.
- Critic value head and training-only auxiliary heads.
"""

from __future__ import annotations

import math
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import ModelInputs


class RecurrentState(NamedTuple):
    gru_h: torch.Tensor       # (1, B, 256)
    obj_slots: torch.Tensor   # (B, 4, 8)


class RayCircular1dEncoder(nn.Module):
    """Circular 1D CNN over 64 360° rays.

    Preserves wraparound topology between ray 63 (right) and ray 0 (straight ahead).
    """

    def __init__(self, in_channels: int = 8, out_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode="circular")
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode="circular")
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Linear(64 * 8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 64)
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        pooled = self.pool(h).flatten(start_dim=1)  # (B, 512)
        return F.silu(self.fc(pooled))               # (B, 96)


class EntityAttentionEncoder(nn.Module):
    """Masked Multi-Head Self-Attention set encoder for <= 32 visible entities.

    Permutation-invariant, handles variable enemy/item counts and zero-entity edge cases safely.
    """

    def __init__(self, feat_dim: int = 5, num_types: int = 12, emb_dim: int = 16, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, emb_dim, padding_idx=0)
        self.in_proj = nn.Linear(feat_dim + emb_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, feats: torch.Tensor, types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feats: (B, 32, 5), types: (B, 32), mask: (B, 32)
        B, N, _ = feats.shape

        # Check which batch items have at least one valid entity
        valid_counts = mask.sum(dim=1)  # (B,)
        has_entities = valid_counts > 0

        type_e = self.type_emb(types)                    # (B, 32, 16)
        x = F.silu(self.in_proj(torch.cat([feats, type_e], dim=-1)))  # (B, 32, 64)

        # For MultiheadAttention, key_padding_mask=True means IGNORE slot.
        # If all slots are masked in a batch item, PyTorch MHA outputs NaNs.
        # Safe fix: pass an unmasked dummy slot for empty items, then zero out results.
        safe_key_padding_mask = ~mask
        if not has_entities.all():
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_entities, 0] = False

        attn_out, _ = self.attn(x, x, x, key_padding_mask=safe_key_padding_mask)
        x = self.norm(x + attn_out)

        # Masked pooling
        mask_f = mask.unsqueeze(-1).float()  # (B, 32, 1)
        sum_pooled = (x * mask_f).sum(dim=1)
        mean_pooled = sum_pooled / valid_counts.unsqueeze(-1).clamp(min=1.0)

        # Masked max
        x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pooled = x_masked.max(dim=1).values

        # Zero out batch items with 0 entities
        if not has_entities.all():
            mean_pooled = mean_pooled * has_entities.unsqueeze(-1).float()
            max_pooled = torch.where(has_entities.unsqueeze(-1), max_pooled, torch.zeros_like(max_pooled))

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # (B, 128)
        return F.silu(self.out_proj(pooled))                   # (B, 64)


class VitalsMLP(nn.Module):
    """Encodes health, armor, weapon, ammo, velocities, and context."""

    def __init__(self, in_dim: int = 36, out_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactVisualCNN(nn.Module):
    """Compact 64x48 RGB+depth CNN with modality dropout."""

    def __init__(self, in_channels: int = 4, out_dim: int = 64, dropout_p: float = 0.25):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # -> (24, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # -> (12, 16)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1)           # -> (6, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(48 * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 48, 64)
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        pooled = torch.cat([self.avg_pool(h).flatten(1), self.max_pool(h).flatten(1)], dim=-1)
        out = F.silu(self.fc(pooled))

        # Modality dropout during training: randomly zero visual vector with prob p
        if self.training and self.dropout_p > 0.0:
            mask = (torch.rand(x.shape[0], 1, device=x.device) > self.dropout_p).float()
            out = out * mask

        return out


class GridCNN(nn.Module):
    """Encodes the 32x32 6-channel egocentric occupancy grid."""

    def __init__(self, in_channels: int = 6, out_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1)  # -> (16, 16)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)           # -> (8, 8)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)           # -> (4, 4)
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 32, 32)
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        flat = h.flatten(start_dim=1)
        return F.silu(self.fc(flat))


class ObjectiveMemorySlots(nn.Module):
    """4 Objective Memory Slots (8 dims each) rotated by odometry d_theta."""

    def __init__(self, num_slots: int = 4, slot_dim: int = 8, gru_dim: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.write_gate = nn.Linear(gru_dim, num_slots * slot_dim)
        self.write_val = nn.Linear(gru_dim, num_slots * slot_dim)

    def forward(
        self,
        gru_out: torch.Tensor,
        slots: torch.Tensor | None,
        d_theta: torch.Tensor,
    ) -> torch.Tensor:
        # gru_out: (B, 256), slots: (B, 4, 8), d_theta: (B,)
        B = gru_out.shape[0]
        if slots is None:
            slots = torch.zeros(B, self.num_slots, self.slot_dim, device=gru_out.device, dtype=gru_out.dtype)

        # 1. Rotate directional components (indices 0, 1: sin, cos) by -d_theta
        cos_dt = torch.cos(-d_theta).view(B, 1)  # (B, 1)
        sin_dt = torch.sin(-d_theta).view(B, 1)

        # For each of the 4 slots, rotate 2D vector [sin_phi, cos_phi]
        sin_curr = slots[:, :, 0]  # (B, 4)
        cos_curr = slots[:, :, 1]  # (B, 4)

        sin_rot = sin_curr * cos_dt + cos_curr * sin_dt
        cos_rot = cos_curr * cos_dt - sin_curr * sin_dt

        slots_rot = slots.clone()
        slots_rot[:, :, 0] = sin_rot
        slots_rot[:, :, 1] = cos_rot

        # 2. Gated write from GRU output
        gate = torch.sigmoid(self.write_gate(gru_out)).view(B, self.num_slots, self.slot_dim)
        val = torch.tanh(self.write_val(gru_out)).view(B, self.num_slots, self.slot_dim)

        next_slots = slots_rot * (1.0 - gate) + val * gate
        return next_slots


class HybridDoomNet(nn.Module):
    """Main neural policy & critic network for Track 1 DOOM agent."""

    def __init__(
        self,
        use_rgb: bool = True,
        use_grid: bool = True,
        rgb_dropout_p: float = 0.25,
        gru_dim: int = 256,
        num_slots: int = 4,
        slot_dim: int = 8,
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_grid = use_grid
        self.gru_dim = gru_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Sensory Encoders
        self.ray_enc = RayCircular1dEncoder(in_channels=8, out_dim=96)
        self.ent_enc = EntityAttentionEncoder(feat_dim=5, num_types=12, d_model=64)
        self.vitals_enc = VitalsMLP(in_dim=36, out_dim=48)
        self.visual_enc = CompactVisualCNN(in_channels=4, out_dim=64, dropout_p=rgb_dropout_p)
        self.grid_enc = GridCNN(in_channels=6, out_dim=64)

        # Fusion projection: 96 + 64 + 48 + 64 + 64 = 336
        fusion_in_dim = 96 + 64 + 48 + 64 + 64
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, gru_dim),
            nn.SiLU(),
        )

        # Recurrent Core
        self.gru = nn.GRU(gru_dim, gru_dim, batch_first=True)

        # Objective Memory Slots
        self.slot_memory = ObjectiveMemorySlots(num_slots=num_slots, slot_dim=slot_dim, gru_dim=gru_dim)

        # Heads (GRU 256 + Slots 4x8=32 -> 288)
        core_dim = gru_dim + (num_slots * slot_dim)
        self.actor = nn.Sequential(
            nn.Linear(core_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 9),
        )
        self.critic = nn.Sequential(
            nn.Linear(core_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Training-Only Auxiliary Heads
        self.aux_next_rays = nn.Linear(core_dim, 16)
        self.aux_door_bearing = nn.Linear(core_dim, 3)  # [has_door, sin, cos]
        self.aux_key_bearing = nn.Linear(core_dim, 3)   # [has_key, sin, cos]
        self.aux_value_compass = nn.Linear(core_dim, 8)  # 8-sector Ego-Directional Value Compass (B, 8)

    def forward(
        self,
        inputs: ModelInputs,
        recurrent_state: RecurrentState | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState, dict]:
        # 1. Forward through individual encoders
        e_ray = self.ray_enc(inputs.rays)
        e_ent = self.ent_enc(inputs.ent_feats, inputs.ent_types, inputs.ent_mask)
        e_vit = self.vitals_enc(inputs.vitals)

        if self.use_rgb:
            e_rgb = self.visual_enc(inputs.rgb_depth)
        else:
            e_rgb = torch.zeros(inputs.rays.shape[0], 64, device=inputs.rays.device)

        if self.use_grid:
            e_grid = self.grid_enc(inputs.grid)
        else:
            e_grid = torch.zeros(inputs.rays.shape[0], 64, device=inputs.rays.device)

        # 2. Fuse features
        fused = torch.cat([e_ray, e_ent, e_vit, e_rgb, e_grid], dim=-1)
        proj = self.fusion(fused)  # (B, 256)

        # 3. Recurrent update
        B = proj.shape[0]
        gru_h = recurrent_state.gru_h if recurrent_state is not None else None
        prev_slots = recurrent_state.obj_slots if recurrent_state is not None else None

        gru_out, next_gru_h = self.gru(proj.unsqueeze(1), gru_h)
        gru_out = gru_out.squeeze(1)  # (B, 256)

        # 4. Objective slots update
        next_slots = self.slot_memory(gru_out, prev_slots, inputs.d_theta)
        next_recurrent_state = RecurrentState(gru_h=next_gru_h, obj_slots=next_slots)

        # 5. Policy and Value predictions
        core_repr = torch.cat([gru_out, next_slots.flatten(start_dim=1)], dim=-1)  # (B, 288)

        logits = self.actor(core_repr)  # (B, 9)
        value = self.critic(core_repr)  # (B, 1)

        # Apply observation action mask if provided
        if inputs.action_mask is not None:
            mask = inputs.action_mask
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # Suppress forbidden actions
            logits = torch.where(mask, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))

        # 6. Auxiliary predictions (training only)
        aux_dict = {}
        if return_aux:
            aux_dict["pred_next_rays"] = self.aux_next_rays(core_repr)
            aux_dict["pred_door_bearing"] = self.aux_door_bearing(core_repr)
            aux_dict["pred_key_bearing"] = self.aux_key_bearing(core_repr)
            aux_dict["value_compass"] = self.aux_value_compass(core_repr)

        return logits, value, next_recurrent_state, aux_dict
