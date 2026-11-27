"""HybridDoomNet: multi-modal recurrent actor-critic with 8-sector Ego-Directional Value Compass.

Strictly ego-centric and zero-entity NaN-safe. Matches planning/Schema.md §2.3 and Design.md §2,
while supporting `submission/agent.py` keyword calls cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from submission.preprocessing import ModelInputs
else:
    try:
        from .preprocessing import ModelInputs
    except ImportError:
        from preprocessing import ModelInputs


@dataclass
class RecurrentState:
    """Recurrent hidden state + objective memory slots carried across timesteps."""

    gru_h: torch.Tensor      # (1, B, gru_dim) float32
    obj_slots: torch.Tensor  # (B, num_slots, slot_dim) float32

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        gru_dim: int = 256,
        num_slots: int = 4,
        slot_dim: int = 8,
        device: torch.device | str = "cpu",
    ) -> "RecurrentState":
        return cls(
            gru_h=torch.zeros((1, batch_size, gru_dim), dtype=torch.float32, device=device),
            obj_slots=torch.zeros((batch_size, num_slots, slot_dim), dtype=torch.float32, device=device),
        )

    def detach(self) -> "RecurrentState":
        return RecurrentState(gru_h=self.gru_h.detach(), obj_slots=self.obj_slots.detach())

    def to(self, device: torch.device | str) -> "RecurrentState":
        return RecurrentState(gru_h=self.gru_h.to(device), obj_slots=self.obj_slots.to(device))

    def mask_done(self, done: torch.Tensor) -> "RecurrentState":
        """Zero hidden states for environments where `done` is True / 1.0."""
        keep = 1.0 - done.to(dtype=self.gru_h.dtype, device=self.gru_h.device)
        return RecurrentState(
            gru_h=self.gru_h * keep.view(1, -1, 1),
            obj_slots=self.obj_slots * keep.view(-1, 1, 1),
        )


class RayCircularEncoder(nn.Module):
    """360-degree circular 1D Conv encoder over 64 rays -> 96 dims."""

    def __init__(self, out_dim: int = 96) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.proj = nn.Sequential(
            nn.Linear(64 * 8, out_dim),
            nn.SiLU(),
        )

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        x = self.conv(rays)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class EntitySetEncoder(nn.Module):
    """Permutation-invariant masked attention encoder over <= 32 entities -> 64 dims.

    Guarantees finite activations and gradients even when `ent_mask.sum(dim=-1) == 0`.
    """

    def __init__(self, num_types: int = 12, emb_dim: int = 16, d_ent: int = 64) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(num_types, emb_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(emb_dim + 5, d_ent),
            nn.SiLU(),
        )
        self.norm1 = nn.LayerNorm(d_ent)
        self.attn = nn.MultiheadAttention(d_ent, num_heads=4, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(d_ent * 2, d_ent),
            nn.SiLU(),
        )

    def forward(
        self,
        ent_feats: torch.Tensor,
        ent_types: torch.Tensor,
        ent_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ent_mask: (B, 32) bool
        has_any = ent_mask.any(dim=-1)  # (B,)
        mask_3d = ent_mask.unsqueeze(-1)
        mask_f = mask_3d.to(dtype=ent_feats.dtype)

        safe_feats = torch.nan_to_num(ent_feats, nan=0.0, posinf=0.0, neginf=0.0)
        safe_feats = torch.where(mask_3d, safe_feats, torch.zeros_like(safe_feats))
        emb = self.type_emb(ent_types.clamp(0, self.type_emb.num_embeddings - 1))
        raw_x = torch.cat([emb, safe_feats], dim=-1)
        raw_x = torch.where(mask_3d, raw_x, torch.zeros_like(raw_x))
        x = self.in_proj(raw_x) * mask_f

        # Ensure at least slot 0 is unmasked for empty rows so softmax never divides by 0 -> NaN
        safe_mask = ent_mask.clone()
        if not torch.all(has_any):
            safe_mask[~has_any, 0] = True
        key_padding_mask = ~safe_mask

        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask, need_weights=False)
        x = (x + attn_out) * mask_f

        counts = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = x.sum(dim=1) / counts

        # Masked max pool; empty rows zeroed after max
        masked_for_max = x.masked_fill(~ent_mask.unsqueeze(-1), -1e4)
        max_pool = masked_for_max.max(dim=1).values * has_any.unsqueeze(-1).to(dtype=x.dtype)

        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        out = self.out_proj(pooled)
        return out * has_any.unsqueeze(-1).to(dtype=out.dtype)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class VisualEncoder(nn.Module):
    """Compact depthwise-separable RGB+Depth CNN (4, 48, 64) -> 64 dims."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            DepthwiseSeparableConv2d(4, 16, stride=2),   # 24x32
            DepthwiseSeparableConv2d(16, 32, stride=2),  # 12x16
            DepthwiseSeparableConv2d(32, 48, stride=2),  # 6x8
        )
        self.proj = nn.Sequential(
            nn.Linear(48 * 2, out_dim),
            nn.SiLU(),
        )

    def forward(self, rgb_depth: torch.Tensor) -> torch.Tensor:
        x = self.stem(rgb_depth)
        avg_p = x.mean(dim=(-2, -1))
        max_p = x.amax(dim=(-2, -1))
        return self.proj(torch.cat([avg_p, max_p], dim=-1))


class GridEncoder(nn.Module):
    """Egocentric 32x32 6-channel occupancy grid CNN -> 64 dims."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=3, stride=2, padding=1),   # 16x16
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 4 * 4, out_dim),
            nn.SiLU(),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = self.conv(grid).flatten(1)
        return self.proj(x)


def _as_tensor_dict(
    batch: ModelInputs | dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    raw = batch.as_dict() if isinstance(batch, ModelInputs) else batch
    out: dict[str, torch.Tensor] = {}
    for key, val in raw.items():
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            t = val.to(device=device)
        else:
            t = torch.as_tensor(np.asarray(val), device=device)
        if key in ("ent_types",):
            t = t.to(dtype=torch.long)
        elif key in ("ent_mask", "action_mask"):
            t = t.to(dtype=torch.bool)
        else:
            t = torch.nan_to_num(t.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = t
    return out


class HybridDoomNet(nn.Module):
    """Multi-modal recurrent actor-critic with objective memory slots and 8-sector ego value compass."""

    def __init__(
        self,
        model_config: Optional[dict[str, Any]] = None,
        *,
        use_rgb: Optional[bool] = None,
        use_grid: Optional[bool] = None,
        rgb_dropout_p: Optional[float] = None,
        d_model: Optional[int] = None,
        gru_dim: Optional[int] = None,
        num_slots: Optional[int] = None,
        slot_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        cfg = dict(model_config or {})
        self.d_model = int(d_model if d_model is not None else cfg.get("d_model", 256))
        self.gru_dim = int(gru_dim if gru_dim is not None else cfg.get("gru_dim", 256))
        self.num_slots = int(num_slots if num_slots is not None else cfg.get("num_slots", 4))
        self.slot_dim = int(slot_dim if slot_dim is not None else cfg.get("slot_dim", 8))
        self.use_rgb = bool(use_rgb if use_rgb is not None else cfg.get("use_rgb", True))
        self.use_grid = bool(use_grid if use_grid is not None else cfg.get("use_grid", True))
        self.rgb_dropout_p = float(rgb_dropout_p if rgb_dropout_p is not None else cfg.get("rgb_dropout_p", 0.0))
        self.model_config = {
            "d_model": self.d_model,
            "gru_dim": self.gru_dim,
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "use_rgb": self.use_rgb,
            "use_grid": self.use_grid,
        }

        self.ray_enc = RayCircularEncoder(out_dim=96)
        self.ent_enc = EntitySetEncoder(num_types=12, emb_dim=16, d_ent=64)
        self.vitals_enc = nn.Sequential(
            nn.Linear(36, 64),
            nn.SiLU(),
            nn.Linear(64, 48),
            nn.SiLU(),
        )
        self.rgb_enc = VisualEncoder(out_dim=64) if self.use_rgb else None
        self.grid_enc = GridEncoder(out_dim=64) if self.use_grid else None

        fused_in = 96 + 64 + 48 + 64 + 64  # 336
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, self.d_model),
            nn.SiLU(),
        )

        self.gru = nn.GRU(self.d_model, self.gru_dim, batch_first=True)

        slot_total = self.num_slots * self.slot_dim
        self.slot_write = nn.Linear(self.gru_dim, slot_total)
        self.slot_gate = nn.Linear(self.gru_dim + self.slot_dim, self.slot_dim)

        core_dim = self.gru_dim + slot_total  # 288 by default
        self.policy_head = nn.Sequential(
            nn.Linear(core_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 9),
        )
        self.value_head = nn.Sequential(
            nn.Linear(core_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Training-only auxiliary heads
        self.aux_next_rays = nn.Sequential(
            nn.Linear(core_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        )
        self.aux_value_compass = nn.Sequential(
            nn.Linear(core_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 8),
        )
        self.aux_door_prox = nn.Linear(core_dim, 3)
        self.aux_key_prox = nn.Linear(core_dim, 3)
        self.aux_exit_prox = nn.Linear(core_dim, 3)

    def init_state(self, batch_size: int, device: torch.device | str | None = None) -> RecurrentState:
        dev = device if device is not None else next(self.parameters()).device
        return RecurrentState.zeros(
            batch_size=batch_size,
            gru_dim=self.gru_dim,
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            device=dev,
        )

    def _encode_inputs(
        self,
        d: dict[str, torch.Tensor],
        rgb_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ray_feat = self.ray_enc(d["rays"])
        ent_feat = self.ent_enc(d["ent_feats"], d["ent_types"], d["ent_mask"])
        vit_feat = self.vitals_enc(d["vitals"])

        bsz = ray_feat.shape[0]
        if self.rgb_enc is not None:
            rgb_feat = self.rgb_enc(d["rgb_depth"])
            if rgb_drop_mask is not None:
                rgb_feat = rgb_feat * rgb_drop_mask.view(bsz, 1).to(dtype=rgb_feat.dtype)
        else:
            rgb_feat = torch.zeros((bsz, 64), dtype=ray_feat.dtype, device=ray_feat.device)

        if self.grid_enc is not None:
            grid_feat = self.grid_enc(d["grid"])
        else:
            grid_feat = torch.zeros((bsz, 64), dtype=ray_feat.dtype, device=ray_feat.device)

        fused = torch.cat([ray_feat, ent_feat, vit_feat, rgb_feat, grid_feat], dim=-1)
        return self.fusion(fused)

    def _rotate_slots(self, slots: torch.Tensor, d_theta: Optional[torch.Tensor]) -> torch.Tensor:
        if d_theta is None or self.slot_dim < 2:
            return slots
        safe_dt = torch.nan_to_num(d_theta, nan=0.0, posinf=0.0, neginf=0.0)
        cos_dt = torch.cos(safe_dt).view(-1, 1)
        sin_dt = torch.sin(safe_dt).view(-1, 1)
        s0 = slots[:, :, 0]
        s1 = slots[:, :, 1]
        r0 = s0 * cos_dt + s1 * sin_dt
        r1 = s1 * cos_dt - s0 * sin_dt
        if self.slot_dim == 2:
            return torch.stack([r0, r1], dim=-1)
        return torch.cat([torch.stack([r0, r1], dim=-1), slots[:, :, 2:]], dim=-1)

    def _update_slots(
        self,
        gru_out: torch.Tensor,
        prev_slots: torch.Tensor,
        d_theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = gru_out.shape[0]
        rot_slots = self._rotate_slots(prev_slots, d_theta)
        cand = torch.tanh(self.slot_write(gru_out)).view(bsz, self.num_slots, self.slot_dim)
        gru_exp = gru_out.unsqueeze(1).expand(-1, self.num_slots, -1)
        gate = torch.sigmoid(self.slot_gate(torch.cat([gru_exp, rot_slots], dim=-1)))
        return rot_slots + gate * (cand - rot_slots)

    def _heads(
        self,
        core_feat: torch.Tensor,
        slots: torch.Tensor,
        action_mask: torch.Tensor,
        return_aux: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        logits = self.policy_head(core_feat).float()
        logits = logits.masked_fill(~action_mask, -1e8)
        value = self.value_head(core_feat).squeeze(-1).float()

        if not return_aux:
            return logits, value, None

        aux_preds = {
            "next_rays": torch.sigmoid(self.aux_next_rays(core_feat)),
            "value_compass": self.aux_value_compass(core_feat),
            "obj_slots": slots,
            "door_prox": self.aux_door_prox(core_feat),
            "key_prox": self.aux_key_prox(core_feat),
            "exit_prox": self.aux_exit_prox(core_feat),
        }
        return logits, value, aux_preds

    def forward(
        self,
        batch: Optional[ModelInputs | dict[str, Any]] = None,
        h: Optional[RecurrentState] = None,
        return_aux: bool = False,
        *,
        inputs: Optional[ModelInputs | dict[str, Any]] = None,
        recurrent_state: Optional[RecurrentState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
        """Single-step forward pass for batch size B.

        Canonical (`Schema.md §2.3`): always returns `(logits_9 (B, 9), value (B,), aux_preds | None, next_h)`.
        """
        actual_batch = inputs if inputs is not None else batch
        if actual_batch is None:
            raise ValueError("HybridDoomNet.forward requires `batch` (or `inputs`)")
        actual_h = recurrent_state if recurrent_state is not None else h

        device = next(self.parameters()).device
        d = _as_tensor_dict(actual_batch, device)
        bsz = d["rays"].shape[0]
        if actual_h is None:
            actual_h = self.init_state(bsz, device=device)

        rgb_drop_mask: Optional[torch.Tensor] = None
        if self.training and self.rgb_dropout_p > 0.0:
            rgb_drop_mask = torch.rand((bsz, 1), device=device) >= self.rgb_dropout_p

        emb = self._encode_inputs(d, rgb_drop_mask=rgb_drop_mask)  # (B, d_model)
        gru_seq, next_gru_h = self.gru(emb.unsqueeze(1), actual_h.gru_h)
        gru_out = gru_seq.squeeze(1)
        next_slots = self._update_slots(gru_out, actual_h.obj_slots, d.get("d_theta"))

        core_feat = torch.cat([gru_out, next_slots.flatten(1)], dim=-1)
        logits, value, aux_preds = self._heads(core_feat, next_slots, d["action_mask"], return_aux)
        next_h = RecurrentState(gru_h=next_gru_h, obj_slots=next_slots)
        return logits, value, aux_preds, next_h

    def forward_sequence(
        self,
        batch_seq: ModelInputs | dict[str, Any],
        h: Optional[RecurrentState] = None,
        done_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        rgb_dropout_p: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]], RecurrentState]:
        """Truncated BPTT sequence forward pass over (B, T, ...) with `(1 - done[t-1])` resetting.

        Returns `(logits (B, T, 9), value (B, T), aux_preds | None, final_h)`.
        """
        device = next(self.parameters()).device
        d = _as_tensor_dict(batch_seq, device)
        bsz, seq_len = d["rays"].shape[:2]
        if h is None:
            h = self.init_state(bsz, device=device)

        flat = {k: v.reshape(bsz * seq_len, *v.shape[2:]) for k, v in d.items()}

        drop_p = self.rgb_dropout_p if rgb_dropout_p is None else float(rgb_dropout_p)
        rgb_drop_mask: Optional[torch.Tensor] = None
        if self.training and drop_p > 0.0:
            chunk_keep = (torch.rand((bsz, 1), device=device) >= drop_p).expand(bsz, seq_len)
            rgb_drop_mask = chunk_keep.reshape(bsz * seq_len)

        emb = self._encode_inputs(flat, rgb_drop_mask=rgb_drop_mask).view(bsz, seq_len, self.d_model)
        d_theta_seq = d.get("d_theta")

        gru_h = h.gru_h
        slots = h.obj_slots
        core_steps: list[torch.Tensor] = []
        slot_steps: list[torch.Tensor] = []

        done_f = done_mask.to(device=device, dtype=torch.float32) if done_mask is not None else None

        for t in range(seq_len):
            if t > 0 and done_f is not None:
                keep = 1.0 - done_f[:, t - 1]
                gru_h = gru_h * keep.view(1, bsz, 1)
                slots = slots * keep.view(bsz, 1, 1)

            gru_out_t, gru_h = self.gru(emb[:, t : t + 1, :], gru_h)
            gru_out_t = gru_out_t.squeeze(1)
            dt_t = d_theta_seq[:, t] if d_theta_seq is not None else None
            slots = self._update_slots(gru_out_t, slots, dt_t)
            core_steps.append(torch.cat([gru_out_t, slots.flatten(1)], dim=-1))
            slot_steps.append(slots)

        core_seq = torch.stack(core_steps, dim=1)   # (B, T, core_dim)
        slots_seq = torch.stack(slot_steps, dim=1)  # (B, T, num_slots, slot_dim)

        logits_flat, value_flat, aux_flat = self._heads(
            core_seq.reshape(bsz * seq_len, -1),
            slots_seq.reshape(bsz * seq_len, self.num_slots, self.slot_dim),
            flat["action_mask"],
            return_aux,
        )
        logits = logits_flat.view(bsz, seq_len, 9)
        value = value_flat.view(bsz, seq_len)

        aux_out: Optional[dict[str, torch.Tensor]] = None
        if aux_flat is not None:
            aux_out = {k: v.view(bsz, seq_len, *v.shape[1:]) for k, v in aux_flat.items()}

        return logits, value, aux_out, RecurrentState(gru_h=gru_h, obj_slots=slots)
