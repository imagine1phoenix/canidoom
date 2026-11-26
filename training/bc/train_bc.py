"""Optional / synthetic-compatible Behavior Cloning pretrainer (`Schema.md §3` & `§4`).

Strictly optional: primary training runs 100% self-evolving RL from scratch (`ppo.py`),
while `train_bc.py` supports warm-starting from any `.npz` trajectories if supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from training.model import HybridDoomNet, RecurrentState
from training.ppo import save_checkpoint
from training.preprocessing import ModelInputs, preprocess, update_action_state


class DemoChunkDataset(Dataset):
    """Loads `Schema.md §3` `.npz` trajectories into fixed `seq_len` preprocessed chunks."""

    def __init__(
        self,
        npz_paths: Sequence[str | Path],
        seq_len: int = 32,
        use_grid: bool = True,
    ) -> None:
        self.seq_len = max(1, int(seq_len))
        self.use_grid = bool(use_grid)
        self.chunks: list[dict[str, Any]] = []

        for raw_p in npz_paths:
            p = Path(raw_p)
            if not p.is_file():
                raise FileNotFoundError(f"Demo file not found: {p}")
            self._load_episode(p)

    def _load_episode(self, path: Path) -> None:
        required_keys = (
            "rgb",
            "depth",
            "vitals",
            "rays_distance",
            "rays_hit_type",
            "ent_features",
            "ent_type",
            "ent_mask",
            "actions",
        )
        with np.load(path, allow_pickle=False) as data:
            for k in required_keys:
                if k not in data:
                    raise KeyError(f"Malformed demo archive {path}: missing key '{k}'")
            rgb = data["rgb"]
            depth = data["depth"]
            vitals = data["vitals"]
            r_dist = data["rays_distance"]
            r_type = data["rays_hit_type"]
            e_feats = data["ent_features"]
            e_type = data["ent_type"]
            e_mask = data["ent_mask"]
            actions = np.asarray(data["actions"], dtype=np.float32)
            dones = (
                np.asarray(data["dones"], dtype=bool)
                if "dones" in data
                else np.zeros((len(actions),), dtype=bool)
            )

        num_steps = len(actions)
        if num_steps == 0:
            return

        ep_inputs: list[ModelInputs] = []
        state = None
        for t in range(num_steps):
            obs_t = {
                "rgb": rgb[t],
                "depth": depth[t],
                "vitals": vitals[t],
                "rays": {"distance": r_dist[t], "hit_type": r_type[t]},
                "visible_entities": {
                    "features": e_feats[t],
                    "entity_type": e_type[t],
                    "mask": e_mask[t],
                },
            }
            inp_t, next_state = preprocess(obs_t, state, device="cpu", use_grid=self.use_grid)
            ep_inputs.append(inp_t)
            if bool(dones[t]):
                state = None
            else:
                state = update_action_state(next_state, actions[t], obs_t)

        # Slice into seq_len chunks, padding short episodes with valid_mask=0
        for start in range(0, num_steps, self.seq_len):
            end = min(num_steps, start + self.seq_len)
            valid_len = end - start
            pad_len = self.seq_len - valid_len

            seq_in = ep_inputs[start:end]
            act_slice = actions[start:end]
            done_slice = dones[start:end].astype(np.float32)
            valid_slice = np.ones((valid_len,), dtype=np.float32)

            if pad_len > 0:
                last_inp = seq_in[-1]
                seq_in = list(seq_in) + [last_inp] * pad_len
                act_slice = np.concatenate([act_slice, np.zeros((pad_len, 9), dtype=np.float32)], axis=0)
                done_slice = np.concatenate([done_slice, np.ones((pad_len,), dtype=np.float32)], axis=0)
                valid_slice = np.concatenate([valid_slice, np.zeros((pad_len,), dtype=np.float32)], axis=0)

            chunk_dict = {
                "rays": torch.cat([x.rays for x in seq_in], dim=0),
                "ent_feats": torch.cat([x.ent_feats for x in seq_in], dim=0),
                "ent_types": torch.cat([x.ent_types for x in seq_in], dim=0),
                "ent_mask": torch.cat([x.ent_mask for x in seq_in], dim=0),
                "vitals": torch.cat([x.vitals for x in seq_in], dim=0),
                "rgb_depth": torch.cat([x.rgb_depth for x in seq_in], dim=0),
                "grid": torch.cat([x.grid for x in seq_in], dim=0),
                "action_mask": torch.cat([x.action_mask for x in seq_in], dim=0),
                "actions": torch.from_numpy(act_slice),
                "dones": torch.from_numpy(done_slice),
                "valid_mask": torch.from_numpy(valid_slice),
            }
            self.chunks.append(chunk_dict)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.chunks[idx]


BCDataset = DemoChunkDataset


def compute_bc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    pos_weight_use: float = 2.5,
    pos_weight_attack: float = 2.5,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute masked MultiBinary(9) BCEWithLogitsLoss with positive-class boost on USE & ATTACK."""
    mask_b = action_mask.bool()
    mask_f = mask_b.to(dtype=torch.float32)
    if valid_mask is not None:
        mask_f = mask_f * valid_mask.unsqueeze(-1).to(dtype=torch.float32)

    pw = torch.ones((9,), dtype=torch.float32, device=logits.device)
    pw[6] = float(pos_weight_attack)
    pw[7] = float(pos_weight_use)

    safe_logits = torch.where(mask_b, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    safe_targets = targets.float().clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(safe_logits, safe_targets, pos_weight=pw, reduction="none")
    denom = mask_f.sum().clamp(min=1.0)
    return (bce * mask_f).sum() / denom


def train(cfg_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Train Behavior Cloning warm-start policy on `.npz` demos and save Schema.md §4 checkpoint."""
    if isinstance(cfg_or_path, dict):
        cfg = dict(cfg_or_path)
    else:
        p = Path(cfg_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"BC config not found: {p}")
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))

    demo_dir = Path(cfg["demo_dir"])
    if not demo_dir.is_dir():
        raise FileNotFoundError(f"demo_dir does not exist: {demo_dir}")
    npz_paths = sorted(demo_dir.rglob("*.npz"))
    if not npz_paths:
        raise ValueError(f"No .npz demo files found inside {demo_dir}")

    torch.manual_seed(int(cfg.get("seed", 42)))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    dev_str = str(cfg.get("device", "auto"))
    device = torch.device("cuda" if (dev_str == "auto" and torch.cuda.is_available()) else ("cpu" if dev_str == "auto" else dev_str))

    model_cfg = dict(
        cfg.get(
            "model_config",
            {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True},
        )
    )
    seq_len = int(cfg.get("seq_len", 32))
    ds = DemoChunkDataset(npz_paths, seq_len=seq_len, use_grid=bool(model_cfg.get("use_grid", True)))
    if len(ds) == 0:
        raise ValueError(f"Demo dataset at {demo_dir} produced 0 valid chunks")

    model = HybridDoomNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(cfg.get("epochs", 10))
    batch_seqs = max(1, min(len(ds), int(cfg.get("batch_seqs", 16))))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    pos_use = float(cfg.get("pos_weight_use", 2.5))
    pos_att = float(cfg.get("pos_weight_attack", 2.5))
    rgb_dropout_p = float(cfg.get("rgb_dropout_p", 0.15))

    last_loss = 0.0
    total_steps = 0
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(ds))
        for start in range(0, len(ds), batch_seqs):
            idx_list = perm[start : start + batch_seqs]
            items = [ds[int(i)] for i in idx_list]
            bsz = len(items)
            batch_dict = {
                k: torch.stack([it[k] for it in items], dim=0).to(device)
                for k in ("rays", "ent_feats", "ent_types", "ent_mask", "vitals", "rgb_depth", "grid", "action_mask")
            }
            targets = torch.stack([it["actions"] for it in items], dim=0).to(device)
            dones = torch.stack([it["dones"] for it in items], dim=0).to(device)
            valid_mask = torch.stack([it["valid_mask"] for it in items], dim=0).to(device)

            h0 = RecurrentState.zeros(
                batch_size=bsz,
                gru_dim=model.gru_dim,
                num_slots=model.num_slots,
                slot_dim=model.slot_dim,
                device=device,
            )
            logits, _, _, _ = model.forward_sequence(
                batch_dict, h=h0, done_mask=dones, return_aux=False, rgb_dropout_p=rgb_dropout_p
            )
            loss = compute_bc_loss(
                logits,
                targets,
                batch_dict["action_mask"],
                pos_weight_use=pos_use,
                pos_weight_attack=pos_att,
                valid_mask=valid_mask,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            last_loss = float(loss.item())
            total_steps += bsz * seq_len

    out_ckpt = Path(cfg.get("output_checkpoint", "runs/bc/bc_warmstart.pt"))
    save_checkpoint(
        out_ckpt,
        model=model,
        model_config=model_cfg,
        global_step=total_steps,
        update_idx=epochs,
    )
    return {"checkpoint": str(out_ckpt), "last_loss": last_loss, "chunks": len(ds)}


train_bc = train


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Optional Behavior Cloning warm-start trainer.")
    parser.add_argument("--config", type=str, default="training/configs/bc.yaml")
    args = parser.parse_args(argv)
    print(train(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
