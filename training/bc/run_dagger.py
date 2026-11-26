"""End-to-end Privileged Teacher -> BC -> Interactive DAgger Distillation pipeline.

Strictly training-only. Generates diverse random procedural maps across Tiers 1–4
(seeds in [300_000, 899_999], disjoint from held-out [1_000, 9_999]), records
Teacher demonstrations and Student-driven DAgger corrective rollouts, trains
`HybridDoomNet` on GPU, and evaluates each round on the 27-map held-out suite.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from doom_env import make_env
from gymnasium.spaces import Box, MultiBinary
from submission.agent import DoomAgent
from training.bc.teacher_bot import PrivilegedTeacherBot
from training.bc.train_bc import DemoChunkDataset, compute_bc_loss
from training.mapgen.tiers import MapGen
from training.model import HybridDoomNet, RecurrentState
from training.ppo import load_checkpoint, save_checkpoint
from training.scorer.score_checkpoint import score_checkpoint


MODEL_CFG = {
    "d_model": 256,
    "gru_dim": 256,
    "num_slots": 4,
    "slot_dim": 8,
    "use_rgb": True,
    "use_grid": True,
}


def _save_episode_npz(path: Path, frames: list[dict[str, Any]], actions: list[np.ndarray], dones: list[bool]) -> None:
    if not frames:
        return
    np.savez_compressed(
        path,
        rgb=np.stack([f["rgb"] for f in frames], axis=0).astype(np.uint8),
        depth=np.stack([f["depth"] for f in frames], axis=0).astype(np.float16),
        vitals=np.stack([f["vitals"] for f in frames], axis=0).astype(np.float32),
        rays_distance=np.stack([f["rays"]["distance"] for f in frames], axis=0).astype(np.float32),
        rays_hit_type=np.stack([f["rays"]["hit_type"] for f in frames], axis=0).astype(np.int64),
        ent_features=np.stack([f["visible_entities"]["features"] for f in frames], axis=0).astype(np.float32),
        ent_type=np.stack([f["visible_entities"]["entity_type"] for f in frames], axis=0).astype(np.int64),
        ent_mask=np.stack([f["visible_entities"]["mask"] for f in frames], axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        dones=np.asarray(dones, dtype=bool),
    )


def collect_episodes(
    out_dir: Path,
    tier_counts: dict[int, int],
    seed_base: int,
    student_ckpt: Optional[Path] = None,
    teacher_beta: float = 1.0,
    device: str = "cuda",
    require_exit_tiers: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """Collect Teacher (beta=1.0) or DAgger (beta<1.0) episodes across Tiers 1-4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_wads = Path(tempfile.mkdtemp(prefix="doom-dagger-wads-"))

    student: Optional[DoomAgent] = None
    if student_ckpt is not None and student_ckpt.is_file() and teacher_beta < 1.0:
        student = DoomAgent(Box(-np.inf, np.inf, (1,), np.float32), MultiBinary(9), device=device)
        ckpt = load_checkpoint(student_ckpt, map_location=device)
        student.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        student.model.eval()

    rng = np.random.default_rng(seed_base)
    saved = 0
    exits = 0
    total_frames = 0
    attempts = 0
    t0 = time.perf_counter()

    try:
        for tier, target_n in tier_counts.items():
            got = 0
            max_tries = max(target_n * 3, target_n + 10)
            tries = 0
            ep_ticks = 3200 if tier >= 3 else 2100

            while got < target_n and tries < max_tries:
                tries += 1
                attempts += 1
                seed = int(rng.integers(300_000, 899_999))
                w_path, map_name = MapGen.sample(tier=tier, seed=seed, out_dir=str(tmp_wads))
                env = make_env(w_path, map_name, episode_ticks=ep_ticks, render_buffers=True)
                obs, _ = env.reset(seed=seed)
                teacher = PrivilegedTeacherBot(w_path, env.sector_polygons())
                if student is not None:
                    student.reset()

                frames: list[dict[str, Any]] = []
                teacher_actions: list[np.ndarray] = []
                dones: list[bool] = []
                done = False
                exit_reached = False
                keys_got = 0

                while not done:
                    priv = env.privileged_state()
                    t_act = teacher.act(obs, priv)
                    if student is not None and rng.random() >= teacher_beta:
                        exec_act = student.act(obs)
                    else:
                        exec_act = t_act

                    frames.append(obs)
                    teacher_actions.append(t_act)
                    obs, _, term, trunc, info = env.step(exec_act)
                    done = bool(term or trunc)
                    dones.append(done)
                    if info.get("exit_reached", False):
                        exit_reached = True

                priv_end = env.privileged_state()
                keys_got = int(np.sum(priv_end.get("keys", np.zeros(6))))
                kills_got = int(priv_end.get("kills", 0))
                dead = bool(priv_end.get("dead", False))
                env.close()

                # Keep winning demos (or high-value DAgger corrective trajectories)
                keep = False
                if teacher_beta < 1.0:
                    # In DAgger, even failed student rollouts carry crucial corrective Teacher labels!
                    keep = len(frames) >= 15
                else:
                    if exit_reached:
                        keep = True
                    elif tier not in require_exit_tiers and (keys_got > 0 or (kills_got >= 2 and not dead)):
                        keep = True

                if keep:
                    ep_file = out_dir / f"t{tier}_s{seed}_ep{saved:04d}.npz"
                    _save_episode_npz(ep_file, frames, teacher_actions, dones)
                    saved += 1
                    got += 1
                    total_frames += len(frames)
                    if exit_reached:
                        exits += 1
            print(
                f"  [Collect] Tier {tier}: saved {got}/{target_n} (tries={tries}, total_frames={total_frames:,}, "
                f"elapsed={time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
    finally:
        shutil.rmtree(tmp_wads, ignore_errors=True)

    return {
        "saved_episodes": saved,
        "exits": exits,
        "attempts": attempts,
        "total_frames": total_frames,
    }


def train_bc_round(
    demo_dirs: list[Path],
    out_ckpt: Path,
    init_ckpt: Optional[Path] = None,
    epochs: int = 12,
    lr: float = 4e-4,
    batch_seqs: int = 32,
    seq_len: int = 32,
    device_str: str = "cuda",
    seed: int = 42,
) -> dict[str, Any]:
    npz_paths: list[Path] = []
    for d in demo_dirs:
        if d.is_dir():
            npz_paths.extend(sorted(d.rglob("*.npz")))
    if not npz_paths:
        raise ValueError(f"No .npz files found in {demo_dirs}")

    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    print(f"[BC Train] Loading {len(npz_paths)} episodes into seq_len={seq_len} chunks...", flush=True)
    ds = DemoChunkDataset(npz_paths, seq_len=seq_len, use_grid=True)
    print(f"[BC Train] Ready: {len(ds)} chunks ({time.perf_counter()-t0:.1f}s). Training {epochs} epochs on {device}...", flush=True)

    model = HybridDoomNet(MODEL_CFG).to(device)
    if init_ckpt is not None and init_ckpt.is_file():
        ckpt = load_checkpoint(init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[BC Train] Warm-started from {init_ckpt}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.15)

    last_loss = 0.0
    total_steps = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(ds))
        ep_losses: list[float] = []
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
                batch_dict, h=h0, done_mask=dones, return_aux=False, rgb_dropout_p=0.20
            )
            loss = compute_bc_loss(
                logits,
                targets,
                batch_dict["action_mask"],
                pos_weight_use=3.0,
                pos_weight_attack=2.5,
                valid_mask=valid_mask,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            ep_losses.append(float(loss.item()))
            total_steps += bsz * seq_len

        scheduler.step()
        last_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(f"  [BC Epoch {ep:2d}/{epochs}] loss={last_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_ckpt, model=model, model_config=MODEL_CFG, global_step=total_steps, update_idx=epochs)
    return {"checkpoint": str(out_ckpt), "loss": last_loss, "chunks": len(ds), "steps": total_steps}


def main() -> int:
    parser = argparse.ArgumentParser(description="Teacher -> BC -> Interactive DAgger GPU Pipeline")
    parser.add_argument("--out-dir", type=str, default="runs/bc_dagger")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=2, help="Number of DAgger rounds after initial BC (Round 0)")
    parser.add_argument("--suite", type=str, default="training/scorer/eval_suite.json")
    args = parser.parse_args()

    base = Path(args.out_dir)
    base.mkdir(parents=True, exist_ok=True)
    demo_dirs: list[Path] = []
    history: list[dict[str, Any]] = []

    # --- Round 0: Pure Teacher Demos + BC Pretrain ---
    print("\n================ ROUND 0: TEACHER DEMOS + BC PRETRAIN ================", flush=True)
    d0 = base / "demos_r0"
    stats0 = collect_episodes(
        out_dir=d0,
        tier_counts={1: 45, 2: 45, 3: 20, 4: 10},
        seed_base=4001,
        teacher_beta=1.0,
        device=args.device,
    )
    demo_dirs.append(d0)
    ckpt0 = base / "bc_round0.pt"
    tr0 = train_bc_round(demo_dirs, ckpt0, epochs=12, lr=5e-4, device_str=args.device, seed=42)
    eval0 = score_checkpoint(checkpoint_path=ckpt0, eval_suite_path=args.suite, device="cpu", out_path=base / "eval_r0.json")
    history.append({"round": 0, "collect": stats0, "train": tr0, "eval": eval0})
    print(f"[Round 0 Summary] total_score={eval0['total_score']:.2f} exit_rate={eval0['exit_rate']*100:.1f}%", flush=True)

    cur_ckpt = ckpt0
    best_score = float(eval0["total_score"])
    best_ckpt = ckpt0

    # --- Rounds 1..R: Interactive DAgger ---
    for r in range(1, args.rounds + 1):
        print(f"\n================ ROUND {r}: INTERACTIVE DAGGER ================", flush=True)
        dr = base / f"demos_r{r}"
        beta = 0.25 if r == 1 else 0.0
        stats_r = collect_episodes(
            out_dir=dr,
            tier_counts={1: 20, 2: 20, 3: 12, 4: 8},
            seed_base=4001 + r * 1000,
            student_ckpt=cur_ckpt,
            teacher_beta=beta,
            device=args.device,
        )
        demo_dirs.append(dr)
        ckpt_r = base / f"dagger_round{r}.pt"
        tr_r = train_bc_round(
            demo_dirs,
            ckpt_r,
            init_ckpt=cur_ckpt,
            epochs=8,
            lr=2.5e-4,
            device_str=args.device,
            seed=42 + r,
        )
        eval_r = score_checkpoint(
            checkpoint_path=ckpt_r,
            eval_suite_path=args.suite,
            device="cpu",
            out_path=base / f"eval_r{r}.json",
        )
        history.append({"round": r, "collect": stats_r, "train": tr_r, "eval": eval_r})
        print(f"[Round {r} Summary] total_score={eval_r['total_score']:.2f} exit_rate={eval_r['exit_rate']*100:.1f}%", flush=True)
        cur_ckpt = ckpt_r
        if float(eval_r["total_score"]) > best_score:
            best_score = float(eval_r["total_score"])
            best_ckpt = ckpt_r

    shutil.copy2(best_ckpt, base / "best_dagger.pt")
    summary = {"best_checkpoint": str(best_ckpt), "best_score": best_score, "history": history}
    (base / "dagger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== DAGGER COMPLETE: best_score={best_score:.2f} ({best_ckpt}) ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
