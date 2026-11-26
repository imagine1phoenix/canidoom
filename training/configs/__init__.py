"""Config loaders and Schema.md / PR #2-#3 validators for training & evaluation YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent


def _require_keys(cfg: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"{label} config missing required keys: {sorted(missing)}")


def validate_ppo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate ppo_curriculum.yaml schema, ranges, BPTT divisibility, and PR #2/#3 bounds."""
    required = {
        "seed",
        "num_envs",
        "rollout_steps",
        "bptt_len",
        "ppo_epochs",
        "minibatch_chunks",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "lr",
        "max_grad_norm",
        "vf_coef",
        "ent_coef",
        "rnd_coef",
        "grid_novelty_coef",
        "aux_coef",
        "bc_kl_beta_init",
        "compass_coef",
        "rgb_dropout_p",
        "scaffolding_end_step",
        "tier_weights",
        "worker_max_episodes",
        "worker_timeout_s",
        "render_buffers_ratio",
    }
    _require_keys(cfg, required, "ppo")

    if int(cfg["num_envs"]) < 1:
        raise ValueError(f"num_envs must be >= 1, got {cfg['num_envs']}")
    if int(cfg["bptt_len"]) < 1:
        raise ValueError(f"bptt_len must be >= 1, got {cfg['bptt_len']}")
    if int(cfg["rollout_steps"]) < int(cfg["bptt_len"]):
        raise ValueError("rollout_steps must be >= bptt_len")
    if int(cfg["rollout_steps"]) % int(cfg["bptt_len"]) != 0:
        raise ValueError(
            f"rollout_steps ({cfg['rollout_steps']}) must be divisible by bptt_len ({cfg['bptt_len']})"
        )
    if int(cfg["ppo_epochs"]) < 1 or int(cfg["minibatch_chunks"]) < 1:
        raise ValueError("ppo_epochs and minibatch_chunks must be >= 1")
    if not (0.0 < float(cfg["gamma"]) <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {cfg['gamma']}")
    if not (0.0 < float(cfg["gae_lambda"]) <= 1.0):
        raise ValueError(f"gae_lambda must be in (0, 1], got {cfg['gae_lambda']}")
    if not (0.0 < float(cfg["clip_coef"]) < 1.0):
        raise ValueError(f"clip_coef must be in (0, 1), got {cfg['clip_coef']}")
    if float(cfg["lr"]) <= 0.0 or float(cfg["max_grad_norm"]) <= 0.0:
        raise ValueError("lr and max_grad_norm must be > 0")
    if not (0.0 <= float(cfg["rgb_dropout_p"]) <= 1.0):
        raise ValueError(f"rgb_dropout_p must be in [0, 1], got {cfg['rgb_dropout_p']}")
    if not (0.0 <= float(cfg["render_buffers_ratio"]) <= 1.0):
        raise ValueError(
            f"render_buffers_ratio must be in [0, 1], got {cfg['render_buffers_ratio']}"
        )
    if len(list(cfg["tier_weights"])) != 4:
        raise ValueError("tier_weights must contain exactly 4 entries")

    if "cpu_threads" in cfg and int(cfg["cpu_threads"]) > 1:
        raise ValueError(f"cpu_threads must be <= 1 for eval parity, got {cfg['cpu_threads']}")
    if "max_vram_mb" in cfg and int(cfg["max_vram_mb"]) > 2048:
        raise ValueError(f"max_vram_mb exceeds PR #2/#3 2048 MB limit: {cfg['max_vram_mb']}")
    if "max_ram_mb" in cfg and int(cfg["max_ram_mb"]) > 8192:
        raise ValueError(f"max_ram_mb exceeds PR #2/#3 8192 MB limit: {cfg['max_ram_mb']}")
    if "agent_ms_per_step" in cfg and float(cfg["agent_ms_per_step"]) > 40.0:
        raise ValueError(f"agent_ms_per_step exceeds 40.0 ms cap: {cfg['agent_ms_per_step']}")
    return cfg


def validate_bc_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate bc.yaml schema and model dimensions."""
    required = {
        "seed",
        "demo_dir",
        "output_checkpoint",
        "seq_len",
        "batch_seqs",
        "epochs",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "pos_weight_use",
        "pos_weight_attack",
        "rgb_dropout_p",
        "val_split",
        "device",
        "wandb_project",
        "wandb_enabled",
        "model_config",
    }
    _require_keys(cfg, required, "bc")
    if int(cfg["seq_len"]) < 1 or int(cfg["batch_seqs"]) < 1 or int(cfg["epochs"]) < 1:
        raise ValueError("seq_len, batch_seqs, and epochs must be >= 1")
    if float(cfg["lr"]) <= 0.0 or float(cfg["max_grad_norm"]) <= 0.0:
        raise ValueError("lr and max_grad_norm must be > 0")
    if not (0.0 <= float(cfg["val_split"]) < 1.0):
        raise ValueError("val_split must be in [0, 1)")
    return cfg


def validate_eval_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate eval.yaml schema and strict PR #2/#3 hardware envelope."""
    required = {
        "suite_path",
        "output_dir",
        "device",
        "cpu_threads",
        "max_vram_mb",
        "max_ram_mb",
        "agent_ms_per_step",
        "latency_target_ms",
        "tier_weights",
        "deterministic_seed",
        "run_docker_validate",
        "heldout_seed_min",
        "heldout_seed_max",
    }
    _require_keys(cfg, required, "eval")

    if int(cfg["cpu_threads"]) != 1:
        raise ValueError(f"cpu_threads must be 1 for official PR #2/#3 eval, got {cfg['cpu_threads']}")
    if not (1 <= int(cfg["max_vram_mb"]) <= 2048):
        raise ValueError(f"max_vram_mb must be in [1, 2048], got {cfg['max_vram_mb']}")
    if not (1 <= int(cfg["max_ram_mb"]) <= 8192):
        raise ValueError(f"max_ram_mb must be in [1, 8192], got {cfg['max_ram_mb']}")
    if not (0.0 < float(cfg["agent_ms_per_step"]) <= 40.0):
        raise ValueError(f"agent_ms_per_step must be in (0, 40.0], got {cfg['agent_ms_per_step']}")
    if not (0.0 < float(cfg["latency_target_ms"]) <= float(cfg["agent_ms_per_step"])):
        raise ValueError("latency_target_ms must be <= agent_ms_per_step")
    if int(cfg["heldout_seed_max"]) >= 100_000:
        raise ValueError("heldout_seed_max must stay < 100_000 to remain disjoint from training PLR")
    return cfg


def validate_config(cfg: dict[str, Any], kind: Optional[str] = None) -> dict[str, Any]:
    """Validate a config dict by `kind` ('ppo', 'bc', 'eval') or auto-detect from keys."""
    if kind is None:
        if "rollout_steps" in cfg:
            kind = "ppo"
        elif "suite_path" in cfg:
            kind = "eval"
        elif "demo_dir" in cfg:
            kind = "bc"
        else:
            raise ValueError("Cannot infer config kind; pass kind='ppo'|'bc'|'eval'")
    k = kind.lower()
    if k == "ppo":
        return validate_ppo_config(cfg)
    if k == "bc":
        return validate_bc_config(cfg)
    if k == "eval":
        return validate_eval_config(cfg)
    raise ValueError(f"Unknown config kind: {kind}")


def load_config(path: str | Path, kind: Optional[str] = None) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {p}, got {type(data).__name__}")
    return validate_config(data, kind=kind)


def load_ppo_config(path: str | Path = CONFIGS_DIR / "ppo_curriculum.yaml") -> dict[str, Any]:
    return load_config(path, kind="ppo")


def load_bc_config(path: str | Path = CONFIGS_DIR / "bc.yaml") -> dict[str, Any]:
    return load_config(path, kind="bc")


def load_eval_config(path: str | Path = CONFIGS_DIR / "eval.yaml") -> dict[str, Any]:
    return load_config(path, kind="eval")
