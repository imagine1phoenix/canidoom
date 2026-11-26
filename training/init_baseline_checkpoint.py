"""Creates an initial baseline checkpoint matching Schema.md §4 for smoke testing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from training.model import HybridDoomNet

WEIGHTS_DIR = ROOT / "submission" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = WEIGHTS_DIR / "best_agent.pt"


def init_checkpoint():
    model = HybridDoomNet(use_rgb=True, use_grid=True)
    payload = {
        "contract_version": "1.0.0-draft",
        "git_sha": "init_baseline",
        "global_step": 0,
        "update_idx": 0,
        "model_config": {
            "d_model": 256,
            "gru_dim": 256,
            "num_slots": 4,
            "slot_dim": 8,
            "use_rgb": True,
            "use_grid": True,
        },
        "model_state_dict": model.state_dict(),
        "scorer_summary": {
            "total_score": 0.0,
            "mean_act_ms": 10.0,
            "p99_act_ms": 18.0,
        },
    }
    torch.save(payload, CHECKPOINT_PATH)
    print(f"Saved initial baseline checkpoint to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    init_checkpoint()
