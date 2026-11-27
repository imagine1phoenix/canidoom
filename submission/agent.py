"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
Implements:
- Single-thread CPU budget enforcement (torch.set_num_threads(1)).
- Hybrid sensory encoding + scan-matched occupancy grid + GRU & Objective Memory Slots.
- Observation-derived legal action masking.
- Low-temperature seeded Bernoulli sampling to break symmetric corner limit-cycles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path so training modules can be imported cleanly
SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from submission.preprocessing import preprocess, OdomGridState, update_action_state, ModelInputs
    from submission.model import HybridDoomNet, RecurrentState
except ImportError:
    from .preprocessing import preprocess, OdomGridState, update_action_state, ModelInputs
    from .model import HybridDoomNet, RecurrentState

WEIGHTS_DIR = SUBMISSION_DIR / "weights"
DEFAULT_CHECKPOINT_PATH = WEIGHTS_DIR / "best_agent.pt"
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def load_strict_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> dict:
    """Strictly load a model checkpoint, failing loudly on missing, LFS pointer, corrupt, or mismatched weights."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with path.open("rb") as f:
        head = f.read(64)
    if head.startswith(_LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"Checkpoint '{path}' is a Git LFS pointer, not a real weights file. "
            "Run 'git lfs pull' to fetch the binary checkpoint."
        )

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Corrupt or unreadable checkpoint '{path}': {exc}") from exc

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(
            f"Invalid checkpoint format in '{path}': expected dict or state_dict, got {type(ckpt).__name__}"
        )

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Empty or invalid state_dict in checkpoint '{path}'")

    # Strict load: raises RuntimeError on any missing, unexpected, or shape-mismatched keys
    model.load_state_dict(state_dict, strict=True)
    return ckpt if isinstance(ckpt, dict) else {"model_state_dict": state_dict}


class _NoStateType:
    pass

_NO_STATE = _NoStateType()


class DoomAgent:
    """Zero-Shot Autonomous DOOM Agent."""

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        device: str = "cpu",
        tau: float = 0.85,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
        allow_untrained: bool = False,
    ):
        """Called once. Strictly loads trained weights onto `device` ("cuda" or "cpu")."""
        self.tau = max(1e-3, float(tau))
        # Strict latency guard: enforce 1 CPU thread to avoid competing with ViZDoom engine
        if device == "cpu":
            torch.set_num_threads(1)

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        # Initialize neural model
        self.model = HybridDoomNet(
            use_rgb=True,
            use_grid=True,
            rgb_dropout_p=0.0,  # No dropout at inference
            gru_dim=256,
            num_slots=4,
            slot_dim=8,
        ).to(device)

        if allow_untrained and checkpoint_path is None:
            pass
        else:
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "checkpoint_path cannot be None unless allow_untrained=True is explicitly requested."
                )
            load_strict_checkpoint(checkpoint_path, self.model, device=device)

        self.model.eval()

        # Warm up JIT and CUDA/CPU GEMM kernels to eliminate first-step latency spike
        try:
            with torch.inference_mode():
                dummy_in = ModelInputs(
                    rays=torch.zeros((1, 8, 64), dtype=torch.float32, device=device),
                    ent_feats=torch.zeros((1, 32, 5), dtype=torch.float32, device=device),
                    ent_types=torch.zeros((1, 32), dtype=torch.int64, device=device),
                    ent_mask=torch.zeros((1, 32), dtype=torch.bool, device=device),
                    vitals=torch.zeros((1, 36), dtype=torch.float32, device=device),
                    rgb_depth=torch.zeros((1, 4, 48, 64), dtype=torch.float32, device=device),
                    grid=torch.zeros((1, 6, 32, 32), dtype=torch.float32, device=device),
                    action_mask=torch.ones((1, 9), dtype=torch.bool, device=device),
                    d_theta=torch.zeros((1,), dtype=torch.float32, device=device),
                )
                self.model(dummy_in)
        except Exception:
            pass

        # Seeded deterministic RNG reset on every map
        self.rng = np.random.default_rng(0)
        self.odom_state: OdomGridState | None = None
        self.recurrent_state: RecurrentState | None = None
        self.last_probs: np.ndarray | None = None

    def reset(self) -> Any:
        """Called before every map. Clear recurrent state, memory buffers and trackers."""
        self.odom_state = OdomGridState.zeros()
        self.recurrent_state = None
        self.rng = np.random.default_rng(0)
        return self.recurrent_state

    @torch.inference_mode()
    def act(self, observation: dict, state: Any = _NO_STATE) -> np.ndarray | tuple[np.ndarray, Any]:
        """Return a MultiBinary action: int array of shape action_space.shape with values in {0, 1}.

        Supports both standard DoomAgent.act(obs) and state-passing Agent.act(obs, h).
        """
        if state is not _NO_STATE and state is not None:
            self.recurrent_state = state

        # 1. Preprocess sensory observation & update egocentric occupancy grid
        if "grid_flat" in observation:
            # Simplified / flat observation bundle
            rays_np = np.asarray(observation["rays"], dtype=np.float32)
            if rays_np.ndim == 2:
                rays_t = torch.from_numpy(rays_np).unsqueeze(0).to(self.device)
            else:
                rays_t = torch.from_numpy(rays_np).to(self.device)

            vitals_np = np.asarray(observation.get("vitals", []), dtype=np.float32)
            vitals_buf = np.zeros(36, dtype=np.float32)
            vitals_buf[: min(len(vitals_np), 36)] = vitals_np[:36]
            vitals_t = torch.from_numpy(vitals_buf).unsqueeze(0).to(self.device)

            grid_flat = np.asarray(observation["grid_flat"], dtype=np.float32)
            grid_buf = np.zeros((6, 32, 32), dtype=np.float32)
            if len(grid_flat) == 1024:
                grid_buf[0] = grid_flat.reshape(32, 32)
            grid_t = torch.from_numpy(grid_buf).unsqueeze(0).to(self.device)

            mask_np = np.asarray(observation.get("action_mask", np.ones(9)), dtype=np.float32)
            mask_t = torch.from_numpy(mask_np > 0.5).unsqueeze(0).to(self.device)

            d_theta_val = float(observation.get("d_theta", 0.0))
            d_theta_t = torch.tensor([d_theta_val], dtype=torch.float32, device=self.device)

            ent_feats = torch.zeros((1, 32, 5), dtype=torch.float32, device=self.device)
            ent_types = torch.zeros((1, 32), dtype=torch.int64, device=self.device)
            ent_mask = torch.zeros((1, 32), dtype=torch.bool, device=self.device)
            rgb_depth = torch.zeros((1, 4, 48, 64), dtype=torch.float32, device=self.device)

            inputs = ModelInputs(
                rays=rays_t,
                ent_feats=ent_feats,
                ent_types=ent_types,
                ent_mask=ent_mask,
                vitals=vitals_t,
                rgb_depth=rgb_depth,
                grid=grid_t,
                action_mask=mask_t,
                d_theta=d_theta_t,
            )
        else:
            inputs, self.odom_state = preprocess(
                obs=observation,
                state=self.odom_state,
                device=self.device,
                use_grid=True,
            )

        # 2. Forward pass through HybridDoomNet
        logits, _, _, self.recurrent_state = self.model(
            inputs=inputs,
            recurrent_state=self.recurrent_state,
            return_aux=False,
        )

        # Logits of shape (1, 9) -> (9,)
        logits_arr = np.nan_to_num(
            logits.squeeze(0).cpu().numpy().astype(np.float32),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

        # 3. Low-temperature Bernoulli sampling (tau)
        # Prevents freezing in limit cycles while maintaining high fidelity to trained weights
        tau = self.tau
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits_arr / tau, -20.0, 20.0)))
        self.last_probs = probs
        rand_vals = self.rng.random(len(probs))
        action = (rand_vals < probs).astype(np.int8)

        # 4. Anti-stutter stabilization & mutual exclusion conflict resolution
        # When model has forward intent (logits > 0), do not randomly stall forward motion
        if logits_arr[0] > 0.0 and logits_arr[0] >= logits_arr[1]:
            action[0] = 1
            action[1] = 0
        elif logits_arr[1] > 0.0 and logits_arr[1] > logits_arr[0]:
            action[1] = 1
            action[0] = 0

        # STRAFE conflict resolution (MOVE_LEFT (2) vs MOVE_RIGHT (3))
        if action[2] == 1 and action[3] == 1:
            if logits_arr[2] >= logits_arr[3]:
                action[3] = 0
            else:
                action[2] = 0

        # TURN conflict resolution & anti-oscillation:
        # Prevent rapid alternating left-right twitching by committing to dominant turn
        if action[4] == 1 and action[5] == 1:
            if logits_arr[4] >= logits_arr[5]:
                action[5] = 0
            else:
                action[4] = 0
        elif logits_arr[4] > 0.2 and logits_arr[4] > logits_arr[5] + 0.15:
            action[4] = 1
            action[5] = 0
        elif logits_arr[5] > 0.2 and logits_arr[5] > logits_arr[4] + 0.15:
            action[5] = 1
            action[4] = 0

        # 5. Enforce observation action mask strictly
        act_mask = inputs.action_mask.squeeze(0).cpu().numpy()
        action = (action & act_mask.astype(np.int8)).astype(np.int8)

        # Update previous action tracker and door USE cooldown in odometry state
        if self.odom_state is not None and "grid_flat" not in observation:
            self.odom_state = update_action_state(self.odom_state, action, observation)

        if state is not _NO_STATE:
            return action, self.recurrent_state

        return action


Agent = DoomAgent
__all__ = ["DoomAgent", "Agent"]
