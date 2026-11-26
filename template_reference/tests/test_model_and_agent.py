"""Unit and regression tests for HybridDoomNet and DoomAgent.

Implements the tests specified in planning/team/Aryan_LOQ.md §4:
- test_zero_entities_no_nan
- test_reset_clears_memory
- test_single_thread_latency (< 25 ms)
- test_action_contract (shape, dtype, values in {0, 1})
- test_action_masking_use_cooldown
- test_scan_matching_rotation
"""

import math
import time
import numpy as np
import pytest
import torch

from gymnasium.spaces import Box, MultiBinary
from submission.agent import DoomAgent
from training.model import HybridDoomNet, RecurrentState
from training.preprocessing import (
    preprocess,
    ModelInputs,
    OdomGridState,
    compute_action_mask,
)
from training.odometry_grid import estimate_yaw_rotation, update_odometry_and_grid


def make_dummy_observation() -> dict:
    """Creates a contract-compliant observation dictionary."""
    return {
        "rgb": np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
        "depth": np.random.randint(0, 256, (120, 160), dtype=np.uint8),
        "vitals": np.array(
            [100.0, 50.0, 1.0, 50.0, 50.0, 20.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "rays": {
            "distance": np.random.uniform(0.1, 0.9, (64,)).astype(np.float32),
            "hit_type": np.random.randint(0, 6, (64,), dtype=np.int64),
        },
        "visible_entities": {
            "features": np.random.uniform(-1.0, 1.0, (32, 5)).astype(np.float32),
            "entity_type": np.random.randint(0, 12, (32,), dtype=np.int64),
            "mask": np.random.choice([0, 1], size=(32,), p=[0.7, 0.3]).astype(np.int8),
        },
    }


def test_zero_entities_no_nan():
    """Verify feeding zero entities does not cause NaN in logits or values."""
    obs = make_dummy_observation()
    obs["visible_entities"]["mask"][:] = 0
    obs["visible_entities"]["features"][:] = 0.0
    obs["visible_entities"]["entity_type"][:] = 0

    inputs, _ = preprocess(obs, device="cpu")
    model = HybridDoomNet()
    model.eval()

    logits, value, next_state, aux = model(inputs, return_aux=True)

    assert not torch.isnan(logits).any(), "NaN found in logits with zero entities!"
    assert not torch.isnan(value).any(), "NaN found in value with zero entities!"
    assert not torch.isnan(next_state.obj_slots).any(), "NaN found in objective slots!"


def test_action_contract():
    """Verify DoomAgent returns np.int8 array of shape (9,) in {0, 1}."""
    obs_space = Box(-np.inf, np.inf, (1,), np.float32)
    act_space = MultiBinary(9)
    agent = DoomAgent(obs_space, act_space, device="cpu")

    obs = make_dummy_observation()
    action = agent.act(obs)

    assert isinstance(action, np.ndarray), "Action must be a numpy array"
    assert action.shape == (9,), f"Action shape must be (9,), got {action.shape}"
    assert action.dtype == np.int8, f"Action dtype must be int8, got {action.dtype}"
    assert set(np.unique(action)).issubset({0, 1}), f"Action values must be in {{0, 1}}, got {set(np.unique(action))}"


def test_reset_clears_memory():
    """Verify reset() creates bit-for-bit identical step 0 action sequence."""
    obs_space = Box(-np.inf, np.inf, (1,), np.float32)
    act_space = MultiBinary(9)

    agent = DoomAgent(obs_space, act_space, device="cpu")
    obs = make_dummy_observation()

    # Step on map A for 10 steps
    for _ in range(10):
        agent.act(obs)

    # Reset before map B
    agent.reset()
    action_after_reset = agent.act(obs)

    # Fresh instance on map B
    fresh_agent = DoomAgent(obs_space, act_space, device="cpu")
    fresh_agent.model.load_state_dict(agent.model.state_dict())
    action_fresh = fresh_agent.act(obs)

    np.testing.assert_array_equal(
        action_after_reset,
        action_fresh,
        err_msg="agent.reset() did not fully isolate recurrent / odometry memory!",
    )


def test_single_thread_latency():
    """Verify single-thread act() completes well within 25 ms target (limit: 40 ms)."""
    torch.set_num_threads(1)
    obs_space = Box(-np.inf, np.inf, (1,), np.float32)
    act_space = MultiBinary(9)
    agent = DoomAgent(obs_space, act_space, device="cpu")

    obs = make_dummy_observation()

    # Warmup
    for _ in range(10):
        agent.act(obs)

    # Timed run
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        agent.act(obs)
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = np.mean(times)
    p99_ms = np.percentile(times, 99)

    print(f"\n[Latency Benchmark] Mean: {mean_ms:.2f} ms | p99: {p99_ms:.2f} ms")
    assert mean_ms < 25.0, f"Mean act() time {mean_ms:.2f} ms exceeds 25 ms budget!"
    assert p99_ms < 50.0, f"p99 act() time {p99_ms:.2f} ms exceeds 50 ms cap!"


def test_action_masking_use_cooldown():
    """Verify door USE cooldown suppresses USE=1 for 6 steps."""
    state = OdomGridState()
    obs = make_dummy_observation()

    # Front ray sees a door close by
    obs["rays"]["hit_type"][0] = 2  # door
    obs["rays"]["distance"][0] = 0.08  # ~80 units

    # Press USE in step 0
    state.prev_action = np.zeros(9, dtype=np.int8)
    state.prev_action[7] = 1

    # Step 1: Preprocess should trigger cooldown
    inputs, state = preprocess(obs, state, device="cpu")
    mask = inputs.action_mask.squeeze(0).cpu().numpy()
    assert mask[7] == False, "USE must be masked during door cooldown!"
    assert state.use_cooldown == 6, f"Cooldown should be 6, got {state.use_cooldown}"


def test_scan_matching_rotation():
    """Verify circular scan matching accurately estimates a simulated 3-bin turn."""
    curr_dist = np.sin(np.linspace(0, 4 * math.pi, 64)).astype(np.float32) * 0.4 + 0.5
    curr_type = np.ones(64, dtype=np.int64)  # walls

    # Simulate 3-bin turn left (shift by +3)
    prev_dist = np.roll(curr_dist, -3)
    prev_action = np.zeros(9, dtype=np.int8)
    prev_action[4] = 1  # TURN_LEFT

    d_theta = estimate_yaw_rotation(curr_dist, curr_type, prev_dist, prev_action)
    expected_d_theta = 3 * (2.0 * math.pi / 64.0)

    assert abs(d_theta - expected_d_theta) < (math.pi / 64.0), (
        f"Estimated d_theta {d_theta:.4f} differs from expected {expected_d_theta:.4f}"
    )
