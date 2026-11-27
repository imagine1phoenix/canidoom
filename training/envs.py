"""Multiprocess vectorized ViZDoom runner (`VecDoomEnv`) with spawn workers and auto-recovery.

Implements:
- `multiprocessing.get_context("spawn")` worker isolation for Linux/WSL2 stability.
- Mixed `render_buffers_ratio` across workers for fast spatial + visual hybrid training.
- Coordinate-free `RewardShaper` integration inside workers exposing `terminal_privileged_state`.
- Automatic worker recycling (`worker_max_episodes`), killed/timed-out worker restart recovery,
  and dynamic `reset_with_wad(...)` map swapping for PLR curriculum.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


def _obs_key_flags(obs: Optional[dict[str, Any]]) -> tuple[Optional[float], bool]:
    """Extract training-time (min_visible_key_dist, locked_door_blocked_without_key) from obs."""
    if not obs:
        return None, False

    key_dist: Optional[float] = None
    vis = obs.get("visible_entities")
    if isinstance(vis, dict):
        mask = vis.get("mask")
        etypes = vis.get("entity_type")
        feats = vis.get("features")
        if mask is not None and etypes is not None and feats is not None:
            for k in range(min(len(mask), len(etypes), len(feats))):
                if float(mask[k]) > 0.5 and int(etypes[k]) == 9:
                    d = float(feats[k][0])
                    if key_dist is None or d < key_dist:
                        key_dist = d

    vitals = obs.get("vitals")
    has_any_key = vitals is not None and len(vitals) >= 14 and float(np.sum(vitals[8:14])) > 0.5
    locked_blocked = False
    if not has_any_key:
        rays = obs.get("rays")
        if isinstance(rays, dict):
            hit_type = rays.get("hit_type")
            dist = rays.get("distance")
            if hit_type is not None and dist is not None and len(hit_type) >= 64 and len(dist) >= 64:
                for idx in (61, 62, 63, 0, 1, 2, 3):
                    if int(hit_type[idx]) == 3 and float(dist[idx]) < 0.14:
                        locked_blocked = True
                        break

    return key_dist, locked_blocked


def _worker_main(
    conn: Any,
    worker_idx: int,
    wad_path: str,
    map_name: str,
    episode_ticks: int,
    render_buffers: bool,
    use_reward_shaper: bool,
    seed: int,
) -> None:
    """Subprocess worker owning a single isolated `doom_env.DoomEnv` and `RewardShaper`."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from doom_env import make_env
    from training.rewards import RewardShaper

    cur_wad = str(wad_path)
    cur_map = str(map_name)
    cur_ticks = int(episode_ticks)
    cur_seed = int(seed)
    env = make_env(cur_wad, map_name=cur_map, episode_ticks=cur_ticks, render_buffers=render_buffers)
    shaper = RewardShaper() if use_reward_shaper else None
    priv_prev: Optional[dict[str, Any]] = None
    episodes_done = 0

    def _do_reset(reset_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal priv_prev
        obs, info = env.reset(seed=reset_seed)
        if shaper is not None:
            shaper.reset(env.sector_polygons())
        priv_prev = env.privileged_state()
        info_out = dict(info)
        info_out["episodes_done"] = episodes_done
        info_out["render_buffers"] = render_buffers
        return obs, info_out

    try:
        while True:
            try:
                cmd, payload = conn.recv()
            except EOFError:
                break

            if cmd == "close":
                break

            if cmd == "reset":
                r_seed = cur_seed if payload is None else int(payload)
                cur_seed = r_seed + 1
                obs, info = _do_reset(r_seed)
                conn.send(("ok", (obs, info)))

            elif cmd == "reset_with_wad":
                new_wad, new_map, new_ticks, r_seed = payload
                cur_wad = str(new_wad)
                cur_map = str(new_map or "MAP01")
                if new_ticks is not None:
                    cur_ticks = int(new_ticks)
                env.close()
                env = make_env(
                    cur_wad,
                    map_name=cur_map,
                    episode_ticks=cur_ticks,
                    render_buffers=render_buffers,
                )
                use_seed = cur_seed if r_seed is None else int(r_seed)
                cur_seed = use_seed + 1
                obs, info = _do_reset(use_seed)
                conn.send(("ok", (obs, info)))

            elif cmd == "step":
                act = np.asarray(payload, dtype=np.int8)
                obs, raw_reward, terminated, truncated, info = env.step(act)
                key_dist_now, locked_blocked = _obs_key_flags(obs)
                priv_now = env.privileged_state()
                info_out = dict(info)
                info_out["key_dist"] = key_dist_now
                info_out["locked_door_blocked"] = locked_blocked

                if shaper is not None and priv_prev is not None:
                    reward = float(shaper.step(priv_prev, priv_now, act, info_out))
                    info_out["reward_components"] = dict(shaper.last_components)
                else:
                    reward = float(raw_reward)

                priv_prev = priv_now
                done = bool(terminated or truncated)
                if done:
                    episodes_done += 1
                    term_priv = dict(priv_now)
                    info_out["terminal_privileged_state"] = term_priv
                    info_out["privileged_state"] = term_priv
                    info_out["exit_reached"] = bool(
                        info_out.get("exit_reached", term_priv.get("exit_reached", False))
                    )
                    obs, _ = _do_reset(cur_seed)
                    cur_seed += 1
                else:
                    info_out["privileged_state"] = priv_now

                info_out["episodes_done"] = episodes_done
                conn.send(("ok", (obs, reward, bool(terminated), bool(truncated), info_out)))

    except Exception:
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


class VecDoomEnv:
    """Subprocess-vectorized `doom_env` manager with fault recovery and map hot-swapping."""

    def __init__(
        self,
        num_envs: int = 1,
        wad_path: str | os.PathLike = "maps/two_rooms.wad",
        map_name: str = "MAP01",
        *,
        episode_ticks: int = 2100,
        render_buffers_ratio: float = 0.25,
        worker_max_episodes: int = 100,
        worker_timeout_s: float = 15.0,
        use_reward_shaper: bool = True,
        seed: int = 42,
        wads: Optional[Sequence[str | os.PathLike]] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        ratio = float(render_buffers_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"render_buffers_ratio must be in [0, 1], got {ratio}")

        self.episode_ticks = int(episode_ticks)
        self.render_buffers_ratio = ratio
        self.worker_max_episodes = max(0, int(worker_max_episodes))
        self.worker_timeout_s = max(0.2, float(worker_timeout_s))
        self.use_reward_shaper = bool(use_reward_shaper)
        self.seed = int(seed)
        self._closed = False
        self._ctx = mp.get_context("spawn")

        if wads is not None:
            if len(wads) != self.num_envs:
                raise ValueError("len(wads) must match num_envs")
            self._wads = [str(Path(w)) for w in wads]
        else:
            self._wads = [str(Path(wad_path))] * self.num_envs
        self._maps = [str(map_name)] * self.num_envs

        if ratio <= 0.0:
            n_render = 0
        elif ratio >= 1.0:
            n_render = self.num_envs
        elif self.num_envs >= 2:
            n_render = max(1, min(self.num_envs - 1, int(round(ratio * self.num_envs))))
        else:
            n_render = 1 if ratio >= 0.5 else 0
        self._render_flags = [i < n_render for i in range(self.num_envs)]

        self._parent_conns: list[Any] = [None] * self.num_envs
        self.processes: list[Any] = [None] * self.num_envs
        self._procs = self.processes
        self.workers = self.processes
        self._episode_counts: list[int] = [0] * self.num_envs

        for idx in range(self.num_envs):
            self._spawn_worker(idx)

    def _spawn_worker(self, idx: int) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(
                child_conn,
                idx,
                self._wads[idx],
                self._maps[idx],
                self.episode_ticks,
                self._render_flags[idx],
                self.use_reward_shaper,
                self.seed + idx * 1009,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._parent_conns[idx] = parent_conn
        self.processes[idx] = proc
        self._episode_counts[idx] = 0

    def _terminate_worker(self, idx: int) -> None:
        conn = self._parent_conns[idx]
        proc = self.processes[idx]
        if conn is not None:
            try:
                conn.send(("close", None))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            self._parent_conns[idx] = None

        if proc is not None:
            try:
                if proc.is_alive():
                    proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
            except Exception:
                pass
            self.processes[idx] = None

    def restart_worker(self, idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Recycle worker `idx` cleanly (new OS PID) and reset its environment.

        Uses a generous startup timeout (30 s) so that slow ViZDoom / WSL2
        process spawn does not trigger the per-step ``worker_timeout_s`` limit.
        """
        if self._closed:
            raise RuntimeError("VecDoomEnv is closed")
        self._terminate_worker(idx)
        self._spawn_worker(idx)
        # Allow extra time for process startup (import + ViZDoom engine init).
        _startup_timeout = max(30.0, self.worker_timeout_s)
        saved = self.worker_timeout_s
        self.worker_timeout_s = _startup_timeout
        try:
            result = self._call_worker(idx, "reset", self.seed + idx * 1009 + self._episode_counts[idx])
        finally:
            self.worker_timeout_s = saved
        return result

    _restart_worker = restart_worker

    def _call_worker(self, idx: int, cmd: str, payload: Any) -> Any:
        conn = self._parent_conns[idx]
        if conn is None:
            raise EOFError("Worker pipe is closed")
        conn.send((cmd, payload))
        if not conn.poll(self.worker_timeout_s):
            raise TimeoutError(f"Worker {idx} timed out after {self.worker_timeout_s}s on {cmd}")
        status, data = conn.recv()
        if status != "ok":
            raise RuntimeError(f"Worker {idx} error on {cmd}:\n{data}")
        return data

    def reset(self, seeds: Optional[Sequence[int]] = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("VecDoomEnv is closed")
        obs_list: list[dict[str, Any]] = []
        info_list: list[dict[str, Any]] = []

        for idx in range(self.num_envs):
            s = None if seeds is None else int(seeds[idx])
            try:
                obs, info = self._call_worker(idx, "reset", s)
            except (EOFError, BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                obs, info = self.restart_worker(idx)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def reset_with_wad(
        self,
        env_idx_or_wads: int | Sequence[str | os.PathLike],
        wad_path: Optional[str | os.PathLike] = None,
        map_name: str = "MAP01",
        *,
        episode_ticks: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[Any, Any]:
        """Hot-swap the WAD/map on one worker (`env_idx`) or all workers (`list[wad]`)."""
        if self._closed:
            raise RuntimeError("VecDoomEnv is closed")

        if isinstance(env_idx_or_wads, (list, tuple)):
            wads_seq = list(env_idx_or_wads)
            if len(wads_seq) != self.num_envs:
                raise ValueError("len(wads) must match num_envs")
            obs_all: list[dict[str, Any]] = []
            info_all: list[dict[str, Any]] = []
            for i, w in enumerate(wads_seq):
                o, inf = self.reset_with_wad(
                    i,
                    w,
                    map_name=map_name,
                    episode_ticks=episode_ticks,
                    seed=None if seed is None else int(seed) + i,
                )
                obs_all.append(o)
                info_all.append(inf)
            return obs_all, info_all

        idx = int(env_idx_or_wads)
        if wad_path is None:
            raise ValueError("wad_path must be provided when resetting a single env_idx")
        self._wads[idx] = str(Path(wad_path))
        self._maps[idx] = str(map_name)
        payload = (self._wads[idx], self._maps[idx], episode_ticks, seed)
        try:
            return self._call_worker(idx, "reset_with_wad", payload)
        except (EOFError, BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return self.restart_worker(idx)

    def step(
        self, actions: np.ndarray | Sequence[Sequence[int]]
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("VecDoomEnv is closed")

        act_arr = np.asarray(actions)
        if act_arr.shape != (self.num_envs, 9):
            raise ValueError(f"Expected actions shape {(self.num_envs, 9)}, got {act_arr.shape}")
        if not np.all((act_arr == 0) | (act_arr == 1)):
            raise ValueError("Action elements must strictly be in {0, 1}")
        act_arr = act_arr.astype(np.int8)

        # Dispatch asynchronously across all workers first
        failed_send: set[int] = set()
        for idx in range(self.num_envs):
            try:
                conn = self._parent_conns[idx]
                if conn is None:
                    raise EOFError("Worker missing")
                conn.send(("step", act_arr[idx]))
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                failed_send.add(idx)

        obs_list: list[dict[str, Any]] = [None] * self.num_envs  # type: ignore[list-item]
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminateds = np.zeros((self.num_envs,), dtype=bool)
        truncateds = np.zeros((self.num_envs,), dtype=bool)
        info_list: list[dict[str, Any]] = [None] * self.num_envs  # type: ignore[list-item]

        for idx in range(self.num_envs):
            if idx in failed_send:
                obs_r, info_r = self.restart_worker(idx)
                info_r["worker_restarted"] = True
                obs_list[idx] = obs_r
                info_list[idx] = info_r
                continue

            try:
                conn = self._parent_conns[idx]
                if not conn.poll(self.worker_timeout_s):
                    raise TimeoutError(f"Worker {idx} timed out on step")
                status, data = conn.recv()
                if status != "ok":
                    raise RuntimeError(str(data))
                obs, rew, term, trunc, info = data
                if term or trunc:
                    self._episode_counts[idx] += 1
                    if self.worker_max_episodes > 0 and self._episode_counts[idx] >= self.worker_max_episodes:
                        obs, r_info = self.restart_worker(idx)
                        info["worker_recycled"] = True
                        info["episodes_done"] = r_info.get("episodes_done", 0)
                obs_list[idx] = obs
                rewards[idx] = float(rew)
                terminateds[idx] = bool(term)
                truncateds[idx] = bool(trunc)
                info_list[idx] = info
            except (EOFError, BrokenPipeError, ConnectionResetError, TimeoutError, OSError, RuntimeError):
                obs_r, info_r = self.restart_worker(idx)
                info_r["worker_restarted"] = True
                obs_list[idx] = obs_r
                rewards[idx] = 0.0
                terminateds[idx] = False
                truncateds[idx] = False
                info_list[idx] = info_r

        return obs_list, rewards, terminateds, truncateds, info_list

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for idx in range(self.num_envs):
            self._terminate_worker(idx)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
