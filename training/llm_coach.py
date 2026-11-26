"""Offline & training-time LLM Curriculum Advisor and Scorecard Analyst (`@cf/openai/gpt-oss-120b`).

Reads `CREO_BASE_URL`, `CREO_MODEL`, and `CREO_API_KEY` solely from environment variables
with a deterministic, zero-network fallback whenever `CREO_API_KEY` is unset or an HTTP/JSON
error occurs.

CRITICAL: Strictly for training/offline analysis. NEVER imported by `submission/agent.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_BASE_URL = "https://creo.creoai.workers.dev/v1"
DEFAULT_MODEL = "@cf/openai/gpt-oss-120b"


def _clamp_knobs(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(fallback)
    if "num_rooms" in raw:
        try:
            out["num_rooms"] = int(max(1, min(12, round(float(raw["num_rooms"])))))
        except (TypeError, ValueError):
            pass
    if "enemies_per_room" in raw:
        try:
            val = float(raw["enemies_per_room"])
            clamped = max(0.0, min(12.0, val))
            out["enemies_per_room"] = int(round(clamped)) if float(clamped).is_integer() else round(clamped, 2)
        except (TypeError, ValueError):
            pass
    if "ammo_mult" in raw:
        try:
            out["ammo_mult"] = round(float(max(0.3, min(2.5, float(raw["ammo_mult"])))), 2)
        except (TypeError, ValueError):
            pass
    if "tier_weights" in raw and isinstance(raw["tier_weights"], (list, tuple)) and len(raw["tier_weights"]) == 4:
        try:
            out["tier_weights"] = [round(float(max(0.2, min(5.0, float(x)))), 2) for x in raw["tier_weights"]]
        except (TypeError, ValueError):
            pass
    return out


def _deterministic_curriculum_fallback(
    telemetry: dict[str, Any],
    current_knobs: dict[str, Any],
) -> dict[str, Any]:
    num_rooms = int(current_knobs.get("num_rooms", 4))
    enemies = float(current_knobs.get("enemies_per_room", 2.0))
    ammo = float(current_knobs.get("ammo_mult", 1.0))

    exit_rate = float(telemetry.get("exit_rate", 0.3))
    stuck_rate = float(telemetry.get("stuck_rate", 0.1))
    coverage = float(telemetry.get("sector_coverage", 0.5))

    if stuck_rate > 0.30 or exit_rate < 0.15:
        num_rooms = max(2, num_rooms - 1)
        enemies = max(1, round(enemies - 0.5))
        ammo = min(2.5, round(ammo + 0.15, 2))
    elif exit_rate > 0.65 and coverage > 0.70:
        num_rooms = min(10, num_rooms + 1)
        enemies = min(8, round(enemies + 1))
        ammo = max(0.4, round(ammo - 0.10, 2))

    base = {
        "num_rooms": num_rooms,
        "enemies_per_room": int(enemies) if float(enemies).is_integer() else enemies,
        "ammo_mult": ammo,
    }
    return _clamp_knobs(base, base)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", stripped, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _chat_completion(prompt: str, timeout: float = 15.0) -> Optional[str]:
    api_key = os.environ.get("CREO_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.environ.get("CREO_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    model = os.environ.get("CREO_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an RL DOOM curriculum advisor. Respond concisely and follow JSON schema strictly when asked.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "OpenAI/Python 1.50.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning")
        return str(content) if content else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def advise_curriculum(
    telemetry: dict[str, Any],
    current_knobs: dict[str, Any],
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Query `@cf/openai/gpt-oss-120b` for bounded PLR knob updates with deterministic fallback."""
    fallback = _deterministic_curriculum_fallback(telemetry or {}, current_knobs or {})
    prompt = (
        "Given DOOM RL training telemetry and current procedural map knobs, return ONLY a compact JSON object "
        "with bounded keys: num_rooms (int 1..12), enemies_per_room (0..12), ammo_mult (float 0.3..2.5).\n"
        f"Telemetry: {json.dumps(telemetry or {})}\n"
        f"Current knobs: {json.dumps(current_knobs or {})}"
    )
    reply = _chat_completion(prompt, timeout=timeout)
    if not reply:
        return fallback
    parsed = _extract_json_object(reply)
    if not parsed:
        return fallback
    return _clamp_knobs(parsed, fallback)


def summarize_evaluation(scorecard: dict[str, Any], timeout: float = 15.0) -> str:
    """Summarize a held-out evaluation scorecard into concise diagnostic notes."""
    sc = scorecard or {}
    total = float(sc.get("total_score", 0.0))
    exit_rate = float(sc.get("exit_rate", 0.0))
    mean_ms = float(sc.get("mean_act_ms", 0.0))
    passed = sc.get("passed_limits", True)
    fallback = (
        f"Offline Scorecard Summary: total_score={total:.2f}, exit_rate={exit_rate * 100:.1f}%, "
        f"mean_act_ms={mean_ms:.2f}ms, passed_limits={passed}."
    )

    prompt = (
        "Summarize this held-out DOOM agent scorecard in 2 concise technical sentences highlighting "
        "tier generalization, bottlenecks, and latency headroom:\n"
        f"{json.dumps(sc)}"
    )
    reply = _chat_completion(prompt, timeout=timeout)
    if reply and reply.strip():
        return reply.strip()
    return fallback


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.llm_coach",
        description="Offline / training-only LLM curriculum advisor & scorecard summarizer.",
    )
    parser.add_argument("--mode", choices=["advise", "summarize"], default="summarize")
    parser.add_argument("--telemetry-json", type=str, default="{}")
    parser.add_argument("--knobs-json", type=str, default='{"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0}')
    parser.add_argument("--scorecard-json", type=str, default="{}")
    args = parser.parse_args(argv)

    if args.mode == "advise":
        tel = json.loads(args.telemetry_json)
        knobs = json.loads(args.knobs_json)
        print(json.dumps(advise_curriculum(tel, knobs), indent=2))
    else:
        sc = json.loads(args.scorecard_json)
        print(summarize_evaluation(sc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
