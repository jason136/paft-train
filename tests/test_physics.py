"""PAFT Robot Physics Tests — individual actuator capabilities and limits.

Tests each DOF in isolation, measures range of motion, tilt response,
and records a video per scenario.

Usage: python tests/test_physics.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import paft_env  # noqa: F401

VIDEO_DIR = "./videos/test_physics"
os.makedirs(VIDEO_DIR, exist_ok=True)


def get_inner(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def measure_tilt(inner):
    qx, qy = inner.data.qpos[4], inner.data.qpos[5]
    qw = inner.data.qpos[3]
    qz = inner.data.qpos[6]
    up_z = 1 - 2 * (qx**2 + qy**2)
    return np.degrees(np.arccos(np.clip(up_z, -1, 1)))


def run(name, action_fn, duration=8.0, desc=""):
    env = RecordVideo(
        gym.make("Paft-v0", render_mode="rgb_array"),
        VIDEO_DIR, name_prefix=name, episode_trigger=lambda _: True,
    )
    env.reset()
    inner = get_inner(env)
    dt = inner.frame_skip * inner.model.opt.timestep
    n = int(duration / dt)

    x0 = inner.data.qpos[0]
    max_tilt = 0.0
    fell = False
    fell_time = None
    peak_sw = 0.0
    peak_sh = 0.0

    for step in range(n):
        t = step * dt
        action = np.array(action_fn(t), dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)

        max_tilt = max(max_tilt, measure_tilt(inner))
        peak_sw = max(peak_sw, abs(inner.data.qpos[7]))
        peak_sh = max(peak_sh, abs(inner.data.qpos[8]))

        if terminated and not fell:
            fell = True
            fell_time = t

    x_fwd = (inner.data.qpos[0] - x0) * 1000
    env.close()

    status = f"FELL@{fell_time:.1f}s" if fell else "upright"
    print(
        f"  {name:30s}  {status:12s}  tilt={max_tilt:5.1f}°"
        f"  sw={peak_sw:.2f}rad  sh={peak_sh:.4f}m"
        f"  Δx={x_fwd:+7.1f}mm   {desc}"
    )
    return {"name": name, "fell": fell, "max_tilt": max_tilt, "x_fwd": x_fwd}


if __name__ == "__main__":
    print("PAFT Physics Tests")
    print(f"Videos → {VIDEO_DIR}/\n")

    header = (
        f"  {'Scenario':30s}  {'Status':12s}  {'Tilt':>7s}"
        f"  {'Swing':>9s}  {'Shrug':>9s}  {'Δx':>9s}"
    )
    print(header)
    print("  " + "-" * 95)

    results = []

    # ── Swing tests ──────────────────────────────────────────────────────

    print()
    results.append(run(
        "both_swing_fwd_ramp",
        lambda t: [min(0.6, t * 0.12), 0, min(0.6, t * 0.12), 0],
        desc="Ramp both to +0.6 over 5s",
    ))
    results.append(run(
        "both_swing_back_ramp",
        lambda t: [max(-0.6, -t * 0.12), 0, max(-0.6, -t * 0.12), 0],
        desc="Ramp both to -0.6 over 5s",
    ))
    results.append(run(
        "both_swing_fwd_step",
        lambda t: [0.5, 0, 0.5, 0] if t > 0.5 else [0, 0, 0, 0],
        desc="Step to +0.5 at t=0.5s",
    ))
    results.append(run(
        "both_swing_back_step",
        lambda t: [-0.5, 0, -0.5, 0] if t > 0.5 else [0, 0, 0, 0],
        desc="Step to -0.5 at t=0.5s",
    ))
    results.append(run(
        "left_swing_fwd",
        lambda t: [min(0.6, t * 0.15), 0, 0, 0],
        desc="Left only to +0.6",
    ))
    results.append(run(
        "right_swing_fwd",
        lambda t: [0, 0, min(0.6, t * 0.15), 0],
        desc="Right only to +0.6",
    ))

    # ── Shrug tests ──────────────────────────────────────────────────────

    print()
    results.append(run(
        "both_shrug_up",
        lambda t: [0, 1.0, 0, 1.0],
        desc="Both shrug full UP",
    ))
    results.append(run(
        "both_shrug_down",
        lambda t: [0, -1.0, 0, -1.0],
        desc="Both shrug full DOWN",
    ))
    results.append(run(
        "shrug_asymmetric",
        lambda t: [0, 1.0, 0, -1.0],
        desc="Left UP, right DOWN",
    ))
    results.append(run(
        "shrug_alternating",
        lambda t: [
            0, np.sin(2 * np.pi * 0.5 * t),
            0, -np.sin(2 * np.pi * 0.5 * t),
        ],
        duration=10.0,
        desc="L vs R alternating 0.5Hz",
    ))

    # ── Range of motion ──────────────────────────────────────────────────

    print()
    results.append(run(
        "swing_full_sweep",
        lambda t: [
            0.8 * np.sin(2 * np.pi * 0.1 * t), 0,
            0.8 * np.sin(2 * np.pi * 0.1 * t), 0,
        ],
        duration=12.0,
        desc="Full sweep ±0.8 at 0.1Hz",
    ))
    results.append(run(
        "combined_sweep",
        lambda t: [
            0.5 * np.sin(2 * np.pi * 0.15 * t),
            np.cos(2 * np.pi * 0.15 * t),
            0.5 * np.sin(2 * np.pi * 0.15 * t),
            np.cos(2 * np.pi * 0.15 * t),
        ],
        duration=12.0,
        desc="Swing+shrug coordinated 0.15Hz",
    ))

    # ── Tilt limits ──────────────────────────────────────────────────────

    print()
    results.append(run(
        "tilt_limit_forward",
        lambda t: [min(1.0, t * 0.05), 0, min(1.0, t * 0.05), 0],
        duration=25.0,
        desc="Slow ramp forward until fall",
    ))
    results.append(run(
        "tilt_limit_backward",
        lambda t: [max(-1.0, -t * 0.05), 0, max(-1.0, -t * 0.05), 0],
        duration=25.0,
        desc="Slow ramp backward until fall",
    ))

    for amp in [0.3, 0.4, 0.5, 0.6]:
        results.append(run(
            f"step_response_A{int(amp*10)}",
            lambda t, a=amp: [a, 0, a, 0] if t > 0.5 else [0, 0, 0, 0],
            duration=6.0,
            desc=f"Step to +{amp} — tilt response",
        ))

    # ── Idle baseline ────────────────────────────────────────────────────

    print()
    results.append(run(
        "idle_standing",
        lambda t: [0, 0, 0, 0],
        duration=10.0,
        desc="No actuation — baseline",
    ))

    # ── Summary ──────────────────────────────────────────────────────────

    fell_tests = [r for r in results if r["fell"]]
    print(f"\n  Survived: {len(results) - len(fell_tests)}/{len(results)}")
    if fell_tests:
        print(f"  Falls:")
        for r in sorted(fell_tests, key=lambda r: r["max_tilt"], reverse=True):
            print(f"    {r['name']:30s}  tilt={r['max_tilt']:.1f}°")

    print(f"\n  Videos saved to {VIDEO_DIR}/")
