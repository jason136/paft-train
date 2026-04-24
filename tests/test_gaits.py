"""PAFT Gait Tests — keyframe-based open-loop gait.

Each keyframe defines a target pose (swing, shrug) and a duration.
The robot interpolates linearly between keyframes. Cycle time is
the sum of all durations.

Usage: python tests/test_gaits.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import paft_env  # noqa: F401
from paft_env import PaftEnv

VIDEO_DIR = "./videos/test_gaits"
os.makedirs(VIDEO_DIR, exist_ok=True)

A = PaftEnv.SWING_AMPLITUDE  # 0.70 rad
S = PaftEnv.SHRUG_AMPLITUDE  # 0.005 m

# Keyframes: (duration_s, swing_target, shrug_target)
# Robot interpolates from previous keyframe to this one over duration_s.
KEYFRAMES = [
    # --- ARM PHASE ---
    (1.5, 0.0, S),  # lift: shrug up, arms neutral
    (1.2, A, S),  # reach: swing forward, shrug stays up
    (0.4, A, -S),  # plant: hold swing, flip shrug down
    (0.8, 0.0, -S),  # reconcile: swing + shrug back to neutral
    (0.3, 0.0, -S),  # rest: neutral
    # --- BODY PHASE ---
    (1.5, 0.0, -S),  # lift body: shrug down, plants arms
    (1.2, -A * 0.9, -S),  # push: swing backward, shrug stays down
    (0.4, -A * 0.9, S),  # release: hold swing, flip shrug up
    (0.4, 0.0, S),  # reconcile: swing + shrug back to neutral
    (0.3, 0.0, S),  # rest: neutral
]

DURATIONS = [kf[0] for kf in KEYFRAMES]
CYCLE_TIME = sum(DURATIONS)
CUM_TIMES = np.cumsum([0.0] + DURATIONS)


def gait_action(t_in_cycle):
    """Interpolate between keyframes at the given cycle time."""
    t = t_in_cycle

    # Find which segment we're in
    for i in range(len(KEYFRAMES)):
        if t < CUM_TIMES[i + 1]:
            frac = (t - CUM_TIMES[i]) / DURATIONS[i]
            # Previous target (or last keyframe if wrapping)
            if i == 0:
                prev_sw = KEYFRAMES[-1][1]
                prev_sh = KEYFRAMES[-1][2]
            else:
                prev_sw = KEYFRAMES[i - 1][1]
                prev_sh = KEYFRAMES[i - 1][2]
            curr_sw = KEYFRAMES[i][1]
            curr_sh = KEYFRAMES[i][2]
            sw = prev_sw + (curr_sw - prev_sw) * frac
            sh = prev_sh + (curr_sh - prev_sh) * frac
            sw_act = sw / 1.5708
            sh_act = sh / 0.005
            return np.array([sw_act, sh_act, sw_act, sh_act], dtype=np.float32)

    # Fallback
    return np.zeros(4, dtype=np.float32)


def get_inner(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def run_gait(name, duration=25.0):
    env = RecordVideo(
        gym.make("Paft-v0", render_mode="rgb_array"),
        VIDEO_DIR,
        name_prefix=name,
        episode_trigger=lambda _: True,
    )
    inner_env = env.unwrapped if not hasattr(env, "env") else env.env.unwrapped
    inner_env.set_start_stage(0, deterministic=True)
    env.reset()
    inner = get_inner(env)
    dt = inner.frame_skip * inner.model.opt.timestep
    n_steps = int(duration / dt)

    x0 = inner.data.qpos[0]
    fell = False
    fell_time = None
    max_tilt = 0.0
    transitions = 0
    prev_stage = 0

    for step in range(n_steps):
        t = step * dt
        t_in_cycle = t % CYCLE_TIME
        action = gait_action(t_in_cycle)
        _, _, terminated, _, info = env.step(action)

        qw, qx, qy, qz = inner.data.qpos[3:7]
        up_z = 1 - 2 * (qx**2 + qy**2)
        tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1)))
        max_tilt = max(max_tilt, tilt)

        stage = info.get("gait_stage", 0)
        if stage != prev_stage:
            transitions += 1
            prev_stage = stage

        if terminated and not fell:
            fell = True
            fell_time = t

    x_fwd = (inner.data.qpos[0] - x0) * 1000
    steps_completed = info.get("steps_completed", 0)
    env.close()

    status = f"FELL @ {fell_time:.1f}s" if fell else "upright"
    print(f"  {name:15s}  cycle={CYCLE_TIME:.1f}s  A={A:.2f}rad  S={S*1000:.0f}mm")
    print(
        f"  {'':15s}  {status:14s}  tilt={max_tilt:5.1f}°"
        f"  trans={transitions}  cycles={steps_completed}"
        f"  Δx={x_fwd:+8.1f}mm"
    )
    print()
    return {
        "name": name,
        "fell": fell,
        "fell_time": fell_time,
        "max_tilt": max_tilt,
        "x_fwd": x_fwd,
        "transitions": transitions,
        "cycles": steps_completed,
    }


if __name__ == "__main__":
    print("PAFT Gait Test — Keyframe Interpolation")
    print(f"Cycle time: {CYCLE_TIME:.1f}s ({1/CYCLE_TIME:.2f} Hz)")
    print(f"Keyframes:")
    t = 0.0
    prev_sw, prev_sh = KEYFRAMES[-1][1], KEYFRAMES[-1][2]
    for dur, sw, sh in KEYFRAMES:
        print(
            f"  {t:5.1f}–{t+dur:5.1f}s  sw:{prev_sw:+.2f}→{sw:+.2f}  sh:{prev_sh*1000:+.0f}→{sh*1000:+.0f}mm  ({dur:.1f}s)"
        )
        prev_sw, prev_sh = sw, sh
        t += dur
    print(f"Amplitudes: swing={A}rad  shrug={S*1000}mm")
    print(f"Videos → {VIDEO_DIR}/\n")

    results = []
    results.append(run_gait("default"))

    print("=" * 80)
    for r in results:
        tag = f"FELL@{r['fell_time']:.1f}s" if r["fell"] else "OK"
        print(
            f"  {r['name']:15s}"
            f"  trans={r['transitions']:3d}  cycles={r['cycles']}"
            f"  Δx={r['x_fwd']:+8.1f}mm  tilt={r['max_tilt']:5.1f}°  {tag}"
        )
    print()
