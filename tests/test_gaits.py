"""PAFT Gait Tests — slow / medium / aggressive variants of the tuned gait.

Uses the exact same piecewise waveform defined in paft_env._gait_targets().
Records one video per variant. The best result should be reflected back
into paft_env.py's GAIT_FREQ and SWING_AMPLITUDE.

Usage: python tests/test_gaits.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import paft_env  # noqa: F401

VIDEO_DIR = "./videos/test_gaits"
os.makedirs(VIDEO_DIR, exist_ok=True)


def get_inner(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def gait_action(phase, amplitude, shrug_amp=0.020):
    """Piecewise gait waveform — mirrors paft_env._gait_targets().

    Shoulders UP → arms forward in air → gravity tilts body → arms land →
    shoulders equalize, body pulls forward → shoulders DOWN to lift body
    off floor → body swings forward → arms recover forward → repeat.

    Cycle:
      0.00–0.06  LIFT: shrug 0 → +S                  (shoulders up)
      0.05–0.15  REACH: swing 0 → +A, shrug +S       (arms fwd in air)
      0.15–0.33  HOLD: swing +A, shrug +S              (wait for gravity tilt)
      0.33–0.71  PULL: swing +A → 0, shrug +S → -S    (shoulders settle past
                                                       neutral as body follows)
      0.71–0.77  PUSH: swing 0 → -A, shrug -S         (arms back, body lifts)
      0.77–0.87  HOLD2: hold -A, shrug -S              (wait for body swing fwd)
      0.87–0.95  RECOVER: swing -A → 0, shrug -S → 0  (arms fwd, equalize)
      0.95–1.00  SETTLE: swing 0, shrug 0              (prep next)
    """
    A = amplitude
    S = shrug_amp

    if phase < 0.05:
        frac = phase / 0.05
        sw, sh = 0.0, S * frac
    elif phase < 0.15:
        frac = (phase - 0.05) / 0.10
        sw, sh = A * frac, S
    elif phase < 0.33:
        sw, sh = A, S
    elif phase < 0.71:
        frac = (phase - 0.33) / 0.38
        sw, sh = A * (1.0 - frac), S * (1.0 - 2.0 * frac)
    elif phase < 0.77:
        frac = (phase - 0.71) / 0.06
        sw, sh = -A * frac, -S
    elif phase < 0.87:
        sw, sh = -A, -S
    elif phase < 0.95:
        frac = (phase - 0.87) / 0.08
        sw, sh = -A * (1.0 - frac), -S * (1.0 - frac)
    else:
        sw, sh = 0.0, 0.0

    sh_act = sh / 0.02
    return np.array([sw, sh_act, sw, sh_act], dtype=np.float32)


def run_gait(name, freq, amplitude, duration=25.0):
    env = RecordVideo(
        gym.make("Paft-v0", render_mode="rgb_array"),
        VIDEO_DIR, name_prefix=name, episode_trigger=lambda _: True,
    )
    env.reset()
    inner = get_inner(env)
    dt = inner.frame_skip * inner.model.opt.timestep
    n_steps = int(duration / dt)
    cycle_time = 1.0 / freq

    x0 = inner.data.qpos[0]
    fell = False
    fell_time = None
    max_tilt = 0.0
    peak_swing = 0.0

    for step in range(n_steps):
        t = step * dt
        phase = (t * freq) % 1.0
        action = gait_action(phase, amplitude)
        _, _, terminated, _, _ = env.step(action)

        qw, qx, qy, qz = inner.data.qpos[3:7]
        up_z = 1 - 2 * (qx**2 + qy**2)
        tilt = np.degrees(np.arccos(np.clip(up_z, -1, 1)))
        max_tilt = max(max_tilt, tilt)
        peak_swing = max(peak_swing, abs(inner.data.qpos[7]))

        if terminated and not fell:
            fell = True
            fell_time = t

    x_fwd = (inner.data.qpos[0] - x0) * 1000
    env.close()

    n_cycles = duration * freq
    alive_time = fell_time if fell else duration
    mm_per_cycle = x_fwd / (alive_time * freq)

    status = f"FELL @ {fell_time:.1f}s" if fell else "upright"
    print(
        f"  {name:15s}  freq={freq:.2f}Hz  A={amplitude:.2f}rad ({np.degrees(amplitude):4.0f}°)"
        f"  cycle={cycle_time:.1f}s"
    )
    print(
        f"  {'':15s}  {status:14s}  max_tilt={max_tilt:5.1f}°"
        f"  peak_sw={peak_swing:.3f}rad"
        f"  Δx={x_fwd:+8.1f}mm  per_cycle={mm_per_cycle:+.1f}mm"
    )
    print()
    return {
        "name": name, "freq": freq, "amplitude": amplitude,
        "fell": fell, "fell_time": fell_time,
        "max_tilt": max_tilt, "x_fwd": x_fwd, "mm_per_cycle": mm_per_cycle,
        "peak_swing": peak_swing,
    }


if __name__ == "__main__":
    from paft_env import PaftEnv

    print("PAFT Gait Tests — Tuned Piecewise Waveform")
    print(f"Current env:  GAIT_FREQ={PaftEnv.GAIT_FREQ}  SWING_AMPLITUDE={PaftEnv.SWING_AMPLITUDE}")
    print(f"Videos → {VIDEO_DIR}/\n")

    results = []

    results.append(run_gait("slow",       freq=0.10, amplitude=0.55))
    results.append(run_gait("medium",     freq=0.15, amplitude=0.70))
    results.append(run_gait("aggressive", freq=0.20, amplitude=0.85))

    print("=" * 80)
    print("RANKING BY FORWARD DISPLACEMENT")
    print("=" * 80)
    results.sort(key=lambda r: r["x_fwd"], reverse=True)
    for r in results:
        tag = f"FELL@{r['fell_time']:.1f}s" if r["fell"] else "OK"
        print(
            f"  {r['name']:15s}  A={r['amplitude']:.2f}  f={r['freq']:.2f}Hz"
            f"  Δx={r['x_fwd']:+8.1f}mm  per_cycle={r['mm_per_cycle']:+6.1f}mm"
            f"  tilt={r['max_tilt']:5.1f}°  {tag}"
        )

    best = results[0]
    matches_env = (
        best["freq"] == PaftEnv.GAIT_FREQ
        and best["amplitude"] == PaftEnv.SWING_AMPLITUDE
    )
    print(f"\nBest: {best['name']}")
    if matches_env:
        print("  Already matches paft_env.py defaults.")
    else:
        print(f"  To update paft_env.py:")
        print(f"    GAIT_FREQ = {best['freq']}")
        print(f"    SWING_AMPLITUDE = {best['amplitude']}")
    print()
