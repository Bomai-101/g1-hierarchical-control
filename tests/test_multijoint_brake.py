"""Deterministic checks for predeclared multi-joint action phases."""

from __future__ import annotations

import numpy as np

from evaluate_multijoint_brake import (
    BLEND_ACTION, CASES, COUNTER_STEPS, HIP_KNEE_COUNTER_ACTION,
    MONITOR_FROM, MultiJointBrake, PULSE_ACTION, PULSE_STEPS,
)


def obs(rate: float, pitch: float = 0.0) -> np.ndarray:
    state = np.zeros(64, dtype=np.float32)
    state[59], state[62] = pitch, rate
    return state


def main() -> None:
    for mode in CASES:
        guard = MultiJointBrake(mode)
        for step in range(PULSE_STEPS):
            assert np.array_equal(guard(step, obs(0)), PULSE_ACTION)
        for step in range(PULSE_STEPS, MONITOR_FROM):
            assert not np.any(guard(step, obs(0)))
        if mode == "pulse_then_pd":
            assert not np.any(guard(MONITOR_FROM, obs(0.01)))
            assert guard.fired_at is None
            continue
        for step in (MONITOR_FROM, MONITOR_FROM + 1):
            assert not np.any(guard(step, obs(0.004)))
        first = MONITOR_FROM + 2
        action = guard(first, obs(0.004))
        assert guard.fired_at == first
        if mode == "strong_knee_guard":
            assert np.allclose(action[[1, 4]], 0.25)
            assert np.count_nonzero(action) == 2
        else:
            assert np.allclose(action[[0, 1, 3, 4]], BLEND_ACTION)
            assert np.count_nonzero(action) == 4
        for step in range(first + 1, first + 5):
            guard(step, obs(0.004))
        second = guard(first + 5, obs(-0.004))
        if mode == "hip_knee_then_counter":
            assert np.allclose(second[[0, 1, 3, 4]], -HIP_KNEE_COUNTER_ACTION)
        elif mode == "hip_knee_then_ankle":
            assert np.allclose(second[[2, 5]], -0.05)
        else:
            assert not np.any(second)
        for step in range(first + 6, first + 5 + COUNTER_STEPS):
            guard(step, obs(-0.004))
        assert not np.any(guard(first + 5 + COUNTER_STEPS, obs(-0.004)))
        assert guard.fired_at == first
    print("multi-joint brake tests: PASS")


if __name__ == "__main__":
    main()
