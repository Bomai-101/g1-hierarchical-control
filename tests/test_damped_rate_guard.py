"""Deterministic bounded-action checks for the damped Day 10 rate guard."""

from __future__ import annotations

import numpy as np

from evaluate_damped_rate_guard import (
    ACTION_LIMIT, DampedRateGuard, GAIN, MONITOR_FROM, PULSE_ACTION,
)


def observation(pitch: float, rate: float) -> np.ndarray:
    obs = np.zeros(64, dtype=np.float32)
    obs[59], obs[62] = pitch, rate
    return obs


def main() -> None:
    guard = DampedRateGuard()
    for step in range(10):
        assert np.array_equal(guard(step, observation(0, 0)), PULSE_ACTION)
    for step in range(10, MONITOR_FROM):
        assert not np.any(guard(step, observation(0, 0)))
    for step in (MONITOR_FROM, MONITOR_FROM + 1):
        assert not np.any(guard(step, observation(0, 0.004)))
    action = guard(MONITOR_FROM + 2, observation(0, 0.004))
    assert guard.fired_at == MONITOR_FROM + 2
    assert abs(float(action[1]) - GAIN * 0.004) < 1e-8
    assert action[1] == action[4] and np.count_nonzero(action) == 2
    action = guard(MONITOR_FROM + 3, observation(0, 0.1))
    assert abs(float(action[1]) - ACTION_LIMIT) < 1e-8
    assert abs(float(action[4]) - ACTION_LIMIT) < 1e-8
    action = guard(MONITOR_FROM + 4, observation(0, -0.002))
    assert abs(float(action[1]) + GAIN * 0.002) < 1e-8
    for step in range(MONITOR_FROM + 5, MONITOR_FROM + 12):
        guard(step, observation(0, 0.1))
    assert not np.any(guard(MONITOR_FROM + 12, observation(0, 0.1)))
    assert guard.fired_at == MONITOR_FROM + 2  # one-shot, no retrigger
    print("damped rate guard tests: PASS")


if __name__ == "__main__":
    main()
