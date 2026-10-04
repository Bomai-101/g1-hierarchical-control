"""Deterministic timing and sign checks for the Day 10 early rate guard."""

from __future__ import annotations

import numpy as np

from evaluate_early_rate_guard import (
    ACTION_AMPLITUDE, ACTION_STEPS, MONITOR_FROM, PULSE_ACTION,
    EarlyRateGuard, post_trigger_metrics,
)


def observation(pitch: float, rate: float) -> np.ndarray:
    obs = np.zeros(64, dtype=np.float32)
    obs[59], obs[62] = pitch, rate
    return obs


def check(direction: int, rate: float, expected_knee: float) -> None:
    guard = EarlyRateGuard(direction)
    quiet = observation(0.0, 0.0)
    active = observation(0.0, rate)
    for step in range(10):
        assert np.array_equal(guard(step, quiet), PULSE_ACTION)
    for step in range(10, MONITOR_FROM):
        assert not np.any(guard(step, quiet))
    assert not np.any(guard(MONITOR_FROM, active))
    assert not np.any(guard(MONITOR_FROM + 1, active))
    for step in range(MONITOR_FROM + 2, MONITOR_FROM + 2 + ACTION_STEPS):
        action = guard(step, active)
        assert action[1] == action[4] == expected_knee
        assert np.count_nonzero(action) == 2
    assert guard.fired_at == MONITOR_FROM + 2
    for step in range(MONITOR_FROM + 2 + ACTION_STEPS,
                      MONITOR_FROM + 2 + ACTION_STEPS + 10):
        assert not np.any(guard(step, active))


def main() -> None:
    check(1, 0.004, ACTION_AMPLITUDE)
    check(1, -0.004, -ACTION_AMPLITUDE)
    check(-1, 0.004, -ACTION_AMPLITUDE)
    guard = EarlyRateGuard(1)
    for step in range(MONITOR_FROM, MONITOR_FROM + 10):
        assert not np.any(guard(step, observation(0.03, 0.004)))
    assert guard.fired_at is None
    guard.fired_at = 32
    guard.trigger_pitch = 0.0
    guard.trigger_rate = 0.004
    trace = [{"pitch_rate": 0.0} for _ in range(42)]
    trace[36]["pitch_rate"] = -0.07
    metrics = post_trigger_metrics({"trace": trace}, guard)
    assert metrics["opposite_rate_at_pulse_end"] is True
    assert metrics["rate_magnitude_reduced_at_pulse_end"] is False
    assert abs(metrics["pulse_end_rate_gain"] - 17.5) < 1e-10
    print("early rate guard tests: PASS")


if __name__ == "__main__":
    main()
