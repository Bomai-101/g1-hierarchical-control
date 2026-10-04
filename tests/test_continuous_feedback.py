"""Deterministic bounds and feedback-switch checks."""

from __future__ import annotations

import numpy as np

from evaluate_continuous_feedback import (
    ACTION_LIMIT, CASES, ContinuousFeedback, MONITOR_FROM, PULSE_ACTION,
    PULSE_STEPS,
)


def obs(pitch: float, rate: float) -> np.ndarray:
    state = np.zeros(64, dtype=np.float32)
    state[59], state[62] = pitch, rate
    return state


def main() -> None:
    for mode in CASES:
        controller = ContinuousFeedback(mode)
        for step in range(PULSE_STEPS):
            assert np.array_equal(controller(step, obs(0, 0)), PULSE_ACTION)
        for step in range(PULSE_STEPS, MONITOR_FROM):
            assert not np.any(controller(step, obs(0, 0)))
        if mode == "pulse_then_pd":
            assert not np.any(controller(MONITOR_FROM, obs(0, 0.004)))
            assert controller.fired_at is None
            continue
        for step in (MONITOR_FROM, MONITOR_FROM + 1):
            assert not np.any(controller(step, obs(0, 0.004)))
        fired = MONITOR_FROM + 2
        action = controller(fired, obs(0, 0.004))
        assert controller.fired_at == fired
        if mode == "strong_knee_guard":
            assert np.allclose(action[[1, 4]], 0.25)
            assert np.count_nonzero(action) == 2
            assert not np.any(controller(fired + 5, obs(0, 0.004)))
            continue
        assert np.allclose(action[[0, 1, 3, 4]], 0.008)
        assert np.count_nonzero(action) == 4
        # Continuous response changes sign rather than replaying a fixed pulse.
        action = controller(fired + 1, obs(0, -0.004))
        assert np.allclose(action[[0, 1, 3, 4]], -0.008)
        assert not np.any(controller(fired + 2, obs(0, 0.0005)))
        action = controller(fired + 3, obs(0, 0.1))
        assert np.allclose(action[[0, 1, 3, 4]], ACTION_LIMIT)
        assert controller.saturated_steps == 1
        if mode == "phase_feedback":
            action = controller(fired + 4, obs(0.01, 0))
            assert np.allclose(action[[0, 1, 3, 4]], 0.01)
        assert controller.fired_at == fired
    print("continuous feedback tests: PASS")


if __name__ == "__main__":
    main()
