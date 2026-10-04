"""Basic reference-action and event-marker checks for pulse timelines."""

from __future__ import annotations

import numpy as np

from compare_first_pulse_trajectories import (
    CASES, PULSE_ACTION, PULSE_STEPS, choose_action, first_sustained,
)
from evaluate_early_rate_guard import EarlyRateGuard


class FakePPO:
    def act(self, obs):
        class Output:
            action = np.full(6, .1, dtype=np.float32)
        return Output()


def main() -> None:
    obs = np.zeros(64, dtype=np.float32)
    ppo = FakePPO()
    guard = EarlyRateGuard(1)
    assert len(CASES) == 4
    assert not np.any(choose_action("no_pulse_pd", 0, obs, ppo, guard))
    assert np.allclose(choose_action("no_pulse_ppo", 0, obs, ppo, guard), .1)
    for step in range(PULSE_STEPS):
        assert np.array_equal(choose_action("pulse_then_pd", step, obs, ppo, guard), PULSE_ACTION)
    assert not np.any(choose_action("pulse_then_pd", PULSE_STEPS, obs, ppo, guard))
    rows = [{"local_step": step, "flag": step in (3, 4, 5, 6)} for step in range(1, 8)]
    assert first_sustained(rows, lambda row: row["flag"], 3) == 3
    assert first_sustained(rows, lambda row: row["flag"], 5) is None
    print("first-pulse timeline tests: PASS")


if __name__ == "__main__":
    main()
