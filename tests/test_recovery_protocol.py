import numpy as np

from g1_control.config import balance as balance_config
from g1_control.config.recovery import RecoveryCriteria
from g1_control.evaluation.recovery_protocol import (
    RecoveryTracker,
    measure_recovery_state,
)


def stable_state():
    observation = np.zeros(balance_config.OBS_DIM, dtype=np.float32)
    default_q = np.zeros(balance_config.NUM_JOINTS, dtype=np.float32)
    return measure_recovery_state(observation, height=0.80, default_q=default_q)


def unstable_state():
    observation = np.zeros(balance_config.OBS_DIM, dtype=np.float32)
    observation[59] = 0.20
    default_q = np.zeros(balance_config.NUM_JOINTS, dtype=np.float32)
    return measure_recovery_state(observation, height=0.80, default_q=default_q)


def test_requires_entry_and_final_hold_window():
    criteria = RecoveryCriteria(entry_steps=3, final_hold_steps=5)
    tracker = RecoveryTracker(criteria)

    for step in range(5):
        tracker.update(step, stable_state())

    assert tracker.entered_at == 0
    assert tracker.succeeded(survived=True)
    assert not tracker.succeeded(survived=False)


def test_transient_entry_is_marked_as_relapse_not_success():
    criteria = RecoveryCriteria(entry_steps=3, final_hold_steps=5)
    tracker = RecoveryTracker(criteria)

    for step in range(3):
        tracker.update(step, stable_state())
    tracker.update(3, unstable_state())

    assert tracker.entered_at == 0
    assert tracker.relapsed
    assert not tracker.succeeded(survived=True)


if __name__ == "__main__":
    test_requires_entry_and_final_hold_window()
    test_transient_entry_is_marked_as_relapse_not_success()
    print("recovery protocol tests: PASS")
