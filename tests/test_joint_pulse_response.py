"""Basic action-isolation checks for the exact-state response probe."""

from __future__ import annotations

import numpy as np

from evaluate_joint_pulse_response import AMPLITUDE, CHANNELS, action_for, case_specs


def main() -> None:
    specs = case_specs()
    assert len(specs) == 19
    assert len(set(specs)) == len(specs)
    assert len(CHANNELS) == 9
    assert not np.any(action_for("zero", 0))
    for channel in CHANNELS:
        for sign in (-1, 1):
            action = action_for(channel, sign)
            assert action.shape == (6,)
            assert set(np.flatnonzero(action)) == set(CHANNELS[channel])
            assert np.allclose(action[list(CHANNELS[channel])], sign * AMPLITUDE)
    for channel, sign in (("zero", 1), ("unknown", 1), ("hip_pair", 0)):
        try:
            action_for(channel, sign)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid action accepted: {channel}, {sign}")
    print("joint pulse response tests: PASS")


if __name__ == "__main__":
    main()
