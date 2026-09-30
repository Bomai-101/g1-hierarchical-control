import numpy as np

from g1_control.config import balance as balance_config
from g1_control.config.recovery import RECOVERY_ACTOR_LOG_STD
from g1_control.learning.networks import Actor


def test_default_actor_preserves_day9_distribution():
    actor = Actor()
    assert np.isclose(float(actor.log_std.mean()), balance_config.ACTOR_LOG_STD)


def test_recovery_actor_uses_wider_fixed_distribution():
    actor = Actor(log_std=RECOVERY_ACTOR_LOG_STD)
    assert np.isclose(float(actor.log_std.mean()), RECOVERY_ACTOR_LOG_STD)
    assert not actor.log_std.requires_grad
    assert np.isclose(float(actor.log_std.exp().mean()), np.exp(-2.0))


if __name__ == "__main__":
    test_default_actor_preserves_day9_distribution()
    test_recovery_actor_uses_wider_fixed_distribution()
    print("actor log_std tests: PASS")
