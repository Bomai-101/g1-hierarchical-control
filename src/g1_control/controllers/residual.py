"""Common residual-action interface for Day 10 controller selection.

The classes here do not produce torque directly. They choose a bounded 6D
residual action, which the existing G1Env converts to target_q, PD torque, and
MuJoCo physics. This keeps the tested Day 9 low-level control path unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from g1_control.config import balance as config
from g1_control.learning.networks import Actor


class ControlMode(str, Enum):
    """Currently available low-level skill executors."""

    PD_STAND = "pd_stand"
    PPO_STAND = "ppo_stand"
    RECOVERY_STAND = "recovery_stand"


@dataclass(frozen=True)
class ControllerOutput:
    """A validated residual action plus provenance for logging."""

    mode: ControlMode
    action: np.ndarray
    action_l2_norm: float


class ResidualController(ABC):
    """Interface consumed by a future selector and safety supervisor."""

    mode: ControlMode

    @abstractmethod
    def act(self, observation: np.ndarray) -> ControllerOutput:
        """Return one bounded residual action for the current observation."""


def _validate_observation(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (config.OBS_DIM,):
        raise ValueError(
            f"Expected observation shape {(config.OBS_DIM,)}, got {obs.shape}"
        )
    if not np.isfinite(obs).all():
        raise ValueError("Observation contains NaN or infinity")
    return obs


def _make_output(mode: ControlMode, action: np.ndarray) -> ControllerOutput:
    bounded = np.asarray(action, dtype=np.float32)
    if bounded.shape != (config.ACTION_DIM,):
        raise ValueError(
            f"Expected action shape {(config.ACTION_DIM,)}, got {bounded.shape}"
        )
    if not np.isfinite(bounded).all():
        raise ValueError("Controller action contains NaN or infinity")
    bounded = np.clip(bounded, -1.0, 1.0)
    return ControllerOutput(
        mode=mode,
        action=bounded,
        action_l2_norm=float(np.linalg.norm(bounded)),
    )


class PDStandController(ResidualController):
    """The calibrated Day 9 PD standing baseline (zero residual)."""

    mode = ControlMode.PD_STAND

    def act(self, observation: np.ndarray) -> ControllerOutput:
        _validate_observation(observation)
        return _make_output(
            self.mode,
            np.zeros(config.ACTION_DIM, dtype=np.float32),
        )


class PPOStandController(ResidualController):
    """A deterministic residual PPO skill executor loaded from a checkpoint."""

    mode = ControlMode.PPO_STAND

    def __init__(self, actor: Actor, device: torch.device):
        self.actor = actor.to(device)
        self.actor.eval()
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
    ) -> "PPOStandController":
        path = checkpoint_path.resolve(strict=True)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("obs_dim", config.OBS_DIM) != config.OBS_DIM:
            raise ValueError(f"obs_dim mismatch in {path}")
        if checkpoint.get("action_dim", config.ACTION_DIM) != config.ACTION_DIM:
            raise ValueError(f"action_dim mismatch in {path}")

        actor = Actor(config.OBS_DIM, config.ACTION_DIM)
        actor.load_state_dict(checkpoint["actor"])
        return cls(actor=actor, device=device)

    def act(self, observation: np.ndarray) -> ControllerOutput:
        obs = _validate_observation(observation)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action = torch.tanh(self.actor(obs_tensor).mean).cpu().numpy()
        return _make_output(self.mode, action)


class ControllerRegistry:
    """A small selector boundary; Day 10 safety logic will sit above it."""

    def __init__(self, controllers: list[ResidualController]):
        self._controllers = {controller.mode: controller for controller in controllers}
        if ControlMode.PD_STAND not in self._controllers:
            raise ValueError("PD_STAND must always be available as the baseline")

    @property
    def available_modes(self) -> tuple[ControlMode, ...]:
        return tuple(self._controllers)

    def act(self, mode: ControlMode, observation: np.ndarray) -> ControllerOutput:
        try:
            controller = self._controllers[mode]
        except KeyError as exc:
            raise ValueError(f"Controller mode is unavailable: {mode}") from exc
        return controller.act(observation)
