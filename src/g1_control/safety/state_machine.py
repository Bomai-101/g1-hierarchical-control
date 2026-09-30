"""Latch-safe controller selection above the Day 9 residual-control path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_control.controllers.residual import (
    ControlMode,
    ControllerOutput,
    ControllerRegistry,
)
from g1_control.safety.watchdog import SafetyLevel, SafetySnapshot, Watchdog, WatchdogDecision


@dataclass(frozen=True)
class SafetyEvent:
    """One persistent controller transition, suitable for later logging."""

    step: int
    previous_mode: ControlMode
    selected_mode: ControlMode
    level: SafetyLevel
    reasons: tuple[str, ...]


class SafetySupervisor:
    """Run a requested controller until the watchdog latches PD fallback."""

    def __init__(
        self,
        registry: ControllerRegistry,
        requested_mode: ControlMode,
        watchdog: Watchdog | None = None,
    ):
        if requested_mode not in registry.available_modes:
            raise ValueError(f"Requested controller is unavailable: {requested_mode}")
        self.registry = registry
        self.requested_mode = requested_mode
        self.watchdog = watchdog or Watchdog()
        self.selected_mode = requested_mode
        self.events: list[SafetyEvent] = []

    @property
    def fallback_latched(self) -> bool:
        return self.selected_mode is ControlMode.PD_STAND and self.requested_mode is not ControlMode.PD_STAND

    def reset(self) -> None:
        """Start a new episode in the requested controller mode."""
        self.selected_mode = self.requested_mode
        self.events.clear()

    def act(
        self,
        observation: np.ndarray,
        height: float,
        step: int,
    ) -> tuple[ControllerOutput, WatchdogDecision]:
        snapshot = SafetySnapshot.from_observation(observation, height)
        decision = self.watchdog.evaluate(snapshot)

        if (
            decision.level is SafetyLevel.FALLBACK
            and self.selected_mode is not ControlMode.PD_STAND
        ):
            previous_mode = self.selected_mode
            self.selected_mode = ControlMode.PD_STAND
            self.events.append(
                SafetyEvent(
                    step=step,
                    previous_mode=previous_mode,
                    selected_mode=self.selected_mode,
                    level=decision.level,
                    reasons=decision.reasons,
                )
            )

        return self.registry.act(self.selected_mode, observation), decision
