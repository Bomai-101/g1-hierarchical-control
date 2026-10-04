"""Measured-goal skill sequencing with a task-message freshness watchdog.

Only scalar robot samples and velocity commands cross this boundary. A stale
message blocks stage advancement; it does not stop the robot. The last skill
command persists until a fresh handoff or the simulation experiment ends.
There is no physical fallback, learned planner, or perception fault detector.
"""
from dataclasses import dataclass
from math import cos, sin, isfinite


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    sequence: int
    issued_s: float


@dataclass(frozen=True)
class RobotSample:
    time_s: float
    world_x_m: float
    world_y_m: float
    heading_unwrapped_rad: float


@dataclass(frozen=True)
class Stage:
    name: str
    goal_kind: str
    goal: float
    command: tuple[float, float, float]
    timeout_s: float = 8.0

    def __post_init__(self):
        if self.goal_kind not in ('forward_distance', 'relative_heading'):
            raise ValueError('Unsupported goal kind')
        if not self.name or len(self.command) != 3:
            raise ValueError('Stage name and 3D velocity command required')
        if not all(isfinite(v) for v in (*self.command, self.goal, self.timeout_s)):
            raise ValueError('Stage values must be finite')
        if self.timeout_s <= 0 or self.goal == 0:
            raise ValueError('Nonzero goal and positive timeout required')
        if self.goal_kind == 'forward_distance' and self.goal < 0:
            raise ValueError('Forward distance must be positive')


@dataclass(frozen=True)
class Decision:
    time_s: float
    task_state: str
    stage_index: int
    stage_name: str
    watchdog_state: str
    message_age_s: float | None
    progress: float
    goal: float
    command: tuple[float, float, float] | None
    reason: str
    switch_authorized: bool = False
    fallback_authorized: bool = False


class TaskSequencer:
    def __init__(self, task_id, stages, *, message_timeout_s=.3,
                 abort_after_stale_s=2., recovery_messages=2, record_events=True):
        self.stages = tuple(stages)
        if not task_id or not self.stages:
            raise ValueError('Task ID and stages required')
        if any(not isfinite(v) or v <= 0 for v in (message_timeout_s, abort_after_stale_s)):
            raise ValueError('Watchdog intervals must be finite and positive')
        if not isinstance(recovery_messages, int) or recovery_messages < 1:
            raise ValueError('Positive recovery message count required')
        self.task_id = task_id
        self.message_timeout_s = message_timeout_s
        self.abort_after_stale_s = abort_after_stale_s
        self.recovery_messages = recovery_messages
        self.record_events = record_events
        self.events = []
        self.state = 'waiting'
        self.index = 0
        self.last_request = None
        self.last_time = None
        self.start_s = None
        self.entry = None
        self.last_command = None
        self.stale_since = None
        self.recovery_count = 0
        self.watchdog_state = 'unavailable'
        self.terminal_reason = ''

    def _event(self, time_s, event, **fields):
        if self.record_events:
            self.events.append(dict(time_s=time_s, event=event, **fields))

    def _receive(self, request, now):
        if request.task_id != self.task_id:
            reason = 'wrong_task'
        elif isinstance(request.sequence, bool) or not isinstance(request.sequence, int) or request.sequence < 0:
            reason = 'invalid_sequence'
        elif not isfinite(request.issued_s) or request.issued_s > now + 1e-9:
            reason = 'future_or_nonfinite_timestamp'
        elif now - request.issued_s > self.message_timeout_s + 1e-9:
            reason = 'expired_message'
        elif self.last_request is not None and request.sequence <= self.last_request.sequence:
            reason = 'replayed_or_out_of_order'
        elif self.last_request is not None and request.issued_s < self.last_request.issued_s - 1e-9:
            reason = 'timestamp_reversed'
        else:
            self.last_request = request
            return True
        self._event(now, 'message_rejected', reason=reason, sequence=request.sequence)
        return False

    def _enter(self, sample):
        self.entry = sample
        self.last_command = self.stages[self.index].command
        self._event(sample.time_s, 'stage_enter', stage=self.stages[self.index].name,
                    index=self.index, command=self.last_command)

    def progress(self, sample):
        if self.entry is None:
            return 0.
        stage = self.stages[self.index]
        if stage.goal_kind == 'relative_heading':
            return sample.heading_unwrapped_rad - self.entry.heading_unwrapped_rad
        dx = sample.world_x_m - self.entry.world_x_m
        dy = sample.world_y_m - self.entry.world_y_m
        return dx*cos(self.entry.heading_unwrapped_rad) + dy*sin(self.entry.heading_unwrapped_rad)

    def update(self, sample, request=None):
        now = sample.time_s
        if not all(isfinite(v) for v in (now, sample.world_x_m, sample.world_y_m, sample.heading_unwrapped_rad)):
            raise ValueError('Finite scalar robot state required')
        if now < 0 or (self.last_time is not None and now <= self.last_time):
            raise ValueError('Time must increase; create a new sequencer after reset')
        self.last_time = now
        if self.start_s is None:
            self.start_s = now
        if self.state in ('completed', 'aborted'):
            raise RuntimeError('Terminal task must be reset explicitly')
        # Detect freshness expiry before receiving: a late message cannot conceal
        # an expired lease merely because it arrives on this sample.
        expired_before = self.last_request is not None and now-self.last_request.issued_s > self.message_timeout_s+1e-9
        if expired_before and self.stale_since is None:
            self.stale_since = now
            self.recovery_count = 0
            self._event(now, 'watchdog_stale', last_sequence=self.last_request.sequence)
        abort_due = self.stale_since is not None and now-self.stale_since >= self.abort_after_stale_s-1e-9
        accepted = self._receive(request, now) if request is not None else False
        age = now-self.last_request.issued_s if self.last_request is not None else None
        fresh = age is not None and age <= self.message_timeout_s+1e-9
        if self.stale_since is not None:
            if accepted and fresh:
                self.recovery_count += 1
            elif not fresh or request is not None:
                self.recovery_count = 0
            if fresh and self.recovery_count >= self.recovery_messages:
                self._event(now, 'watchdog_recovered', stale_duration_s=now-self.stale_since)
                self.stale_since = None
                self.recovery_count = 0
        self.watchdog_state = 'fresh' if fresh and self.stale_since is None else 'recovering' if fresh else 'stale' if age is not None else 'unavailable'
        if abort_due or (self.last_request is None and now-self.start_s >= self.abort_after_stale_s-1e-9):
            self.state = 'aborted'
            self.terminal_reason = 'communication_timeout'
            self._event(now, 'task_aborted', reason=self.terminal_reason)
        elif self.state == 'waiting' and self.watchdog_state == 'fresh':
            self.state = 'running'
            self._enter(sample)
        if self.state == 'running':
            stage = self.stages[self.index]
            value = self.progress(sample)
            reached = value >= stage.goal if stage.goal > 0 else value <= stage.goal
            if reached and self.watchdog_state == 'fresh':
                self._event(now, 'stage_complete', stage=stage.name, progress=value, goal=stage.goal)
                if self.index == len(self.stages)-1:
                    self.state = 'completed'
                    self.terminal_reason = 'all_goals_reached'
                    self._event(now, 'task_completed')
                else:
                    self.index += 1
                    self._enter(sample)
            elif now-self.entry.time_s >= stage.timeout_s-1e-9:
                self.state = 'aborted'
                self.terminal_reason = 'stage_timeout'
                self._event(now, 'task_aborted', reason=self.terminal_reason, stage=stage.name)
        reason = self.terminal_reason or ('hold_last_skill_no_new_stage' if self.watchdog_state != 'fresh' else 'goal_tracking')
        return Decision(now, self.state, self.index, self.stages[self.index].name,
                        self.watchdog_state, age, self.progress(sample),
                        self.stages[self.index].goal, self.last_command, reason)
