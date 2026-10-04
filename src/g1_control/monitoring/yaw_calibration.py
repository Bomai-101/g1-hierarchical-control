"""Labeled signal-level calibration; not physical robot validation."""

import numpy as np
from g1_control.monitoring.yaw_window import bias_events, trailing_error_stats

FAMILIES = ('steady_tracking', 'slow_zero_dc', 'slow_command_tracking',
            'command_steps', 'short_pulse', 'startup_only',
            'negative_bias', 'positive_bias')


def make_case(seed, family, duration_s=40., dt=.02):
    """Generate commands and measured body-z with known injected DC labels.

    Nominal tracking is a first-order signal model, not a G1 dynamics model.
    Zero injected DC does not imply every finite window has zero mean.
    """
    if family not in FAMILIES:
        raise ValueError('Unknown calibration family')
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family)]))
    t = np.arange(0., duration_s + dt/2, dt)
    phase = float(rng.uniform(0, 2*np.pi))
    fast_hz = float(rng.uniform(2.5, 4.5))
    amplitude = float(rng.uniform(.3, .8))
    fast = amplitude*np.sin(2*np.pi*fast_hz*t+phase)
    command = np.zeros_like(t)
    residual = fast.copy()
    onset, direction = None, 0
    params = dict(fast_hz=fast_hz, fast_amplitude_rps=amplitude, phase_rad=phase)
    if family == 'slow_zero_dc':
        frequency = float(rng.uniform(.08, .65))
        slow_amplitude = float(rng.uniform(.35, .85))
        residual += slow_amplitude*np.sin(2*np.pi*frequency*t+phase)
        params.update(slow_hz=frequency, slow_amplitude_rps=slow_amplitude)
    elif family in ('slow_command_tracking', 'command_steps'):
        if family == 'slow_command_tracking':
            frequency = float(rng.uniform(.03, .12))
            command = .2*np.sin(2*np.pi*frequency*t)
            params['command_hz'] = frequency
        else:
            command = np.select([t >= 30, t >= 20, t >= 10], [.2, -.2, .2], default=0.)
        tau = float(rng.uniform(.2, 1.))
        measured_nominal = np.zeros_like(t)
        decay = np.exp(-dt/tau)
        for i in range(1, len(t)):
            measured_nominal[i] = decay*measured_nominal[i-1]+(1-decay)*command[i-1]
        residual += measured_nominal-command
        params['tracking_tau_s'] = tau
    elif family == 'short_pulse':
        pulse_duration = float(rng.uniform(.1, .5))
        pulse_amplitude = float(rng.uniform(.4, .9))
        residual += np.where((t >= 12) & (t < 12+pulse_duration), pulse_amplitude, 0.)
        params.update(pulse_duration_s=pulse_duration, pulse_amplitude_rps=pulse_amplitude)
    elif family == 'startup_only':
        residual += np.where(t < 1.5, .8, 0.)
    elif family in ('negative_bias', 'positive_bias'):
        onset = float(rng.choice([8., 12., 16.]))
        direction = -1 if family == 'negative_bias' else 1
        bias = float(rng.uniform(.35, .75))
        # Add slower varying error too, so positives are not pure easy offsets.
        slow_frequency = float(rng.uniform(.15, .7))
        slow_amplitude = float(rng.uniform(.1, .35))
        residual += slow_amplitude*np.sin(2*np.pi*slow_frequency*t+phase/2)
        residual += np.where(t >= onset, direction*bias, 0.)
        params.update(bias_rps=direction*bias, slow_hz=slow_frequency,
                      slow_amplitude_rps=slow_amplitude)
    return dict(seed=int(seed), family=family, times=t, command=command,
                measured_body_z=command+residual, onset_s=onset, direction=direction,
                parameters=params)


def tracking_error(measured, command):
    measured, command = np.asarray(measured, dtype=float), np.asarray(command, dtype=float)
    if measured.ndim != 1 or command.shape != measured.shape:
        raise ValueError('Commands and measured rates must align')
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(command)):
        raise ValueError('Command and measured rates must be finite')
    # Subtract the command at each sample BEFORE the window integration.
    return measured-command


def score_events(times, events, onset_s, direction, eligible_after_s):
    """Score newly entered candidates; pre-existing alarms cannot prove detection."""
    times = np.asarray(times, dtype=float)
    if onset_s is not None and (not np.isfinite(onset_s) or onset_s < eligible_after_s):
        raise ValueError('Labeled onset must follow eligibility')
    if direction not in (-1, 0, 1) or (onset_s is None) != (direction == 0):
        raise ValueError('Invalid onset/direction label')
    entries = [e for e in events if e['event']=='candidate_enter']
    prior = [e for e in entries if onset_s is None or e['time_s'] < onset_s-1e-10]
    correct = [e for e in entries if onset_s is not None and e['time_s'] >= onset_s-1e-10
               and e['direction']==direction]
    wrong = [e for e in entries if onset_s is not None and e['time_s'] >= onset_s-1e-10
             and e['direction'] != direction]
    false_end = float(times[-1]) if onset_s is None else onset_s
    active, start, false_duration = False, None, 0.
    active_at_onset = False
    if onset_s is not None:
        for event in events:
            if event['time_s'] >= onset_s-1e-10:
                break
            active_at_onset = event['event'] == 'candidate_enter'
    for event in events:
        if event['event']=='candidate_enter':
            active, start = True, event['time_s']
        elif active:
            false_duration += max(0., min(event['time_s'], false_end)-max(start, eligible_after_s))
            active = False
    if active:
        false_duration += max(0., false_end-max(start, eligible_after_s))
    first = correct[0]['time_s'] if correct else None
    return dict(negative_case=onset_s is None, false_candidate_count=len(prior),
                false_active_duration_s=false_duration,
                false_exposure_s=max(0., false_end-eligible_after_s),
                detected=first is not None,
                detection_delay_s=None if first is None else first-onset_s,
                wrong_direction_entries=len(wrong), preexisting_alarm_at_onset=active_at_onset)


def evaluate_case(case, window_s, threshold_rps, dwell_s, warmup_s=2.):
    t = case['times']
    error = tracking_error(case['measured_body_z'], case['command'])
    mean, _ = trailing_error_stats(t, error, window_s)
    eligible = warmup_s+window_s
    events = bias_events(t, mean, threshold_rps, dwell_s, eligible)
    return score_events(t, events, case['onset_s'], case['direction'], eligible)
