"""Offline, causal yaw-bias diagnostics. No simulator or control access.

Body angular-z is interpolated linearly between recorded samples. Euler
heading rate is constant over each interval between unwrapped headings.
Only complete trailing windows are reported; startup entries are NaN.
"""

from math import isfinite
import numpy as np


def _series(times, values):
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if times.ndim != 1 or values.shape != times.shape or len(times) < 2:
        raise ValueError('Expected matching one-dimensional series with at least two samples')
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError('Series must be finite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Time must strictly increase')
    return times, values


def _windows(times, window_s):
    if not isfinite(window_s) or window_s <= 0:
        raise ValueError('Window must be finite and positive')
    starts = times - window_s
    valid = starts >= times[0] - 1e-10
    return np.maximum(starts, times[0]), valid


def trailing_error_stats(times, errors, window_s):
    """Exact time means/RMSE of the piecewise-linear sampled error signal."""
    times, errors = _series(times, errors)
    starts, valid = _windows(times, window_s)
    dt = np.diff(times)
    slope = np.diff(errors) / dt
    cumulative = np.r_[0., np.cumsum(dt * (errors[:-1] + errors[1:]) / 2)]
    squares = np.r_[0., np.cumsum(dt * (errors[:-1]**2 +
                                     errors[:-1]*errors[1:] + errors[1:]**2) / 3)]
    idx = np.clip(np.searchsorted(times, starts, side='right') - 1, 0, len(times) - 2)
    offset = starts - times[idx]
    left_integral = cumulative[idx] + errors[idx]*offset + slope[idx]*offset**2/2
    left_square = (squares[idx] + errors[idx]**2*offset +
                   errors[idx]*slope[idx]*offset**2 + slope[idx]**2*offset**3/3)
    mean = np.where(valid, (cumulative - left_integral)/window_s, np.nan)
    rmse = np.where(valid, np.sqrt(np.maximum(0., (squares - left_square)/window_s)), np.nan)
    return mean, rmse


def heading_from_wxyz(quaternions):
    """World ZYX Euler yaw in radians from body-to-world w,x,y,z quaternions."""
    quat = np.asarray(quaternions, dtype=float)
    if quat.ndim != 2 or quat.shape[1] != 4 or not np.all(np.isfinite(quat)):
        raise ValueError('Expected finite Nx4 wxyz quaternions')
    norm = np.linalg.norm(quat, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError('Quaternion norm must be nonzero')
    w, x, y, z = (quat / norm[:, None]).T
    return np.unwrap(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def trailing_heading_stats(times, heading, command_yaw_rps, window_s):
    """Window error of Euler-heading interval rate, distinct from body omega-z.

    Mean = (heading(t) - heading(t-W))/W - command. RMSE uses the exact
    integral of squared piecewise-constant interval rate errors. No central
    differences or future samples are used.
    """
    times, heading = _series(times, heading)
    if not isfinite(command_yaw_rps):
        raise ValueError('Command must be finite')
    starts, valid = _windows(times, window_s)
    dt = np.diff(times)
    rate_error = np.diff(heading)/dt - command_yaw_rps
    square_integral = np.r_[0., np.cumsum(rate_error**2*dt)]
    left_heading = np.interp(starts, times, heading)
    left_square = np.interp(starts, times, square_integral)
    mean = np.where(valid, (heading-left_heading)/window_s-command_yaw_rps, np.nan)
    rmse = np.where(valid, np.sqrt(np.maximum(0., (square_integral-left_square)/window_s)), np.nan)
    return mean, rmse


def bias_events(times, signed_errors, threshold_rps=.3, dwell_s=.2, eligible_after_s=0.):
    """Require a sustained *same-sign* bias; NaN marks unavailable windows.

    Events are exploratory candidates, never safety or recovery decisions.
    Eligibility resets dwell; a sign reversal clears and restarts it.
    """
    times, values = np.asarray(times, dtype=float), np.asarray(signed_errors, dtype=float)
    if times.ndim != 1 or values.shape != times.shape or len(times) < 2:
        raise ValueError('Expected matching series')
    if (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
            or np.any(np.isinf(values))):
        raise ValueError('Invalid times or infinite error')
    if any(not isfinite(x) or x <= 0 for x in (threshold_rps, dwell_s)):
        raise ValueError('Threshold and dwell must be finite and positive')
    if not isfinite(eligible_after_s):
        raise ValueError('Eligibility time must be finite')
    events, pending, sign, active = [], None, 0, False
    for t, value in zip(times, values):
        eligible = t >= eligible_after_s - 1e-10 and isfinite(value) and abs(value) > threshold_rps
        new_sign = int(np.sign(value)) if eligible else 0
        if new_sign != sign:
            if active:
                events.append(dict(event='candidate_clear', time_s=float(t), direction=sign))
            pending, sign, active = (float(t) if eligible else None), new_sign, False
        if eligible and not active and t-pending >= dwell_s-1e-10:
            active = True
            events.append(dict(event='candidate_enter', time_s=float(t), onset_time_s=pending,
                               direction=sign, signed_mean_error_rps=float(value),
                               threshold_rps=threshold_rps))
    return events


def exceedance_runs(times, errors, threshold_rps=.3, start_s=0.):
    """Sample-domain contiguous |error| runs using the legacy reset condition."""
    times, errors = _series(times, errors)
    mask = (times >= start_s - 1e-10) & (np.abs(errors) > threshold_rps)
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    stops = np.flatnonzero(np.diff(padded.astype(int)) == -1) - 1
    durations = times[stops] - times[starts]
    return dict(run_count=len(starts),
                longest_sampled_exceedance_s=float(np.max(durations)) if len(starts) else 0.,
                above_threshold_sample_fraction=float(np.mean(mask[times >= start_s-1e-10])),
                runs=[dict(start_s=float(times[a]), last_exceeding_sample_s=float(times[b]),
                           sampled_duration_s=float(d)) for a, b, d in zip(starts, stops, durations)])
