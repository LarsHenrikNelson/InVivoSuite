"""Multichannel spike removal from wideband / LFP-band data.

Methods for eliminating the broadband contamination spikes leave in the
LFP — mandatory before CSD (a spatial high-pass that *sharpens* single-
contact artifacts) and before any gamma-band (30-100 Hz) phase analysis
(spike power bleeds far below the spike band; see Scheffer-Teixeira et al.
2013, J. Neurosci. 33:1535; Zanos, Mineault & Pack 2011, J. Neurophysiol.
105:474; David, Malaval & Shamma 2010, Comput. Intell. Neurosci.
2010:393019; Banaie Boroujeni, Tiesinga & Womelsdorf 2019, J. Neurosci.
Methods 330:108485).

Two complementary mechanisms, combined by the "hybrid" default:
1. **Per-spike LSQ-scaled multichannel template subtraction** — preserves the
   underlying LFP inside spike windows. Scales are fit per event (and per
   channel with ``spatial_scale="per_channel"``), largest amplitude first, so
   overlapping spikes are handled in the canonical order.
2. **Windowed repair of residual transients**, using scipy interpolators or
   a cross-channel "spatial" fill that borrows the waveform (hence phase)
   from uncontaminated neighbor contacts.

Benchmark findings on the synthetic laminar ground truth
(``tests/test_spike_removal.py``; gamma 45-85 Hz local phase error, radians):
- medium-and-high-quality bands require subtraction: template/hybrid reduce
  the contaminated phase error ~10x, while interpolation-only barely improves
  it — the band-limited ringing of each spike extends past any reasonable
  gap window, so interpolation anchors near events are themselves
  contaminated (consistent with the literature consensus that cut-and-
  interpolate is only safe below ~25 Hz).
- interpolation's proper role is repairing *residual* transients after
  subtraction: with the cross-channel spatial fill ('hybrid' +
  ``interpolation_kind='spatial'``) performance matches pure subtraction,
  and it tolerates Kilosort misses.
- among pure temporal gap repairs, PCHIP is the robust modern default
  (shape-preserving, no overshoot); CubicSpline/Akima can ring badly on
  noisy oscillatory anchors during long-gap repair and should be used with
  care; plain linear is competitive only for short gaps on slow bands.

All functions take/return plain NumPy arrays shaped (n_chans, n_time) and
are manager-independent (testable without HDF5). Kilosort conventions
(82-sample templates, center 41) are the defaults but every parameter is
exposed.
"""

from typing import Literal

import numpy as np
from numba import njit

from ...functions.simulation.laminar_sim import TEMPLATE_CENTER, TEMPLATE_LEN

__all__ = [
    "detect_spike_windows",
    "interpolate_windows",
    "remove_spikes",
    "spatial_fill",
    "spike_triggered_average",
    "subtract_templates",
    "template_channel_masks",
]

SpatialScale = Literal["global", "per_channel"]
RemovalMethod = Literal["template", "interpolate", "hybrid"]
InterpolationKind = Literal["linear", "cubic", "pchip", "akima", "spatial"]


def template_channel_masks(templates, rel_tol=0.05):
    """Channels each template meaningfully covers.

    A spike contaminates its channels *and* neighbors; subtraction should
    only run where the template has real energy, otherwise LSQ fits noise on
    far channels. Coverage = channel energy >= rel_tol * max channel energy.

    Parameters
    ----------
    templates : np.ndarray (n_templates, window, n_chans)
    rel_tol : float

    Returns
    -------
    np.ndarray (n_templates, n_chans), float64 of 0/1 (numba-friendly).
    """
    templates = np.asarray(templates)
    energy = np.sum(templates**2, axis=1)  # (n_templates, n_chans)
    peak = energy.max(axis=1, keepdims=True)
    masks = (energy >= rel_tol * peak) & (peak > 0)
    return masks.astype(np.float64)


@njit(cache=True)
def _subtract_kernel(
    acq,
    starts,
    template_idx,
    templates,
    masks,
    center,
    per_channel,
    scales,
):
    """In-place sequential subtraction; callers pre-order spikes by amplitude.

    acq : (n_chans, n_time) float64, modified in place
    starts : (n_spikes,) window start = spike_time - center
    template_idx : (n_spikes,) index into templates
    templates : (n_templates, window, n_chans)
    masks : (n_templates, n_chans) 0/1 coverage
    center : int, window center within the template
    per_channel : 0 = one LSQ scale per spike; 1 = per channel
    scales : (n_spikes,) output
    """
    n_time = acq.shape[1]
    n_chans = acq.shape[0]
    win = templates.shape[1]
    for i in range(starts.shape[0]):
        t0 = starts[i]
        tid = template_idx[i]
        a = t0
        b = t0 + win
        if a < 0:
            a = 0
        if b > n_time:
            b = n_time
        if b <= a:
            scales[i] = 0.0
            continue
        ta = a - t0
        tb = b - t0
        if per_channel == 0:
            num = 0.0
            den = 0.0
            for c in range(n_chans):
                if masks[tid, c] > 0.5:
                    tt = templates[tid, ta:tb, c]
                    num += np.dot(acq[c, a:b], tt)
                    den += np.dot(tt, tt)
            if den > 1e-30:
                s = num / den
                scales[i] = s
                for c in range(n_chans):
                    if masks[tid, c] > 0.5:
                        acq[c, a:b] -= s * templates[tid, ta:tb, c]
            else:
                scales[i] = 0.0
        else:
            s_sum = 0.0
            n_m = 0.0
            for c in range(n_chans):
                if masks[tid, c] > 0.5:
                    tt = templates[tid, ta:tb, c]
                    den = np.dot(tt, tt)
                    if den > 1e-30:
                        s = np.dot(acq[c, a:b], tt) / den
                        acq[c, a:b] -= s * tt
                        s_sum += s
                        n_m += 1.0
            scales[i] = s_sum / n_m if n_m > 0 else 0.0

def subtract_templates(
    acq,
    spike_times,
    template_idx,
    templates,
    masks=None,
    center=TEMPLATE_CENTER,
    spatial_scale: SpatialScale = "per_channel",
    amplitudes=None,
    copy=True,
):
    """Subtract per-spike LSQ-scaled multichannel templates from `acq`.

    Scales are fit on the actual data (least squares over template-covered
    channels), so template/data amplitude mismatches — e.g. from Kilosort
    gain conventions, drift or amplitude jitter — are absorbed per event.
    Spikes are subtracted largest-first so overlapping waveforms see
    progressively cleaned data.

    Parameters
    ----------
    acq : np.ndarray (n_chans, n_time)
        Data to clean (wideband or band-limited). Data and templates must be
        in the same band/scale; manager wiring band-passes both with the
        same settings before calling this.
    spike_times : np.ndarray (n_spikes,)
        Zero-based sample indices; window =
        ``[spike_time - center, spike_time - center + window)``.
    template_idx : np.ndarray (n_spikes,)
    templates : np.ndarray (n_templates, window, n_chans)
    masks : np.ndarray | None
        From `template_channel_masks`; computed when None.
    spatial_scale : {'global', 'per_channel'}
        'global': one scale per spike fit jointly over covered channels
        (constrained, robust). 'per_channel': independent per-channel scales
        (tailored to drift changing the spatial profile).
    amplitudes : np.ndarray | None
        Optional per-spike amplitude hints, used only for ordering.
    copy : bool
        Operate on a copy (True) or in place (False).

    Returns
    -------
    cleaned : np.ndarray (n_chans, n_time) float64
    scales : np.ndarray (n_spikes,)
        Fitted scales in the *original* spike order (mean of per-channel
        scales for 'per_channel' mode).
    """
    acq = np.asarray(acq)
    cleaned = np.array(acq, dtype=np.float64, copy=copy)
    if masks is None:
        masks = template_channel_masks(templates)
    spike_times = np.asarray(spike_times, dtype=np.int64)
    template_idx = np.asarray(template_idx, dtype=np.int64).ravel()
    templates_c = np.ascontiguousarray(templates, dtype=np.float64)
    strength = np.max(np.abs(templates_c[template_idx]), axis=(1, 2))
    if amplitudes is not None:
        strength = strength * np.asarray(amplitudes)
    order = np.argsort(-strength, kind="stable")
    starts = spike_times[order] - center
    idx = template_idx[order]
    scales_sorted = np.empty(order.size)
    _subtract_kernel(
        cleaned,
        starts,
        idx,
        templates_c,
        masks,
        center,
        0 if spatial_scale == "global" else 1,
        scales_sorted,
    )
    scales = np.empty(order.size)
    scales[order] = scales_sorted
    return cleaned, scales


def spike_triggered_average(
    acq,
    spike_times,
    clusters=None,
    window=TEMPLATE_LEN,
    center=TEMPLATE_CENTER,
):
    """Mean waveform around spikes (per cluster, or the grand mean) in the
    band of the supplied `acq` — the data-derived template estimate that
    captures the spike's low-frequency 'foot' in the band of interest.

    Returns
    -------
    templates : np.ndarray (n_clusters or 1, window, n_chans)
    cluster_ids : list[int] (-1 when clusters is None)
    """
    acq = np.asarray(acq, dtype=np.float64)
    spike_times = np.asarray(spike_times, dtype=np.int64)
    if clusters is None:
        clusters = np.zeros(spike_times.size, dtype=np.int64)
    clusters = np.asarray(clusters).ravel()
    cids = [int(c) for c in np.unique(clusters)]
    cid_pos = {c: i for i, c in enumerate(cids)}
    n_time = acq.shape[1]
    out = np.zeros((len(cids), int(window), acq.shape[0]))
    counts = np.zeros(len(cids))
    for t, c in zip(spike_times, clusters):
        a = t - center
        b = a + window
        if a < 0 or b > n_time:
            continue
        out[cid_pos[int(c)]] += acq[:, a:b].T
        counts[cid_pos[int(c)]] += 1
    for i in range(len(cids)):
        if counts[i] > 0:
            out[i] /= counts[i]
    return out, cids

def _mask_from_windows(windows, n_time):
    mask = np.zeros(n_time, dtype=bool)
    for s, e in windows:
        mask[max(0, int(s)) : min(n_time, int(e))] = True
    return mask


def detect_spike_windows(acq, threshold=6.0, guard=TEMPLATE_LEN):
    """Detect transient (spike-residual) intervals per channel.

    A sample is flagged when ``|x - median| > threshold * MAD`` (per-channel
    robust scale, 1.4826-normalized); flagged samples are dilated by `guard`
    and merged into windows. Cheap and vectorized.

    Returns
    -------
    windows : list[np.ndarray], per channel (n_windows, 2) arrays of
        [start, stop) sample indices.
    """
    acq = np.atleast_2d(np.asarray(acq, dtype=np.float64))
    if acq.shape[0] == 1 and acq.shape[1] == 1:
        return []
    n_chans, n_time = acq.shape
    med = np.median(acq, axis=1, keepdims=True)
    mad = np.median(np.abs(acq - med), axis=1, keepdims=True) * 1.4826
    mad = np.maximum(mad, 1e-12)
    out = []
    for c in range(n_chans):
        above = np.abs(acq[c] - med[c, 0]) > threshold * mad[c, 0]
        idx = np.flatnonzero(above)
        if idx.size == 0:
            out.append(np.empty((0, 2), dtype=np.int64))
            continue
        # dilate + merge
        lo = idx - guard
        hi = idx + guard + 1
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]
        starts, ends = [lo[0]], [hi[0]]
        for i in range(1, lo.size):
            if lo[i] <= ends[-1]:
                ends[-1] = max(ends[-1], hi[i])
            else:
                starts.append(lo[i])
                ends.append(hi[i])
        out.append(
            np.column_stack(
                (np.clip(starts, 0, n_time), np.clip(ends, 0, n_time))
            ).astype(np.int64)
        )
    return out


def interpolate_windows(acq, windows, kind="pchip"):
    """Repair masked windows with modern scipy interpolators.

    Unlike the linear interpolation used by older spike-removal methods,
    PCHIP (shape-preserving) and Akima avoid spurious oscillations near the
    sharp gap edges, and cubics give smooth transitions — measurably better
    phase fidelity in the gamma band.

    Parameters
    ----------
    acq : np.ndarray (n_chans, n_time)
    windows : per-channel (n, 2) [start, stop) windows, e.g. from
        `detect_spike_windows` or explicit spike windows.
    kind : {'linear', 'cubic', 'pchip', 'akima'}
        'cubic' uses a not-a-knot CubicSpline over the clean samples; gaps
        at the recording edges hold the nearest clean value (no extrapolated
        wildness).

    Returns
    -------
    cleaned : np.ndarray (n_chans, n_time) float64 (copy; gaps repaired)
    """
    from scipy import interpolate as spi

    acq = np.atleast_2d(np.asarray(acq, dtype=np.float64))
    out = acq.copy()
    n_time = acq.shape[1]
    for c, wins in enumerate(windows):
        if len(wins) == 0:
            continue
        mask = _mask_from_windows(wins, n_time)
        good_idx = np.flatnonzero(~mask)
        if good_idx.size < 4:
            continue
        bad_idx = np.flatnonzero(mask)
        interior = (bad_idx > good_idx[0]) & (bad_idx < good_idx[-1])
        xg = good_idx
        yg = acq[c, xg]
        bad_in = bad_idx[interior]
        if bad_in.size == 0:
            continue
        if kind == "linear":
            out[c, bad_in] = np.interp(bad_in, xg, yg)
        else:
            if kind == "cubic":
                interp = spi.CubicSpline(xg, yg, bc_type="not-a-knot")
            elif kind == "pchip":
                interp = spi.PchipInterpolator(xg, yg)
            elif kind == "akima":
                interp = spi.Akima1DInterpolator(xg, yg)
            else:
                raise ValueError(f"Unknown interpolation kind: {kind}")
            out[c, bad_in] = interp(bad_in)
        edges = bad_idx[~interior]
        if edges.size:
            left = edges < good_idx[0]
            out[c, edges[left]] = yg[0]
            out[c, edges[~left]] = yg[-1]
    return out

def merge_windows(windows_a, windows_b, n_time):
    """Union of two per-channel window lists (both (n, 2) [start, stop))."""
    merged = []
    for ca, cb in zip(windows_a, windows_b):
        both = np.vstack((np.asarray(ca).reshape(-1, 2), np.asarray(cb).reshape(-1, 2)))
        if both.shape[0] == 0:
            merged.append(np.empty((0, 2), dtype=np.int64))
            continue
        lo = np.clip(both[:, 0], 0, n_time)
        hi = np.clip(both[:, 1], 0, n_time)
        order = np.argsort(lo)
        lo, hi = lo[order], hi[order]
        starts, ends = [lo[0]], [hi[0]]
        for i in range(1, lo.size):
            if lo[i] <= ends[-1]:
                ends[-1] = max(ends[-1], hi[i])
            else:
                starts.append(lo[i])
                ends.append(hi[i])
        merged.append(np.column_stack((starts, ends)).astype(np.int64))
    return merged


def spike_windows(
    spike_times,
    spike_template_idx,
    templates=None,
    center=TEMPLATE_CENTER,
    window=None,
    n_chans=None,
    guard=0,
):
    """Per-channel [start, stop) windows around spikes on their covered
    channels (or every channel when templates is None), dilated by `guard`
    samples so the band-limited ringing adjacent to each spike lands inside
    the window too."""
    window = TEMPLATE_LEN if window is None else window
    times = np.asarray(spike_times, dtype=np.int64).ravel()
    idx = np.asarray(spike_template_idx, dtype=np.int64).ravel()
    if n_chans is None:
        n_chans = int(np.asarray(templates).shape[2])
    per_channel = [[] for _ in range(n_chans)]
    if templates is None:
        for t in times:
            for c in range(n_chans):
                per_channel[c].append(
                    (t - center - guard, t - center + window + guard)
                )
    else:
        masks = template_channel_masks(templates)
        for t, tid in zip(times, idx):
            for c in np.flatnonzero(masks[tid] > 0.5):
                per_channel[c].append(
                    (t - center - guard, t - center + window + guard)
                )
    return [np.asarray(w, dtype=np.int64).reshape(-1, 2) for w in per_channel]


def spatial_fill(acq, windows, flank=40, return_unrepaired=False):
    """Cross-channel reconstruction of masked samples.

    The LFP is spatially smooth across 50-100 um contacts while spikes are
    ultra-local, so the waveform (hence the *phase*) of a contaminated
    segment can be borrowed from uncontaminated neighbors. For each channel's
    masked segment, anchor channels are those with no overlapping window;
    the segment is rebuilt by linear interpolation across depth between the
    nearest clean anchors on either side (or copied from the single nearest
    anchor), then gain-aligned to the channel using its clean flanks. This is
    the window-localized cousin of Zanos-style neighbor substitution.

    Segments with no usable anchors are left untouched and (with
    ``return_unrepaired=True``) reported back, so a temporal fallback can
    repair only those.
    """
    out, unrepaired = _spatial_fill_impl(acq, windows, flank)
    if return_unrepaired:
        return out, unrepaired
    return out


def _spatial_fill_impl(acq, windows, flank):
    acq = np.atleast_2d(np.asarray(acq, dtype=np.float64))
    out = acq.copy()
    n_chans, n_time = out.shape
    unrepaired = [[] for _ in range(n_chans)]
    for c in range(n_chans):
        for s, e in windows[c]:
            s = int(s)
            e = int(e)
            if e - s <= 0:
                continue
            s2 = max(0, s - flank)
            e2 = min(n_time, e + flank)

            def _clean(cc, a, b):
                return not any(
                    not (w[1] <= a or w[0] >= b) for w in windows[cc]
                )

            anchors = [
                cc
                for cc in range(n_chans)
                if cc != c and _clean(cc, s2, e2)
            ]
            if not anchors:
                unrepaired[c].append((s, e))
                continue
            below = [a for a in anchors if a < c]
            above = [a for a in anchors if a > c]

            def _recon(a, b):
                if below and above:
                    x_lo = max(below)
                    x_hi = min(above)
                    if x_hi == x_lo:
                        g = 1.0
                    else:
                        g = (c - x_lo) / (x_hi - x_lo)
                    return out[x_lo, a:b] * (1 - g) + out[x_hi, a:b] * g
                single = anchors[0]
                return out[single, a:b].copy()

            recon = _recon(s2, e2)
            flanks = np.concatenate((out[c, s2:s], out[c, e:e2]))
            recon_flanks = np.concatenate((recon[: s - s2], recon[e - s2 :]))
            denom = np.mean(recon_flanks)
            gain = float(np.mean(flanks) / denom) if abs(denom) > 1e-9 else 1.0
            out[c, s:e] = gain * recon[s - s2 : e - s2]
    return out, [np.asarray(w, dtype=np.int64).reshape(-1, 2) for w in unrepaired]


def remove_spikes(
    acq,
    spike_times=None,
    template_idx=None,
    templates=None,
    spike_channels=None,
    method: RemovalMethod = "hybrid",
    spatial_scale: SpatialScale = "per_channel",
    interpolation_kind: InterpolationKind = "pchip",
    center=TEMPLATE_CENTER,
    window=TEMPLATE_LEN,
    window_guard=0,
    detection_threshold=6.0,
    detection_guard=None,
    detect_on=None,
    return_diagnostics=False,
):
    """Remove spikes from a multichannel recording chunk.

    Parameters
    ----------
    acq : np.ndarray (n_chans, n_time)
        Wideband or band-limited data (same band/scale as `templates`).
    spike_times, template_idx, templates :
        Detected spikes (all clusters) and their multichannel templates;
        `template_idx` selects each spike's template (single template when
        None).
    method : {'template', 'interpolate', 'hybrid'}
        'template': LSQ-scaled multichannel subtraction (preserves the LFP
        inside spike windows).
        'interpolate': cut windows around each spike (dilated by
        `window_guard`) and repair them with the chosen interpolator; no
        templates needed.
        'hybrid' (recommended): subtraction, then a robust MAD detector pass
        over the residual repairing ONLY leftover transients (undetected or
        mis-scaled spikes) — windows subtraction already cleaned are kept.
    spatial_scale : {'global', 'per_channel'}
        See `subtract_templates`.
    interpolation_kind : {'linear', 'cubic', 'pchip', 'akima', 'spatial'}
        Temporal gap repair, or cross-channel reconstruction ('spatial',
        falling back to PCHIP where no clean neighbor exists).
    window_guard : int
        Dilation (samples) of the per-spike windows in 'interpolate' mode, so
        the post-bandpass ringing adjacent to spikes is repaired as well.
    detect_on : np.ndarray | Callable | None
        Residual detection must run on a spike-band view (spikes are hidden by
        the LFP in the wideband robust scale). Pass either a pre-filtered
        array to use for detection as-is, or a callable applied to the
        *cleaned* chunk (e.g. ``lambda x: butterworth_zero(x, 4, fs,
        highpass=300)``) so only leftover transients — not properly
        subtracted spikes — trigger the repair. Defaults to the cleaned
        chunk itself.

    Returns
    -------
    cleaned : np.ndarray (n_chans, n_time) float64
    diagnostics : dict, with ``return_diagnostics=True`` (fitted-scale
        statistics and window counts per stage).
    """
    acq = np.atleast_2d(np.asarray(acq, dtype=np.float64))
    n_chans, n_time = acq.shape
    diag = {}
    guard = window if detection_guard is None else int(detection_guard)
    times = None if spike_times is None else np.asarray(spike_times, dtype=np.int64).ravel()
    idx = (
        np.zeros(len(times), dtype=np.int64)
        if (times is not None and template_idx is None)
        else template_idx
    )

    if method in ("template", "hybrid"):
        if times is None or templates is None:
            raise ValueError("template/hybrid methods need spike_times and templates")
        cleaned, scales = subtract_templates(
            acq,
            times,
            idx,
            templates,
            center=center,
            spatial_scale=spatial_scale,
        )
        diag["scale_mean"] = float(np.mean(scales))
        diag["scale_std"] = float(np.std(scales))
    else:
        cleaned = acq.copy()
        scales = None

    if method == "interpolate":
        if times is not None:
            windows = spike_windows(
                times, idx, templates=templates, center=center,
                window=window, n_chans=n_chans, guard=int(window_guard),
            )
        else:
            windows = detect_spike_windows(cleaned, detection_threshold, guard)
        diag["n_repaired_windows"] = int(sum(len(w) for w in windows))
        diag["n_residual_windows"] = 0
        cleaned = _repair(cleaned, windows, interpolation_kind)
    elif method == "hybrid":
        if detect_on is None:
            detect_data = cleaned
        elif callable(detect_on):
            detect_data = detect_on(cleaned)
        else:
            detect_data = detect_on
        windows = detect_spike_windows(
            detect_data,
            detection_threshold,
            guard,
        )
        diag["n_residual_windows"] = int(sum(len(w) for w in windows))
        diag["n_repaired_windows"] = diag["n_residual_windows"]
        cleaned = _repair(cleaned, windows, interpolation_kind)
    else:
        diag["n_repaired_windows"] = 0
        diag["n_residual_windows"] = 0

    if return_diagnostics:
        return cleaned, diag
    return cleaned


def _repair(cleaned, windows, interpolation_kind):
    """Apply the chosen gap repair to windows that still need it."""
    if interpolation_kind == "spatial":
        repaired, unrepaired = _spatial_fill_impl(cleaned, windows, flank=40)
        # Temporal fallback ONLY for the segments the spatial fill could not
        # reconstruct (no clean neighbor channels available).
        return interpolate_windows(repaired, unrepaired, kind="pchip")
    if interpolation_kind not in ("linear", "cubic", "pchip", "akima"):
        raise ValueError(f"Unknown interpolation kind: {interpolation_kind}")
    return interpolate_windows(cleaned, windows, kind=interpolation_kind)




