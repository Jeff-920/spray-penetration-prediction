"""Data processing pipeline for spray penetration curves.

Covers the full chain:
  raw samples -> curve fitting / filtering / hybrid denoising
  -> [0,1] normalisation -> time-window augmentation -> time normalisation
  -> PyTorch Dataset ready for FNO training or inference.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize as opt
import torch
from scipy.signal import savgol_filter
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Exponential penetration model  g(x) = -exp(-k*x + b) + exp(b)
# ---------------------------------------------------------------------------


def func(x: np.ndarray, k: float, b: float) -> np.ndarray:
    """Physical penetration model:  g = -exp(-k*x + b) + exp(b)."""
    return -1.0 * np.exp(-k * np.asarray(x) + b) + np.exp(b)


def datafitting(data: list[dict]) -> tuple[list[dict], list[np.ndarray]]:
    """Batch curve_fit with exponential ``func``; stores ``fit_param`` in each sample."""
    fit_params: list[np.ndarray] = []
    for i, entry in enumerate(data):
        if data[i]["time"][0] < 0 or data[i]["penetration"][0] < 0:
            data[i]["time"] = [j - data[i]["time"][0] for j in data[i]["time"]]
            data[i]["penetration"][0] = 0
        if data[i]["time"][0] > 0 and data[i]["penetration"][0] > 0:
            data[i]["time"] = [0] + data[i]["time"]
            data[i]["penetration"] = [0] + data[i]["penetration"]

        pairs = list(zip(data[i]["time"], data[i]["penetration"]))
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        t_sorted, p_sorted = zip(*pairs_sorted)
        data[i]["time"] = list(t_sorted)
        data[i]["penetration"] = list(p_sorted)

        x = np.array(data[i]["time"])
        y = np.array(data[i]["penetration"])
        fit_param, _ = opt.curve_fit(func, x, y, maxfev=2000)
        fit_params.append(fit_param)
        data[i]["fit_param"] = fit_param.tolist()
    return data, fit_params


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------


def _sanitize_and_sort_single_sample(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sort a sample by time and align curve origin near (0, 0)."""
    time_arr = np.array(sample["time"], dtype=np.float64)
    penetration_arr = np.array(sample["penetration"], dtype=np.float64)
    if time_arr.size == 0 or penetration_arr.size == 0:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    if time_arr[0] < 0 or penetration_arr[0] < 0:
        time_arr = time_arr - time_arr[0]
        penetration_arr[0] = 0.0
    if time_arr[0] > 0 and penetration_arr[0] > 0:
        time_arr = np.insert(time_arr, 0, 0.0)
        penetration_arr = np.insert(penetration_arr, 0, 0.0)
    pairs = sorted(zip(time_arr.tolist(), penetration_arr.tolist()), key=lambda x: x[0])
    t_sorted, p_sorted = zip(*pairs)
    t_arr = np.array(t_sorted, dtype=np.float64)
    p_arr = np.array(p_sorted, dtype=np.float64)
    if t_arr[0] > 0:
        t_arr = np.insert(t_arr, 0, 0.0)
        p_arr = np.insert(p_arr, 0, 0.0)
    else:
        t_arr[0] = 0.0
        p_arr[0] = 0.0
    return t_arr, p_arr


def _anchor_origin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Force denoised curves to pass through (0, 0) when the grid starts at t=0."""
    y_out = np.asarray(y, dtype=np.float64).copy()
    if y_out.size and abs(float(np.asarray(x, dtype=np.float64)[0])) <= 1e-12:
        y_out[0] = 0.0
    return y_out


def _moving_average_filter(y: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving-average filter."""
    if window <= 1:
        return y.copy()
    w = int(window)
    if w <= 1:
        return y.copy()
    left = w // 2
    right = w - 1 - left
    y_pad = np.pad(y, (left, right), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(y_pad, kernel, mode="valid")


def _apply_filter_1d(
    y: np.ndarray,
    filter_type: str = "savgol",
    window: int = 15,
    polyorder: int = 1,
) -> np.ndarray:
    """Apply 1D smoothing filter (savgol / moving_avg / linear) on penetration curve."""
    n = int(y.size)
    if n <= 2:
        return y.copy()
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y.copy()

    if filter_type in ("linear", "interp", "none"):
        return y.copy()
    if filter_type == "moving_avg":
        return _moving_average_filter(y, w)
    # savgol
    p = max(0, int(polyorder))
    p = min(p, w - 1)
    if p >= w:
        p = w - 1
    return savgol_filter(y, window_length=w, polyorder=p, mode="interp")


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    """Smooth Hermite blend on [0, 1] -> [0, 1]."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grid_dt(x: np.ndarray) -> float:
    """Median time-step from a 1-D grid."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 1.0
    return float(np.median(np.diff(x)))


def _effective_blend_ms(x: np.ndarray, blend_ms: float, min_grid_points: int = 4) -> float:
    return max(float(blend_ms), int(min_grid_points) * _grid_dt(x))


# ---------------------------------------------------------------------------
# Plateau detection & flattening (evaporative-spray tail processing)
# ---------------------------------------------------------------------------


def _detect_plateau_start(
    x: np.ndarray,
    y: np.ndarray,
    slope_eps: float = 0.15,
    min_duration_ms: float = 0.15,
) -> float | None:
    """Detect where a filtered curve enters a near-flat evaporative plateau."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 8:
        return None

    dt = float(np.median(np.diff(x)))
    if dt <= 0:
        return None

    min_pts = max(3, int(np.ceil(float(min_duration_ms) / dt)))
    if n < min_pts + 3:
        return None

    y_max = float(np.max(y))
    amp = float(np.ptp(y))
    if amp < 1e-6:
        return None

    min_tail_ptp = max(0.5, 0.01 * amp)
    start_search = max(1, int(0.08 * n))

    for i in range(start_search, n - min_pts + 1):
        if float(y[i]) < 0.85 * y_max:
            continue
        seg_x = x[i:] - x[i]
        seg_y = y[i:]
        duration = max(float(seg_x[-1]), 1e-9)
        net_slope = (float(seg_y[-1]) - float(seg_y[0])) / duration
        if abs(net_slope) > float(slope_eps):
            continue
        if float(np.ptp(seg_y)) < min_tail_ptp:
            continue
        return float(x[i])
    return None


def _flatten_plateau_median(
    x: np.ndarray,
    y: np.ndarray,
    t_start: float,
    blend_ms: float = 0.12,
) -> np.ndarray:
    """Replace the plateau segment with its median level and smoothstep-blend."""
    x = np.asarray(x, dtype=np.float64)
    y_out = np.asarray(y, dtype=np.float64).copy()
    plateau_mask = x >= float(t_start)
    if not np.any(plateau_mask):
        return y_out

    plateau_val = float(np.median(y_out[plateau_mask]))
    y_out[plateau_mask] = plateau_val

    blend_ms = _effective_blend_ms(x, blend_ms)
    blend_left = float(t_start) - blend_ms
    blend_mask = (x >= blend_left) & (x < float(t_start))
    if np.any(blend_mask):
        alpha = _smoothstep01((x[blend_mask] - blend_left) / blend_ms)
        y_out[blend_mask] = (1.0 - alpha) * y_out[blend_mask] + alpha * plateau_val
    return y_out


def _smooth_knee_region(
    x: np.ndarray,
    y: np.ndarray,
    t_knee: float,
    knee_smooth_ms: float = 0.18,
) -> np.ndarray:
    """Locally smooth the rising-to-plateau inflection with edge feathering."""
    x = np.asarray(x, dtype=np.float64)
    y_out = np.asarray(y, dtype=np.float64).copy()
    dt = _grid_dt(x)
    width = max(float(knee_smooth_ms), 3.0 * dt)
    left = float(t_knee) - width
    center = float(t_knee)
    region = (x >= left) & (x <= center)
    n_region = int(np.count_nonzero(region))
    if n_region < 5:
        return y_out

    w = min(9, max(3, (n_region // 3) | 1))
    y_ma = _apply_filter_1d(y_out, filter_type="moving_avg", window=w, polyorder=0)
    for i in np.where(region)[0]:
        xi = float(x[i])
        mix = 0.85 * float(_smoothstep01(np.array([(xi - left) / max(center - left, 1e-12)]))[0])
        y_out[i] = (1.0 - mix) * y_out[i] + mix * float(y_ma[i])
    return y_out


# ---------------------------------------------------------------------------
# Per-sample rule helpers (filter_effect_rules.json / sim_filter_effect_rules.json)
# ---------------------------------------------------------------------------


def _norm_rule_key(path_like: str | Path) -> str:
    s = str(path_like).replace("\\", "/").strip().lower()
    if ":/" in s:
        s = s.split(":/", 1)[1]
    while s.startswith("/"):
        s = s[1:]
    return s


def _load_rule_map(rule_json_path: str = "") -> dict[str, dict]:
    if not rule_json_path:
        return {}
    p = Path(rule_json_path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for r in payload.get("rules", []):
        fp = r.get("sample_file")
        if not fp:
            continue
        out[_norm_rule_key(fp)] = r
    return out


def load_merged_rule_map(
    rule_json_path: str = "",
    sim_rule_json_path: str = "",
) -> dict[str, dict]:
    """Experimental rules first; simulation rules override on duplicate keys."""
    merged = _load_rule_map(rule_json_path)
    sim_map = _load_rule_map(sim_rule_json_path)
    if sim_map:
        merged.update(sim_map)
    return merged


def _resolve_max_time_from_rule(rule: dict, default_max_time: float = 3.0) -> float:
    if rule.get("max_time_ms") is not None:
        return float(rule["max_time_ms"])
    return float(default_max_time)


def _infer_sim_rule_mode(rule: dict, default: str = "fit") -> str:
    mode = str(rule.get("mode", "")).lower().strip()
    if mode:
        return mode
    if rule.get("plateau_force") or rule.get("plateau_t_start_ms") is not None:
        return "filter"
    if rule.get("max_time_ms") is not None:
        return "fit"
    return str(default).lower().strip()


def _trim_tail_by_rule(
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    drop_last_points: int = 0,
    tail_policy: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    x = x_sorted.copy()
    y = y_sorted.copy()
    n_drop = max(0, int(drop_last_points))
    if n_drop > 0 and len(x) - n_drop >= 3:
        x = x[:-n_drop]
        y = y[:-n_drop]

    if str(tail_policy).lower().strip() == "auto_drop_abnormal_tail":
        for _ in range(3):
            if len(y) < 6:
                break
            dy = np.diff(y)
            last_drop = float(dy[-1])
            if last_drop >= 0:
                break
            baseline = float(np.median(np.abs(dy[:-1]))) if len(dy) > 1 else 0.0
            amp = float(np.ptp(y))
            threshold = max(0.5, 4.0 * baseline, 0.03 * amp)
            if abs(last_drop) <= threshold:
                break
            x = x[:-1]
            y = y[:-1]
        for _ in range(3):
            if len(y) < 8:
                break
            y_prev = y[:-1]
            ref = float(np.median(y_prev[-5:])) if len(y_prev) >= 5 else float(np.median(y_prev))
            amp = float(np.ptp(y))
            drop = ref - float(y[-1])
            if drop <= max(0.8, 0.04 * amp):
                break
            x = x[:-1]
            y = y[:-1]
    return x, y


def _strip_sim_augmented_prefix(file_path: str) -> str:
    fp = str(file_path).replace("\\", "/")
    if fp.startswith("sim_augmented::"):
        return fp.split("sim_augmented::", 1)[1]
    return fp


def _is_sim_sample(sample: dict) -> bool:
    fp = str(sample.get("file_path", "")).replace("\\", "/")
    if fp.startswith("sim_augmented::"):
        return True
    return "random_penetration/" in fp.lower()



def _sim_file_stem(sample: dict) -> int | None:
    fp = _strip_sim_augmented_prefix(str(sample.get("file_path", "")))
    if not fp:
        return None
    try:
        return int(Path(fp).stem)
    except ValueError:
        return None


def _sim_default_mode(sample: dict) -> str:
    """random_penetration: stem < 200 -> filter; stem >= 200 -> fit."""
    stem = _sim_file_stem(sample)
    if stem is not None and stem >= 200:
        return "fit"
    return "filter"


def _resolve_sample_rule(
    sample: dict,
    sample_rule_map: dict[str, dict] | None = None,
) -> dict:
    if not sample_rule_map:
        return {}
    fp = _strip_sim_augmented_prefix(sample.get("file_path", ""))
    if not fp:
        return {}
    rk = _norm_rule_key(fp)
    r = sample_rule_map.get(rk, {})
    if r:
        return r
    parts = rk.split("/")
    if len(parts) >= 3:
        r = sample_rule_map.get("/".join(parts[-3:]), {})
        if r:
            return r
    # fallback: match by filename only (supports relocated test data)
    filename = parts[-1] if parts else ""
    if filename:
        return sample_rule_map.get(filename, {})
    return {}


def _resolve_denoise_mode(rule: dict, denoise_method: str, rule_fallback: str = "filter") -> str:
    method = str(denoise_method).lower().strip()
    if method == "rule":
        return str(rule.get("mode", rule_fallback)).lower().strip()
    return method


def _resolve_sample_denoise_plan(
    sample: dict,
    rule: dict,
    denoise_method: str,
    *,
    rule_fallback: str = "filter",
    default_tail_policy: str = "auto_drop_abnormal_tail",
    sim_denoise_method: str = "",
) -> tuple[str, int, str]:
    """Per-sample: (mode, drop_last_points, tail_policy)."""
    method = str(denoise_method).lower().strip()
    sim_override = str(sim_denoise_method or "").lower().strip()
    if _is_sim_sample(sample) and sim_override and sim_override != "rule":
        drop_last_points = int(rule.get("drop_last_points", 0)) if rule else 0
        tail_policy = str(rule.get("tail_policy", "")) if rule else ""
        return sim_override, drop_last_points, tail_policy
    if method == "rule" and _is_sim_sample(sample):
        default_mode = _sim_default_mode(sample)
        if not rule:
            return default_mode, 0, ""
        mode = _infer_sim_rule_mode(rule, default=default_mode)
        drop_last_points = int(rule.get("drop_last_points", 0))
        tail_policy = str(rule.get("tail_policy", ""))
        return mode, drop_last_points, tail_policy
    mode = _resolve_denoise_mode(rule, method, rule_fallback=rule_fallback)
    drop_last_points = int(rule.get("drop_last_points", 0))
    tail_policy = str(rule.get("tail_policy", default_tail_policy))
    return mode, drop_last_points, tail_policy


def _rule_optional_float(rule: dict, key: str) -> float | None:
    if key not in rule or rule[key] is None:
        return None
    return float(rule[key])


def _rule_optional_int(rule: dict, key: str, default: int) -> int:
    if key not in rule or rule[key] is None:
        return int(default)
    return int(rule[key])


def _curve_overrides_from_rule(
    rule: dict,
    *,
    filter_window: int,
    filter_polyorder: int,
    plateau_flatten: bool,
    plateau_slope_eps: float,
    plateau_min_duration_ms: float,
    plateau_blend_ms: float,
) -> dict:
    """Per-sample denoise overrides from filter_effect_rules.json."""
    pf = rule.get("plateau_flatten", plateau_flatten)
    return {
        "filter_window": _rule_optional_int(rule, "filter_window", filter_window),
        "filter_polyorder": _rule_optional_int(rule, "filter_polyorder", filter_polyorder),
        "filter_post_ma_window": max(0, _rule_optional_int(rule, "filter_post_ma_window", 0)),
        "plateau_flatten": bool(pf) if pf is not None else bool(plateau_flatten),
        "plateau_slope_eps": float(rule.get("plateau_slope_eps", plateau_slope_eps)),
        "plateau_min_duration_ms": float(rule.get("plateau_min_duration_ms", plateau_min_duration_ms)),
        "plateau_blend_ms": float(rule.get("plateau_blend_ms", plateau_blend_ms)),
        "plateau_t_start_ms": _rule_optional_float(rule, "plateau_t_start_ms"),
        "plateau_force": bool(rule.get("plateau_force", False)),
    }


def _resolve_plateau_start(
    x: np.ndarray,
    y: np.ndarray,
    slope_eps: float = 0.15,
    min_duration_ms: float = 0.15,
    plateau_t_start_ms: float | None = None,
    plateau_force: bool = False,
) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if plateau_t_start_ms is not None:
        t_fixed = float(plateau_t_start_ms)
        if len(x) and 0.0 <= t_fixed <= float(x[-1]) + 1e-9:
            return t_fixed

    t_auto = _detect_plateau_start(x, y, slope_eps=slope_eps, min_duration_ms=min_duration_ms)
    if t_auto is not None:
        return t_auto

    if plateau_force:
        y_max = float(np.max(y))
        if y_max > 1e-9:
            i_peak = int(np.argmax(y))
            dt = _grid_dt(x)
            min_pts = max(3, int(np.ceil(float(min_duration_ms) / dt)))
            for i in range(i_peak, len(x) - min_pts + 1):
                seg_y = y[i:]
                seg_x = x[i:] - x[i]
                duration = max(float(seg_x[-1]), 1e-9)
                net_slope = (float(seg_y[-1]) - float(seg_y[0])) / duration
                if abs(net_slope) > float(slope_eps):
                    continue
                if float(np.mean(seg_y)) < 0.92 * y_max:
                    continue
                return float(x[i])
            if i_peak >= len(x) - min_pts:
                onset = 0.90 * y_max
                start_i = max(1, int(0.15 * len(x)))
                idx = np.where((y >= onset) & (np.arange(len(y)) >= start_i))[0]
                if len(idx):
                    return float(x[int(idx[0])])
            return float(x[i_peak])
    return None


# ---------------------------------------------------------------------------
# Core: single-curve denoised reconstruction
# ---------------------------------------------------------------------------


def _build_denoised_curve(
    sample: dict,
    max_time: float = 3.0,
    resolution: int = 128,
    mode: str = "fit",
    filter_type: str = "savgol",
    filter_window: int = 15,
    filter_polyorder: int = 1,
    fit_only_if_short_ms: float = 1.0,
    splice_time_ms: float = 0.5,
    blend_width_ms: float = 0.08,
    post_blend_ma_window: int = 9,
    drop_last_points: int = 0,
    tail_policy: str = "",
    plateau_flatten: bool = True,
    plateau_slope_eps: float = 0.15,
    plateau_min_duration_ms: float = 0.15,
    plateau_blend_ms: float = 0.12,
    plateau_t_start_ms: float | None = None,
    plateau_force: bool = False,
    filter_post_ma_window: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single denoised (time, penetration) curve on a uniform grid.

    Modes
    -----
    - ``"fit"``:   exponential fit only
    - ``"filter"``: linear interpolation + smoothing + plateau flattening
    - ``"hybrid"``: fit for early time, filtered curve for tail, with smooth blending
    """
    x_sorted, y_sorted = _sanitize_and_sort_single_sample(sample)
    x_sorted, y_sorted = _trim_tail_by_rule(x_sorted, y_sorted, drop_last_points=drop_last_points, tail_policy=tail_policy)
    x_max = min(float(x_sorted.max()), float(max_time))
    x_new = np.linspace(0.0, x_max, int(resolution), dtype=np.float64)

    # fitting baseline
    if "fit_param" in sample and sample["fit_param"] and len(sample["fit_param"]) >= 2:
        y_fit = func(x_new, *sample["fit_param"])
    else:
        try:
            fit_param, _ = opt.curve_fit(func, x_sorted, y_sorted, maxfev=3000)
            y_fit = func(x_new, *fit_param)
        except Exception:
            y_fit = np.interp(x_new, x_sorted, y_sorted)

    y_interp = np.interp(x_new, x_sorted, y_sorted)
    y_filt = _apply_filter_1d(y_interp, filter_type=filter_type, window=filter_window, polyorder=filter_polyorder)
    if int(filter_post_ma_window) > 1:
        y_filt = _apply_filter_1d(y_filt, filter_type="moving_avg", window=int(filter_post_ma_window), polyorder=0)

    m = str(mode).lower().strip()
    if m == "fit":
        return x_new, _anchor_origin(x_new, y_fit)
    if m == "filter":
        y_out = y_filt.copy()
        if plateau_flatten:
            t_start = _resolve_plateau_start(
                x_new, y_out,
                slope_eps=plateau_slope_eps,
                min_duration_ms=plateau_min_duration_ms,
                plateau_t_start_ms=plateau_t_start_ms,
                plateau_force=plateau_force,
            )
            if t_start is not None:
                y_out = _flatten_plateau_median(x_new, y_out, t_start, blend_ms=plateau_blend_ms)
                y_out = _smooth_knee_region(x_new, y_out, t_knee=t_start, knee_smooth_ms=plateau_blend_ms)
                plateau_mask = x_new >= float(t_start)
                if np.any(plateau_mask):
                    y_out[plateau_mask] = float(np.median(y_out[plateau_mask]))
        return x_new, _anchor_origin(x_new, y_out)

    # hybrid
    if x_max < float(fit_only_if_short_ms):
        y_out = y_fit.copy()
    else:
        y_out = y_fit.copy()
        splice_t = float(splice_time_ms)
        tail_mask = x_new > splice_t
        y_out[tail_mask] = y_filt[tail_mask]
        bw = max(1e-6, float(blend_width_ms))
        left = splice_t - 0.5 * bw
        right = splice_t + 0.5 * bw
        blend_mask = (x_new >= left) & (x_new <= right)
        if np.any(blend_mask):
            alpha = _smoothstep01((x_new[blend_mask] - left) / max(right - left, 1e-12))
            y_out[blend_mask] = (1.0 - alpha) * y_fit[blend_mask] + alpha * y_filt[blend_mask]

    w_post = max(1, int(post_blend_ma_window))
    if w_post > 1:
        y_out = _apply_filter_1d(y_out, filter_type="moving_avg", window=w_post, polyorder=0)
    return x_new, _anchor_origin(x_new, y_out)


# ---------------------------------------------------------------------------
# Batch processing pipelines
# ---------------------------------------------------------------------------


def _denoise_sample(
    sample: dict,
    method: str,
    max_time: float,
    resolution: int,
    filter_type: str,
    filter_window: int,
    filter_polyorder: int,
    blend_width_ms: float,
    post_blend_ma_window: int,
    fit_only_if_short_ms: float,
    splice_time_ms: float,
    plateau_flatten: bool,
    plateau_slope_eps: float,
    plateau_min_duration_ms: float,
    plateau_blend_ms: float,
    rule_fallback: str,
    default_tail_policy: str,
    sim_denoise_method: str,
    sample_rule_map: dict[str, dict] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve per-sample denoise rule and build the denoised (time, penetration) curve."""
    rule = _resolve_sample_rule(sample, sample_rule_map=sample_rule_map)
    mode, drop_last_points, tail_policy = _resolve_sample_denoise_plan(
        sample, rule, method,
        rule_fallback=rule_fallback,
        default_tail_policy=default_tail_policy,
        sim_denoise_method=sim_denoise_method,
    )
    sample_max_time = _resolve_max_time_from_rule(rule, max_time)
    curve_kw = _curve_overrides_from_rule(
        rule,
        filter_window=filter_window,
        filter_polyorder=filter_polyorder,
        plateau_flatten=plateau_flatten,
        plateau_slope_eps=plateau_slope_eps,
        plateau_min_duration_ms=plateau_min_duration_ms,
        plateau_blend_ms=plateau_blend_ms,
    )
    return _build_denoised_curve(
        sample,
        max_time=sample_max_time,
        resolution=resolution,
        mode=mode,
        filter_type=filter_type,
        drop_last_points=drop_last_points,
        tail_policy=tail_policy,
        fit_only_if_short_ms=fit_only_if_short_ms,
        splice_time_ms=splice_time_ms,
        blend_width_ms=blend_width_ms,
        post_blend_ma_window=post_blend_ma_window,
        **curve_kw,
    )


def fit_and_augment_data(
    dataset: list[dict],
    normalize: bool = True,
    max_time: float = 3.0,
    resolution: int = 128,
    denoise_method: str = "fit",
    filter_type: str = "savgol",
    filter_window: int = 15,
    filter_polyorder: int = 1,
    sample_rule_map: dict[str, dict] | None = None,
    default_tail_policy: str = "auto_drop_abnormal_tail",
    fit_only_if_short_ms: float = 1.0,
    splice_time_ms: float = 0.5,
    blend_width_ms: float = 0.12,
    post_blend_ma_window: int = 11,
    plateau_flatten: bool = True,
    plateau_slope_eps: float = 0.15,
    plateau_min_duration_ms: float = 0.15,
    plateau_blend_ms: float = 0.12,
    rule_fallback: str = "filter",
    sim_denoise_method: str = "",
) -> tuple[list[dict], float, float]:
    """Full preprocessing: denoise every sample, resample to ``resolution`` grid, normalise to [0, 1].

    Returns
    -------
    dataset, global_y_max, global_y_min
    """
    all_fitted_penetration: list[float] = []
    method = str(denoise_method).lower().strip()
    if method == "fit":
        dataset, _ = datafitting(dataset)

    for sample in dataset:
        x_new, y_new = _denoise_sample(
            sample, method, max_time, resolution,
            filter_type, filter_window, filter_polyorder,
            blend_width_ms, post_blend_ma_window,
            fit_only_if_short_ms, splice_time_ms,
            plateau_flatten, plateau_slope_eps, plateau_min_duration_ms, plateau_blend_ms,
            rule_fallback, default_tail_policy, sim_denoise_method, sample_rule_map,
        )
        sample["time"] = x_new.tolist()
        sample["penetration"] = y_new.tolist()
        all_fitted_penetration.extend(y_new.tolist())

    all_fitted_penetration = np.array(all_fitted_penetration, dtype=np.float64)
    global_y_min = float(all_fitted_penetration.min())
    global_y_max = float(all_fitted_penetration.max())

    if normalize:
        for sample in dataset:
            y_fitted = np.array(sample["penetration"], dtype=np.float64)
            y_normalized = (y_fitted - global_y_min) / (global_y_max - global_y_min)
            sample["penetration"] = y_normalized.tolist()

    print(f"[denoise={method}] y_max={global_y_max:.6f}, y_min={global_y_min:.6f}")
    return dataset, global_y_max, global_y_min


def prepare_test_holdout(
    test_data: list[dict],
    max_p: float,
    min_p: float,
    resolution: int = 128,
    max_time: float = 3.0,
    denoise_method: str = "fit",
    filter_type: str = "savgol",
    filter_window: int = 15,
    filter_polyorder: int = 1,
    sample_rule_map: dict[str, dict] | None = None,
    default_tail_policy: str = "auto_drop_abnormal_tail",
    fit_only_if_short_ms: float = 1.0,
    splice_time_ms: float = 0.5,
    blend_width_ms: float = 0.12,
    post_blend_ma_window: int = 11,
    plateau_flatten: bool = True,
    plateau_slope_eps: float = 0.15,
    plateau_min_duration_ms: float = 0.15,
    plateau_blend_ms: float = 0.12,
    rule_fallback: str = "filter",
    sim_denoise_method: str = "",
) -> list[dict]:
    """Process holdout test set with the SAME denoise pipeline as training.

    Uses training-set ``max_p`` / ``min_p`` for normalisation so predictions
    are directly comparable.
    """
    method = str(denoise_method).lower().strip()
    if method == "fit":
        test_data, _ = datafitting(test_data)
    for sample in test_data:
        x_new, y_new = _denoise_sample(
            sample, method, max_time, resolution,
            filter_type, filter_window, filter_polyorder,
            blend_width_ms, post_blend_ma_window,
            fit_only_if_short_ms, splice_time_ms,
            plateau_flatten, plateau_slope_eps, plateau_min_duration_ms, plateau_blend_ms,
            rule_fallback, default_tail_policy, sim_denoise_method, sample_rule_map,
        )
        y_normalized = (np.array(y_new, dtype=np.float64) - min_p) / (max_p - min_p)
        sample["time"] = x_new.tolist()
        sample["penetration"] = y_normalized.tolist()
    return normalize_time(test_data)


# ---------------------------------------------------------------------------
# Time-window augmentation
# ---------------------------------------------------------------------------


def _scale_penetration_after_time_augment(
    y: np.ndarray,
    *,
    from_fit_func: bool,
    penetration_already_normalized: bool,
    min_p: float,
    max_p: float,
) -> np.ndarray:
    """After time-window slicing, map penetration into [0, 1] space."""
    y = np.asarray(y, dtype=np.float64)
    den = float(max_p) - float(min_p)
    if from_fit_func:
        if den <= 0:
            return np.clip(y, 0.0, 1.0)
        return (y - float(min_p)) / den
    if penetration_already_normalized:
        return y
    if den <= 0:
        return np.clip(y, 0.0, 1.0)
    return (y - float(min_p)) / den


def data_enhance(
    dataset: list[dict],
    max_p: float,
    min_p: float,
    factor: int,
    window_size: list[float] | float,
    resolution: int | None = None,
    penetration_already_normalized: bool = True,
) -> list[dict]:
    """Time-window slice augmentation.

    For each sample whose total duration >= 1.5 ms, produces up to ``factor``
    sub-windows (beginning, end, middle), resampled to ``resolution``.
    """
    if not dataset:
        return dataset
    if isinstance(window_size, (int, float)):
        window_size = [float(window_size)]
    lens = {len(s.get("time", [])) for s in dataset}
    if len(lens) != 1:
        raise ValueError(f"data_enhance: all samples must share the same time length before slice-aug; got lengths {sorted(lens)}")
    (in_len,) = tuple(lens)
    if resolution is None:
        resolution = int(in_len)
    else:
        resolution = int(resolution)
        if resolution != in_len:
            raise ValueError(
                f"data_enhance: resolution={resolution} but fitted curves have length {in_len}; "
                "pass the same value as fit_and_augment_data(..., resolution=...)."
            )

    new_dataset: list[dict] = []
    for sample in dataset:
        raw_time = sample["time"].copy()
        raw_pene = sample["penetration"].copy()
        len_time = len(raw_time)
        for f in window_size:
            new_time_set: list[list[float]] = []
            new_penetration_set: list[list[float]] = []
            if raw_time[-1] >= 1.5:
                target_len = int(len_time * f)
                # --- front window ---
                new_time_0 = raw_time[:target_len]
                new_time_0_np = np.linspace(new_time_0[0], new_time_0[-1], resolution)
                if "fit_param" in sample:
                    new_penetration_0 = func(new_time_0_np, *sample["fit_param"])
                    from_fit = True
                else:
                    new_penetration_0 = np.interp(
                        new_time_0_np,
                        np.array(raw_time, dtype=np.float64),
                        np.array(raw_pene, dtype=np.float64),
                    )
                    from_fit = False
                new_penetration_0 = _scale_penetration_after_time_augment(
                    new_penetration_0,
                    from_fit_func=from_fit,
                    penetration_already_normalized=penetration_already_normalized,
                    min_p=min_p, max_p=max_p,
                )
                new_time_set.append(new_time_0_np.tolist())
                new_penetration_set.append(new_penetration_0.tolist())

                if factor == 3:
                    # --- rear window ---
                    new_time_1 = raw_time[-target_len:]
                    new_time_1_np = np.linspace(new_time_1[0], new_time_1[-1], resolution)
                    if "fit_param" in sample:
                        new_penetration_1 = func(new_time_1_np, *sample["fit_param"])
                        from_fit = True
                    else:
                        new_penetration_1 = np.interp(
                            new_time_1_np,
                            np.array(raw_time, dtype=np.float64),
                            np.array(raw_pene, dtype=np.float64),
                        )
                        from_fit = False
                    new_penetration_1 = _scale_penetration_after_time_augment(
                        new_penetration_1,
                        from_fit_func=from_fit,
                        penetration_already_normalized=penetration_already_normalized,
                        min_p=min_p, max_p=max_p,
                    )
                    new_time_set.append(new_time_1_np.tolist())
                    new_penetration_set.append(new_penetration_1.tolist())

                    # --- middle window ---
                    start_idx = (len_time - target_len) // 2
                    end_idx = start_idx + target_len
                    new_time_2 = raw_time[start_idx:end_idx]
                    new_time_2_np = np.linspace(new_time_2[0], new_time_2[-1], resolution)
                    if "fit_param" in sample:
                        new_penetration_2 = func(new_time_2_np, *sample["fit_param"])
                        from_fit = True
                    else:
                        new_penetration_2 = np.interp(
                            new_time_2_np,
                            np.array(raw_time, dtype=np.float64),
                            np.array(raw_pene, dtype=np.float64),
                        )
                        from_fit = False
                    new_penetration_2 = _scale_penetration_after_time_augment(
                        new_penetration_2,
                        from_fit_func=from_fit,
                        penetration_already_normalized=penetration_already_normalized,
                        min_p=min_p, max_p=max_p,
                    )
                    new_time_set.append(new_time_2_np.tolist())
                    new_penetration_set.append(new_penetration_2.tolist())

                for i in range(factor):
                    sub_sample: dict[str, Any] = {}
                    sub_sample["time"] = new_time_set[i]
                    sub_sample["penetration"] = new_penetration_set[i]
                    sub_sample["params"] = sample["params"].copy()
                    new_dataset.append(sub_sample)
    dataset += new_dataset
    return dataset


# ---------------------------------------------------------------------------
# Time normalisation
# ---------------------------------------------------------------------------


def normalize_time(dataset: list[dict], extend: bool = False, max_time: float = 3.0) -> list[dict]:
    """Normalise time to [0, 1] and append ``cutoff = max_t / max_time`` as 5th param."""
    for z, sample in enumerate(dataset):
        time = sample["time"]
        max_t, min_t = np.array(time).max(), np.array(time).min()
        new_time = [(i - min_t) / (max_t - min_t) for i in time]
        dataset[z]["time"] = new_time
        cutoff = max_t / max_time
        cutoff = float(np.clip(cutoff, 0.0, 1.0))
        if extend:
            dataset[z]["params"].append(1.0)
        else:
            dataset[z]["params"].append(cutoff)
    return dataset


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SpraySet(Dataset):
    """Torch Dataset wrapping a list of preprocessed samples."""

    def __init__(self, dataset: list[dict]):
        super().__init__()
        self.samples = dataset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        times = torch.tensor(self.samples[index]["time"], dtype=torch.float32)
        penetration = torch.tensor(self.samples[index]["penetration"], dtype=torch.float32)
        params = torch.tensor(self.samples[index]["params"], dtype=torch.float32)
        return times, params, penetration


# ---------------------------------------------------------------------------
# Sample weighting for sparse-region up-sampling
# ---------------------------------------------------------------------------


def _compute_1d_hist(values: np.ndarray, num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    counts, bin_edges = np.histogram(values, bins=num_bins)
    counts = counts.astype(np.float64)
    counts[counts == 0] = 1.0
    bin_indices = np.searchsorted(bin_edges, values, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
    sample_counts = counts[bin_indices]
    return sample_counts, bin_edges


def compute_sample_weights(
    dataset: list[dict],
    num_bins: int = 10,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
) -> np.ndarray:
    """Compute per-sample weights inversely proportional to 4-D parameter density.

    Sparse regions get higher weight for WeightedRandomSampler.
    """
    if len(dataset) == 0:
        return np.array([], dtype=np.float64)

    params = np.array([sample["params"][:4] for sample in dataset], dtype=np.float64)
    N, D = params.shape
    assert D == 4, f"Expected 4 physical params, got {D}"

    per_dim_counts = []
    for d in range(D):
        counts_d, _ = _compute_1d_hist(params[:, d], num_bins=num_bins)
        per_dim_counts.append(counts_d)

    per_dim_counts = np.stack(per_dim_counts, axis=1)  # (N, 4)
    per_dim_weights = 1.0 / per_dim_counts
    weights = per_dim_weights.mean(axis=1)
    mean_w = float(weights.mean())
    if mean_w > 0:
        weights = weights / mean_w
    weights = np.clip(weights, min_weight, max_weight)

    print(f"[data_augment] sample weights stats: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    return weights
