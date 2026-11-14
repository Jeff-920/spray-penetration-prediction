"""Inference and evaluation for trained spray penetration FNO models.

Loads a pre-trained FNO checkpoint + VAEClassify, runs prediction on
holdout test data, and computes physical-unit (mm) metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from data.processor import (
    SpraySet,
    _load_rule_map,
    normalize_time,
    prepare_test_holdout,
)
from data.reader import process_file_list, process_folder, process_single_file
from model.fno import SprayModel
from train.classifier import VAEClassify


# ---------------------------------------------------------------------------
# Standalone checkpoint helpers (duplicated from trainer.py to avoid circular import)
# ---------------------------------------------------------------------------


def safe_state_dict(ckpt: dict) -> dict:
    """Extract model weights from a checkpoint dict regardless of wrapping keys."""
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            ckpt = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        if "_metadata" in ckpt:
            ckpt = {k: v for k, v in ckpt.items() if k != "_metadata"}
    return ckpt


def torch_load_compat(path: str, map_location: str):
    """PyTorch 2.6-compatible checkpoint loading (weights_only fallback)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)

# ---------------------------------------------------------------------------
# Metric constants
# ---------------------------------------------------------------------------

MSE_REL_SQ_MIN_TIME_MS = 0.5


# ---------------------------------------------------------------------------
# Denormalisation & time helpers
# ---------------------------------------------------------------------------


def denorm_penetration(y_norm: np.ndarray, min_p: float, max_p: float) -> np.ndarray:
    """Map normalised penetration [0, 1] back to physical mm."""
    y = np.asarray(y_norm, dtype=np.float64)
    return y * (float(max_p) - float(min_p)) + float(min_p)


def _cutoff_from_params(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    if p.ndim == 1:
        return np.asarray(float(p[4])) if p.size >= 5 else np.asarray(float(p[-1]))
    return p[..., 4] if p.shape[-1] >= 5 else p[..., -1]


def physical_time_ms_from_norm(times: np.ndarray, params: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    cutoff = np.asarray(_cutoff_from_params(params), dtype=np.float64)
    if cutoff.ndim == 0:
        return t * float(cutoff) * 3.0
    if cutoff.ndim == 1:
        cutoff = cutoff.reshape(-1, 1)
    return t * cutoff * 3.0


# ---------------------------------------------------------------------------
# Pointwise metrics
# ---------------------------------------------------------------------------


def compute_pointwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    time_ms: np.ndarray | None = None,
    ignore_zero_true: bool = True,
    mse_rel_sq_min_time_ms: float = MSE_REL_SQ_MIN_TIME_MS,
) -> dict[str, float | int]:
    """MAE, MSE, global R²; MSE_rel_sq only where t > threshold."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {yt.shape} vs {yp.shape}")

    mask = np.isfinite(yt) & np.isfinite(yp)
    if ignore_zero_true:
        mask &= yt != 0.0

    n_valid = int(mask.sum())
    if n_valid == 0:
        return {
            "n_points": 0, "n_points_rel": 0,
            "mse": float("nan"), "mae": float("nan"),
            "mse_rel_sq": float("nan"), "r2": float("nan"),
            "mse_rel_sq_min_time_ms": float(mse_rel_sq_min_time_ms),
        }

    yt_v, yp_v = yt[mask], yp[mask]

    out: dict[str, float | int] = {
        "n_points": n_valid,
        "mse": float(mean_squared_error(yt_v, yp_v)),
        "mae": float(mean_absolute_error(yt_v, yp_v)),
        "r2": float(r2_score(yt_v, yp_v)),
        "mse_rel_sq_min_time_ms": float(mse_rel_sq_min_time_ms),
    }

    if time_ms is None:
        out["n_points_rel"] = 0
        out["mse_rel_sq"] = float("nan")
        return out

    t_flat = np.asarray(time_ms, dtype=np.float64).reshape(-1)
    if t_flat.shape != yt.shape:
        raise ValueError(f"time_ms and y_true must have the same shape, got {t_flat.shape} vs {yt.shape}")

    rel_mask = mask & (t_flat > float(mse_rel_sq_min_time_ms))
    if ignore_zero_true:
        rel_mask &= yt != 0.0
    n_rel = int(rel_mask.sum())
    out["n_points_rel"] = n_rel
    if n_rel > 0:
        rel = (yp[rel_mask] - yt[rel_mask]) / yt[rel_mask]
        out["mse_rel_sq"] = float(np.mean(rel ** 2))
    else:
        out["mse_rel_sq"] = float("nan")
    return out


def compute_batch_curve_metrics(
    penetration: np.ndarray,
    prediction: np.ndarray,
    *,
    times: np.ndarray | None = None,
    params: np.ndarray | None = None,
    min_p: float | None = None,
    max_p: float | None = None,
    ignore_zero_true: bool = True,
    mse_rel_sq_min_time_ms: float = MSE_REL_SQ_MIN_TIME_MS,
) -> dict[str, float | int]:
    """Flatten (n_samples, seq_len) batches and compute pointwise metrics."""
    pen = np.asarray(penetration, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if pen.shape != pred.shape:
        raise ValueError(f"penetration and prediction shapes differ: {pen.shape} vs {pred.shape}")
    if min_p is not None and max_p is not None:
        pen = denorm_penetration(pen, min_p, max_p)
        pred = denorm_penetration(pred, min_p, max_p)

    time_ms = None
    if times is not None and params is not None:
        time_ms = physical_time_ms_from_norm(times, params).reshape(-1)

    return compute_pointwise_metrics(pen, pred, time_ms=time_ms, ignore_zero_true=ignore_zero_true, mse_rel_sq_min_time_ms=mse_rel_sq_min_time_ms)


def compute_merged_test_metrics(
    penetration_a: np.ndarray, prediction_a: np.ndarray,
    penetration_b: np.ndarray, prediction_b: np.ndarray,
    *, times_a: np.ndarray, params_a: np.ndarray,
    times_b: np.ndarray, params_b: np.ndarray,
    min_p: float, max_p: float,
    n_holdout: int | None = None, n_extra: int | None = None,
    ignore_zero_true: bool = True,
    mse_rel_sq_min_time_ms: float = MSE_REL_SQ_MIN_TIME_MS,
) -> dict[str, float | int]:
    """Pool holdout + extra curves into global metrics."""
    pen = np.concatenate([np.asarray(penetration_a, dtype=np.float64), np.asarray(penetration_b, dtype=np.float64)], axis=0)
    pred = np.concatenate([np.asarray(prediction_a, dtype=np.float64), np.asarray(prediction_b, dtype=np.float64)], axis=0)
    times = np.concatenate([np.asarray(times_a, dtype=np.float64), np.asarray(times_b, dtype=np.float64)], axis=0)
    params = np.concatenate([np.asarray(params_a, dtype=np.float64), np.asarray(params_b, dtype=np.float64)], axis=0)

    out = compute_batch_curve_metrics(pen, pred, times=times, params=params, min_p=min_p, max_p=max_p,
                                      ignore_zero_true=ignore_zero_true, mse_rel_sq_min_time_ms=mse_rel_sq_min_time_ms)
    out["n_samples"] = int(pen.shape[0])
    if n_holdout is not None:
        out["n_holdout"] = int(n_holdout)
    if n_extra is not None:
        out["n_extra"] = int(n_extra)

    pen_mm = denorm_penetration(pen, min_p, max_p)
    pred_mm = denorm_penetration(pred, min_p, max_p)
    sample_r2s: list[float] = []
    for i in range(pen_mm.shape[0]):
        yt = pen_mm[i].reshape(-1)
        yp = pred_mm[i].reshape(-1)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if ignore_zero_true:
            mask &= yt != 0.0
        if int(mask.sum()) < 2:
            continue
        sample_r2s.append(float(r2_score(yt[mask], yp[mask])))
    out["r2_sample_mean"] = float(np.mean(sample_r2s)) if sample_r2s else float("nan")
    return out


def print_test_metrics(metrics: dict[str, Any], label: str = "Test") -> None:
    n_samples = metrics.get("n_samples")
    if n_samples is not None:
        print(f"\n[{label}] {n_samples} samples, pointwise n={metrics.get('n_points', '?')}:")
    else:
        print(f"\n[{label}] pointwise metrics (n={metrics.get('n_points', '?')}):")
    print(f"  MSE:         {metrics.get('mse', float('nan')):.6f}")
    print(f"  MAE:         {metrics.get('mae', float('nan')):.6f}")
    rel_min_t = metrics.get("mse_rel_sq_min_time_ms", MSE_REL_SQ_MIN_TIME_MS)
    print(f"  MSE_rel_sq:  {metrics.get('mse_rel_sq', float('nan')):.6f}  (t>{rel_min_t} ms, n={metrics.get('n_points_rel', '?')})")
    print(f"  R2:          {metrics.get('r2', float('nan')):.6f}")
    r2_sm = metrics.get("r2_sample_mean")
    if r2_sm is not None and not (isinstance(r2_sm, float) and np.isnan(r2_sm)):
        print(f"  R2_sample_mean: {r2_sm:.6f}")


def metrics_to_jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_holdout_test(args, test_loader: DataLoader, max_p: float, min_p: float):
    """Run FNO prediction on holdout dataloader; return curves + R² (mm)."""
    device = torch.device(args.device)
    class_model = VAEClassify().to(device)
    ckpt = torch_load_compat(args.class_path, map_location=device)
    class_model.load_state_dict(safe_state_dict(ckpt))
    class_model.eval()

    model_path = os.path.join(args.save_dir, args.save_path)
    state = torch_load_compat(model_path, map_location=device)
    model = SprayModel(
        hidden_channels=args.hidden_channels, num_frq=args.num_frq, n_modes=args.n_modes,
        num_layer=args.num_layer,
        encoder_layer_sizes=args.encoder_layer_sizes,
        decoder_layer_sizes=args.decoder_layer_sizes,
        latent_size=args.latent_size, device=args.device,
    )
    model.load_state_dict(safe_state_dict(state), strict=False)
    model.to(device)
    model.eval()

    all_times, all_params, all_prediction, all_penetration = [], [], [], []
    total_r2 = 0.0
    n_batch = 0

    with torch.no_grad():
        for times, params, penetration in test_loader:
            times, params, penetration = times.to(device), params.to(device), penetration.to(device)
            raw_params = params[:, :4]
            _, _, _, class_logit = class_model(raw_params)
            pred_class = torch.sigmoid(class_logit).detach()
            prediction = model(params, times, pred_class)

            # R² as computed in train (flattened)
            pred_np = prediction.detach().cpu().numpy()
            pen_np = penetration.detach().cpu().numpy()
            batch_r2 = float(r2_score(pen_np.reshape(-1, 1), pred_np.reshape(-1, 1)))
            total_r2 += batch_r2
            n_batch += 1

            all_times.append(times.cpu().numpy())
            all_params.append(params.cpu().numpy())
            all_prediction.append(prediction.cpu().numpy())
            all_penetration.append(penetration.cpu().numpy())

    avg_r2 = total_r2 / n_batch if n_batch else 0.0
    times = np.concatenate(all_times, axis=0) if all_times else np.array([])
    params = np.concatenate(all_params, axis=0) if all_params else np.array([])
    prediction = np.concatenate(all_prediction, axis=0) if all_prediction else np.array([])
    penetration = np.concatenate(all_penetration, axis=0) if all_penetration else np.array([])

    point_metrics = compute_batch_curve_metrics(penetration, prediction, times=times, params=params, min_p=min_p, max_p=max_p)
    point_metrics["r2_batch_mean"] = float(avg_r2)
    holdout_r2 = float(point_metrics["r2"])
    return times, params, prediction, penetration, holdout_r2, point_metrics


def evaluate_split_holdout_after_train(
    args, *, max_p: float, min_p: float, split_dir: Path, project_root: Path,
) -> dict | None:
    """Run 40-sample holdout inference after training; print and return metrics."""
    test_list_path = split_dir / "test_files.txt"
    if not test_list_path.is_file():
        return None
    test_paths = [line.strip().replace("\\", "/") for line in open(test_list_path, encoding="utf-8") if line.strip()]
    if not test_paths:
        return None
    base = str(project_root / "new") if test_paths[0].startswith("data_raw") else str(project_root)
    test_raw = process_file_list(base, test_paths)
    scale_path = split_dir / "scale_params.json"
    resolution = int(getattr(args, "fit_resolution", 128))
    if scale_path.is_file():
        with open(scale_path, encoding="utf-8") as f:
            resolution = int(json.load(f).get("resolution", resolution))
    test_proc = prepare_test_holdout(
        test_raw, max_p, min_p, resolution=resolution,
        denoise_method=args.denoise_method, filter_type=args.filter_type,
        filter_window=args.filter_window, filter_polyorder=args.filter_polyorder,
        sample_rule_map=_load_rule_map(getattr(args, "rule_json", "")),
        default_tail_policy="" if str(getattr(args, "default_tail_policy", "")).lower() == "none" else str(getattr(args, "default_tail_policy", "")),
        fit_only_if_short_ms=args.fit_only_if_short_ms,
        splice_time_ms=args.splice_time_ms, blend_width_ms=args.blend_width_ms,
        post_blend_ma_window=args.post_blend_ma_window,
        plateau_flatten=args.plateau_flatten, plateau_slope_eps=args.plateau_slope_eps,
        plateau_min_duration_ms=args.plateau_min_duration_ms,
        plateau_blend_ms=args.plateau_blend_ms, rule_fallback=args.rule_fallback_mode,
    )
    holdout_loader = DataLoader(SpraySet(test_proc), batch_size=args.batch_size, shuffle=False)
    _, _, _, _, holdout_r2, holdout_metrics = run_holdout_test(args, holdout_loader, max_p, min_p)
    print("=" * 50)
    print(f"Holdout Test ({len(test_proc)} samples) R2 (global pooled, mm): {holdout_r2:.6f}")
    print_test_metrics(holdout_metrics, label="Holdout (40, physical mm)")
    print("=" * 50)
    return {"holdout_r2": float(holdout_r2), "n_holdout": int(len(test_proc)), "holdout_metrics": metrics_to_jsonable(holdout_metrics)}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_holdout_regression_scatter(
    penetration_true: np.ndarray,
    prediction_true: np.ndarray,
    out_path: str,
    dpi: int = 300,
    time_stride: int = 30,
) -> None:
    """Subsampled true-vs-predicted scatter (mm) with y=x line + R²."""
    pt = np.asarray(penetration_true, dtype=np.float64)
    pp = np.asarray(prediction_true, dtype=np.float64)
    stride = max(1, int(time_stride))
    if pt.ndim == 1:
        pt = pt.reshape(1, -1)
        pp = pp.reshape(1, -1)
    yt_parts, yp_parts = [], []
    for i in range(pt.shape[0]):
        yt_parts.append(pt[i, ::stride])
        yp_parts.append(pp[i, ::stride])
    yt = np.concatenate(yt_parts)
    yp = np.concatenate(yp_parts)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 2:
        return

    r2_flat = r2_score(yt, yp)
    lo = min(float(yt.min()), float(yp.min()))
    hi = max(float(yt.max()), float(yp.max()))
    span = hi - lo
    margin = 0.02 * (span + 1e-12)
    lo, hi = lo - margin, hi + margin

    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(yt, yp, s=24, alpha=0.50, c="#F4B183", edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
    ax.set_xlabel("True penetration [mm]", fontsize=17)
    ax.set_ylabel("Predicted penetration [mm]", fontsize=17)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_holdout_curve_outputs(
    *, index: int, prefix: str,
    times_row: np.ndarray, params_row: np.ndarray,
    penetration_true_row: np.ndarray, prediction_true_row: np.ndarray,
    raw_sample: dict,
    test_path: str, test_fig_path: str, fig_dpi: int = 300,
) -> None:
    """Save per-sample prediction vs experiment .txt + .png."""
    end_time = float(params_row[-1]) * 3.0
    time_true = np.linspace(0, end_time, len(times_row))
    plt.scatter(raw_sample["time"], raw_sample["penetration"], s=18, label="Experiment")
    plt.plot(time_true, prediction_true_row, linewidth=2.0, label="Prediction")
    plt.xlabel("Time (ms)")
    plt.ylabel("Spray tip penetration (mm)")
    plt.legend()
    out_name = f"{prefix}_{index}"
    os.makedirs(test_fig_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    plt.savefig(os.path.join(test_fig_path, f"{out_name}.png"), dpi=fig_dpi)
    plt.clf()
    with open(os.path.join(test_path, f"{out_name}.txt"), "w", encoding="utf-8") as f:
        f.write("Time Penetration Prediction\n")
        for j in range(len(penetration_true_row)):
            f.write(f"{time_true[j]} {penetration_true_row[j]} {prediction_true_row[j]}\n")


# ---------------------------------------------------------------------------
# Standalone inference CLI
# ---------------------------------------------------------------------------


def _collect_test_samples(test_data_dir: str) -> list[dict]:
    """Read all .dat/.txt files from a directory into raw samples."""
    test_dir = Path(test_data_dir)
    samples: list[dict] = []
    for fp in sorted(test_dir.glob("*")):
        if fp.suffix.lower() in (".dat", ".txt"):
            sample = process_single_file(str(fp))
            sample["file_path"] = str(fp)
            samples.append(sample)
    if not samples:
        raise FileNotFoundError(f"No .dat/.txt files found in {test_data_dir}")
    return samples


def run_inference(
    test_data_dir: str = "data/test_data",
    weight_path: str = "model_weight/fno_model/trained_with_experimental_data.pth",
    class_path: str = "model_weight/vae_model/vae_classify.pth",
    output_dir: str = "fig",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    denoise_method: str = "fit",
    rule_json: str = "",
    rule_fallback_mode: str = "filter",
    max_p: float = 165.00189471502551,
    min_p: float = -1.9572725888325389,
    resolution: int = 128,
    batch_size: int = 8,
    hidden_channels: int = 112,
    num_frq: int = 8,
    n_modes: int | tuple = 20,
    num_layer: int = 6,
    encoder_layer_sizes: list[int] | None = None,
    decoder_layer_sizes: list[int] | None = None,
    latent_size: int = 8,
) -> tuple[dict, str]:
    """End-to-end inference on test data.

    Returns
    -------
    metrics_dict, scatter_png_path
    """
    if encoder_layer_sizes is None:
        encoder_layer_sizes = [5, 16, 8]
    if decoder_layer_sizes is None:
        decoder_layer_sizes = [8, 16, 5]

    os.makedirs(output_dir, exist_ok=True)
    device_obj = torch.device(device)

    # ---- load raw test data ----
    print(f"[inference] Loading test data from {test_data_dir}")
    test_raw = _collect_test_samples(test_data_dir)
    n_test = len(test_raw)
    print(f"[inference] {n_test} test samples loaded")

    # ---- preprocess (on deep copy so raw data is preserved for plotting) ----
    sample_rule_map = _load_rule_map(rule_json) if rule_json else None
    test_proc = prepare_test_holdout(
        copy.deepcopy(test_raw), max_p, min_p, resolution=resolution,
        denoise_method=denoise_method,
        sample_rule_map=sample_rule_map,
        rule_fallback=rule_fallback_mode,
    )
    test_loader = DataLoader(SpraySet(test_proc), batch_size=batch_size, shuffle=False)

    # ---- load models ----
    print(f"[inference] Loading classifier from {class_path}")
    class_model = VAEClassify().to(device_obj)
    class_ckpt = torch_load_compat(class_path, map_location=device)
    class_model.load_state_dict(safe_state_dict(class_ckpt))
    class_model.eval()

    print(f"[inference] Loading FNO weights from {weight_path}")
    model = SprayModel(
        hidden_channels=hidden_channels, num_frq=num_frq, n_modes=n_modes,
        num_layer=num_layer,
        encoder_layer_sizes=encoder_layer_sizes,
        decoder_layer_sizes=decoder_layer_sizes,
        latent_size=latent_size, device=device,
    )
    state = torch_load_compat(weight_path, map_location=device)
    model.load_state_dict(safe_state_dict(state), strict=False)
    model.to(device_obj)
    model.eval()

    # ---- inference ----
    all_times, all_params, all_pred, all_pen = [], [], [], []
    with torch.no_grad():
        for times, params, penetration in test_loader:
            times, params, penetration = times.to(device_obj), params.to(device_obj), penetration.to(device_obj)
            raw_params = params[:, :4]
            _, _, _, class_logit = class_model(raw_params)
            pred_class = torch.sigmoid(class_logit).detach()
            prediction = model(params, times, pred_class)

            all_times.append(times.cpu().numpy())
            all_params.append(params.cpu().numpy())
            all_pred.append(prediction.cpu().numpy())
            all_pen.append(penetration.cpu().numpy())

    times = np.concatenate(all_times, axis=0)
    params = np.concatenate(all_params, axis=0)
    prediction = np.concatenate(all_pred, axis=0)
    penetration = np.concatenate(all_pen, axis=0)

    # ---- compute metrics (physical mm) ----
    metrics = compute_batch_curve_metrics(
        penetration, prediction, times=times, params=params, min_p=min_p, max_p=max_p,
        ignore_zero_true=False,
    )
    metrics["n_samples"] = n_test
    print("=" * 60)
    print(f"Inference Results ({n_test} test samples)")
    print_test_metrics(metrics, label="Holdout Test")
    print("=" * 60)

    # ---- save per-sample curves + scatter ----
    penetration_true = denorm_penetration(penetration, min_p, max_p)
    prediction_true = denorm_penetration(prediction, min_p, max_p)

    txt_out = os.path.join(output_dir, "txt")
    for i in range(n_test):
        _save_holdout_curve_outputs(
            index=i + 1, prefix="holdout",
            times_row=times[i], params_row=params[i],
            penetration_true_row=penetration_true[i],
            prediction_true_row=prediction_true[i],
            raw_sample=test_raw[i],
            test_path=txt_out, test_fig_path=output_dir,
        )

    scatter_png = os.path.join(output_dir, "holdout_regression_scatter.png")
    plot_holdout_regression_scatter(penetration_true, prediction_true, scatter_png)
    print(f"[inference] Per-sample curves saved to {output_dir}/")
    print(f"[inference] Regression scatter saved to {scatter_png}")

    # ---- save metrics JSON ----
    metrics_json_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_jsonable(metrics), f, indent=2)
    print(f"[inference] Metrics saved to {metrics_json_path}")

    return metrics, scatter_png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained FNO spray model.")
    parser.add_argument("--test_data_dir", type=str, default="data/test_data")
    parser.add_argument("--weight_path", type=str, default="model_weight/fno_model/trained_with_experimental_data.pth")
    parser.add_argument("--class_path", type=str, default="model_weight/vae_model/vae_classify.pth")
    parser.add_argument("--output_dir", type=str, default="fig")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--denoise_method", type=str, default="fit", choices=["fit", "filter", "hybrid"])
    parser.add_argument("--max_p", type=float, default=165.00189471502551)
    parser.add_argument("--min_p", type=float, default=-1.9572725888325389)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_channels", type=int, default=112)
    parser.add_argument("--num_frq", type=int, default=8)
    parser.add_argument("--n_modes", type=int, default=20)
    parser.add_argument("--num_layer", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        test_data_dir=args.test_data_dir,
        weight_path=args.weight_path,
        class_path=args.class_path,
        output_dir=args.output_dir,
        device=args.device,
        denoise_method=args.denoise_method,
        max_p=args.max_p,
        min_p=args.min_p,
        resolution=args.resolution,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        num_frq=args.num_frq,
        n_modes=args.n_modes,
        num_layer=args.num_layer,
    )
