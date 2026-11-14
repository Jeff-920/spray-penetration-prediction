"""Mixed experimental + simulation FNO training entry point.

Injects simulation samples into training set only (validation/test stay pure-experiment
from ``new/split``), with optional dynamic weighting of simulation samples over epochs.

Replicates: ``new/exp2/run_train_with_sim.py``
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

# extend path for original-relative configs
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from data.processor import (
    SpraySet,
    _load_rule_map,
    compute_sample_weights,
    data_enhance,
    fit_and_augment_data,
    load_merged_rule_map,
    normalize_time,
    prepare_test_holdout,
)
from data.reader import process_file_list, process_folder
from model.fno import SprayModel
from train.classifier import VAEClassify
from train.trainer import (
    build_log_path,
    parse_n_modes,
    train_validation_1d,
)
from inference.predictor import safe_state_dict, torch_load_compat
from inference.predictor import (
    compute_batch_curve_metrics,
    evaluate_split_holdout_after_train,
    metrics_to_jsonable,
    plot_holdout_regression_scatter,
    print_test_metrics,
    run_holdout_test,
)

# ---------------------------------------------------------------------------
# Seeding (deterministic)
# ---------------------------------------------------------------------------


def set_seed(seed: int, *, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dynamic sampler with source-aware epoch weighting
# ---------------------------------------------------------------------------


class DynamicSourceWeightedSampler(Sampler):
    """Epoch-aware weighted sampler.

    - Experimental samples: multiplier 1.0
    - Simulation samples: multiplier changes with epoch (linear / exp schedule)
    """

    def __init__(
        self,
        base_weights: np.ndarray,
        source_flags: np.ndarray,
        num_samples: int,
        total_epochs: int,
        schedule: str,
        sim_weight_start: float,
        sim_weight_end: float,
        sim_weight_exp_k: float,
        seed: int,
    ):
        self.base_weights = np.asarray(base_weights, dtype=np.float64)
        self.source_flags = np.asarray(source_flags, dtype=bool)
        self.num_samples = int(num_samples)
        self.total_epochs = max(1, int(total_epochs))
        self.schedule = str(schedule)
        self.sim_weight_start = float(sim_weight_start)
        self.sim_weight_end = float(sim_weight_end)
        self.sim_weight_exp_k = float(sim_weight_exp_k)
        self.seed = int(seed)
        self.epoch_idx = 0

    def _sim_weight_at(self, epoch_idx: int) -> float:
        if self.total_epochs <= 1:
            progress = 1.0
        else:
            progress = min(max(epoch_idx / (self.total_epochs - 1), 0.0), 1.0)
        if self.schedule == "linear":
            return self.sim_weight_start + (self.sim_weight_end - self.sim_weight_start) * progress
        if self.schedule == "exp":
            return self.sim_weight_end + (self.sim_weight_start - self.sim_weight_end) * np.exp(-self.sim_weight_exp_k * progress)
        return self.sim_weight_start

    def __iter__(self):
        sim_w = self._sim_weight_at(self.epoch_idx)
        source_multiplier = np.where(self.source_flags, sim_w, 1.0)
        weights = self.base_weights * source_multiplier
        weights = np.clip(weights, 1e-12, None)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch_idx)
        indices = torch.multinomial(torch.as_tensor(weights, dtype=torch.double), self.num_samples, replacement=True, generator=g)
        self.epoch_idx += 1
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FNO_PHYSICAL_PARAM_DIM = 4


def _reset_params_for_normalize(samples: list[dict]) -> None:
    for sample in samples:
        params = list(sample.get("params", []))
        if len(params) < _FNO_PHYSICAL_PARAM_DIM:
            raise ValueError(f"Sample {sample.get('file_path', '?')} has {len(params)} params; expected at least {_FNO_PHYSICAL_PARAM_DIM}.")
        sample["params"] = params[:_FNO_PHYSICAL_PARAM_DIM]


def _prepare_holdout_subset(raw_samples: list[dict], max_p: float, min_p: float, denoise_kw: dict) -> list[dict]:
    work = copy.deepcopy(raw_samples)
    _reset_params_for_normalize(work)
    return prepare_test_holdout(work, max_p, min_p, **denoise_kw)


# ---------------------------------------------------------------------------
# Simulation-data normalisation (same as data_read.py)
# ---------------------------------------------------------------------------

_PHYS_PARAM_COLS = ["ambient_density", "ambient_temperature", "injection_pressure", "nozzle_radius"]


def normalize_phys_params_like_experiment(raw_params: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_params, dtype=np.float64).copy()
    x[:, 0] = (x[:, 0] - 1.93e-2) / 0.0134
    x[:, 1] = (x[:, 1] - 600.0) / 577.0
    x[:, 2] = (x[:, 2] - 100.0) / 84.25
    x[:, 3] = (x[:, 3] - 0.009) / 0.0055
    return x


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO with simulation-data augmentation.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split_dir", type=str, default="new/split")
    parser.add_argument("--class_path", type=str, default="model_weight/vae_model/vae_classify.pth")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_path", type=str, default="Final_model_fno_sim_dynw_2.pth")
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_modes", type=str, default="16")
    parser.add_argument("--num_layer", type=int, default=6)
    parser.add_argument("--hidden_channels", type=int, default=112)
    parser.add_argument("--num_frq", type=int, default=8)
    parser.add_argument("--encoder_layer_sizes", type=int, nargs="+", default=[5, 16, 8])
    parser.add_argument("--decoder_layer_sizes", type=int, nargs="+", default=[8, 16, 5])
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=120)
    parser.add_argument("--time_loss_weight", type=float, default=0.12)
    parser.add_argument("--fno_mono_weight", type=float, default=0.05)
    parser.add_argument("--denoise_method", type=str, default="rule", choices=["fit", "filter", "hybrid", "rule"])
    parser.add_argument("--sim_denoise_method", type=str, default="", choices=["", "fit", "filter", "hybrid", "rule"])
    parser.add_argument("--rule_fallback_mode", type=str, default="filter", choices=["fit", "filter", "hybrid"])
    parser.add_argument("--rule_json", type=str, default="new/filter_effect_rules.json")
    parser.add_argument("--sim_rule_json", type=str, default="new/sim_filter_effect_rules.json")
    parser.add_argument("--fit_resolution", type=int, default=128)
    parser.add_argument("--filter_type", type=str, default="savgol", choices=["linear", "savgol", "moving_avg"])
    parser.add_argument("--filter_window", type=int, default=15)
    parser.add_argument("--filter_polyorder", type=int, default=1)
    parser.add_argument("--default_tail_policy", type=str, default="auto_drop_abnormal_tail")
    parser.add_argument("--fit_only_if_short_ms", type=float, default=1.0)
    parser.add_argument("--splice_time_ms", type=float, default=0.5)
    parser.add_argument("--blend_width_ms", type=float, default=0.12)
    parser.add_argument("--post_blend_ma_window", type=int, default=11)
    parser.add_argument("--plateau_flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plateau_slope_eps", type=float, default=0.15)
    parser.add_argument("--plateau_min_duration_ms", type=float, default=0.15)
    parser.add_argument("--plateau_blend_ms", type=float, default=0.12)
    parser.add_argument("--skip_time_augment", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_name", type=str, default="fno_sim_dynw_2")
    parser.add_argument("--augment_factor", type=int, default=1, choices=[1, 3])
    parser.add_argument("--augment_windows", type=float, nargs="+", default=[0.3, 0.6])
    parser.add_argument("--sim_data_dir", type=str, default="new/random_penetration")
    parser.add_argument("--sim_add_count", type=int, default=300)
    parser.add_argument("--sim_sample_with_replacement", action="store_true")
    parser.add_argument("--sim_sample_weight", type=float, default=1.0)
    parser.add_argument("--sim_weight_schedule", type=str, default="exp", choices=["constant", "linear", "exp"])
    parser.add_argument("--sim_weight_start", type=float, default=0.2)
    parser.add_argument("--sim_weight_end", type=float, default=0.001)
    parser.add_argument("--sim_weight_exp_k", type=float, default=5.0)
    parser.add_argument("--report_path", type=str, default="new/exp2/output/sim_aug_report.json")
    parser.add_argument("--scale_out_path", type=str, default="new/exp2/output/scale_params_sim.json")

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a JSON object.")
        parser.set_defaults(**cfg)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    args.n_modes = parse_n_modes(args.n_modes)

    set_seed(args.seed, deterministic=True)
    os.makedirs(args.save_dir, exist_ok=True)

    default_tail_policy = "" if str(getattr(args, "default_tail_policy", "")).lower() == "none" else str(getattr(args, "default_tail_policy", ""))
    sample_rule_map = _load_rule_map(args.rule_json)
    sim_rule_map = _load_rule_map(args.sim_rule_json)
    merged_rule_map = load_merged_rule_map(args.rule_json, args.sim_rule_json)

    project_root = Path(__file__).resolve().parent.parent
    split_dir = Path(args.split_dir)
    _fit_res = int(getattr(args, "fit_resolution", 128))

    # ---- load split ----
    with open(split_dir / "split_config.json", encoding="utf-8") as f:
        split_cfg = json.load(f)
    train_paths = [p.replace("\\", "/") for p in split_cfg["train_files"]]
    val_paths = [p.replace("\\", "/") for p in split_cfg["val_files"]]
    base = str(project_root / "new") if train_paths[0].startswith("data_raw") else str(project_root)
    train_raw = process_file_list(base, train_paths)
    val_raw = process_file_list(base, val_paths)

    # ---- load simulation samples ----
    sim_data_dir = Path(args.sim_data_dir)
    if not sim_data_dir.is_absolute():
        sim_data_dir = project_root / sim_data_dir
    print(f"[sim] Loading simulation samples from {sim_data_dir}")
    sim_samples = process_folder(str(sim_data_dir))
    sim_samples = sorted(sim_samples, key=lambda s: str(s.get("file_path", "")))
    print(f"[sim] {len(sim_samples)} simulation samples available")

    # ---- select sim samples ----
    n_sim_select = min(args.sim_add_count, len(sim_samples))
    n_exp_raw = len(train_raw)
    sim_indices: list[int] = []
    if n_sim_select > 0:
        rng = np.random.RandomState(args.seed)
        if args.sim_sample_with_replacement or n_sim_select < len(sim_samples):
            sim_indices = sorted(rng.choice(len(sim_samples), size=n_sim_select, replace=args.sim_sample_with_replacement).tolist())
        else:
            sim_indices = list(range(n_sim_select))
        print(f"[sim] Selected {len(sim_indices)} simulation samples for training injection")

    train_raw_extended = copy.deepcopy(train_raw)
    sim_raw_params: list[list[float]] = []
    for idx in sim_indices:
        s = copy.deepcopy(sim_samples[idx])
        s["is_sim"] = True
        # Tag source so rule resolution works
        old_fp = s.get("file_path", "")
        s["file_path"] = f"sim_augmented::{old_fp}"
        # Extract raw params before normalisation overwrites
        raw_p = list(s.get("params", []))[:4]
        sim_raw_params.append(raw_p)
        train_raw_extended.append(s)

    print(f"[data] Extended training set: {len(train_raw_extended)} samples ({n_exp_raw} exp + {len(sim_indices)} sim)")

    # ---- preprocess ----
    train_fitted, max_p, min_p = fit_and_augment_data(
        train_raw_extended,
        denoise_method=args.denoise_method, filter_type=args.filter_type,
        filter_window=args.filter_window, filter_polyorder=args.filter_polyorder,
        sample_rule_map=merged_rule_map, default_tail_policy=default_tail_policy,
        fit_only_if_short_ms=args.fit_only_if_short_ms,
        splice_time_ms=args.splice_time_ms, blend_width_ms=args.blend_width_ms,
        post_blend_ma_window=args.post_blend_ma_window,
        plateau_flatten=args.plateau_flatten, plateau_slope_eps=args.plateau_slope_eps,
        plateau_min_duration_ms=args.plateau_min_duration_ms,
        plateau_blend_ms=args.plateau_blend_ms,
        resolution=_fit_res, rule_fallback=args.rule_fallback_mode,
        sim_denoise_method=args.sim_denoise_method,
    )

    n_train_before_enhance = len(train_fitted)
    if getattr(args, "skip_time_augment", False):
        train_enhanced = train_fitted
    else:
        train_enhanced = data_enhance(train_fitted, max_p, min_p, args.augment_factor, args.augment_windows, resolution=_fit_res)

    # Rebuild source flags matching data_enhance append order
    source_flags: list[bool] = []
    for i in range(n_train_before_enhance):
        source_flags.append(i >= n_exp_raw)
    for i in range(n_train_before_enhance):
        if len(train_fitted[i].get("time", [])) > 0 and train_fitted[i]["time"][-1] >= 1.5:
            for _ in args.augment_windows:
                for _ in range(args.augment_factor):
                    source_flags.append(i >= n_exp_raw)

    train_subset = normalize_time(train_enhanced)
    n_sim_total = sum(source_flags)
    print(f"[data] After enhance: {len(train_subset)} samples ({len(train_subset) - n_sim_total} exp + {n_sim_total} sim)")

    # ---- validation set (pure experimental) ----
    denoise_kw = dict(
        denoise_method=args.denoise_method, filter_type=args.filter_type,
        filter_window=args.filter_window, filter_polyorder=args.filter_polyorder,
        sample_rule_map=sample_rule_map, default_tail_policy=default_tail_policy,
        fit_only_if_short_ms=args.fit_only_if_short_ms,
        splice_time_ms=args.splice_time_ms, blend_width_ms=args.blend_width_ms,
        post_blend_ma_window=args.post_blend_ma_window,
        plateau_flatten=args.plateau_flatten, plateau_slope_eps=args.plateau_slope_eps,
        plateau_min_duration_ms=args.plateau_min_duration_ms,
        plateau_blend_ms=args.plateau_blend_ms, rule_fallback=args.rule_fallback_mode,
        sim_denoise_method=args.sim_denoise_method,
    )
    val_subset = _prepare_holdout_subset(val_raw, max_p, min_p, {**denoise_kw, "resolution": _fit_res})

    # ---- dataloaders ----
    sample_weights = compute_sample_weights(train_subset, num_bins=getattr(args, "augment_num_bins", 10))
    sim_w_start = args.sim_weight_start if args.sim_weight_start is not None else args.sim_sample_weight
    sim_w_end = args.sim_weight_end if args.sim_weight_end is not None else sim_w_start

    if args.sim_weight_schedule == "constant":
        src_mult = np.where(np.array(source_flags[:len(train_subset)], dtype=bool), sim_w_start, 1.0)
        weights = sample_weights * src_mult
        sampler = torch.utils.data.WeightedRandomSampler(
            [float(w) for w in weights], num_samples=len(train_subset), replacement=True,
        )
        train_loader = DataLoader(SpraySet(train_subset), batch_size=args.batch_size, sampler=sampler)
    else:
        sampler = DynamicSourceWeightedSampler(
            base_weights=sample_weights,
            source_flags=np.array(source_flags[:len(train_subset)], dtype=bool),
            num_samples=len(train_subset),
            total_epochs=args.epochs,
            schedule=args.sim_weight_schedule,
            sim_weight_start=sim_w_start,
            sim_weight_end=sim_w_end,
            sim_weight_exp_k=args.sim_weight_exp_k,
            seed=args.seed,
        )
        train_loader = DataLoader(SpraySet(train_subset), batch_size=args.batch_size, sampler=sampler)

    val_loader = DataLoader(SpraySet(val_subset), batch_size=args.batch_size, shuffle=False)

    print(f"[split] Train: {len(train_subset)} ({n_exp_raw} exp + {len(sim_indices)} sim + aug), Val: {len(val_subset)}")

    # ---- train ----
    log_path = build_log_path(args.log_dir, args.save_path, args.log_name)

    model = train_validation_1d(
        train_loader, val_loader, args.epochs, args.lr, args.hidden_channels, args.num_frq,
        args.n_modes, args.num_layer,
        encoder_layer_sizes=args.encoder_layer_sizes,
        decoder_layer_sizes=args.decoder_layer_sizes,
        latent_size=args.latent_size,
        class_save_path=args.class_path, device=args.device,
        save_dir=args.save_dir, save_path=args.save_path,
        early_stop_patience=args.early_stop_patience,
        time_loss_weight=args.time_loss_weight,
        log_path=log_path, fno_mono_weight=args.fno_mono_weight,
    )

    # ---- holdout evaluation ----
    if (split_dir / "test_files.txt").exists():
        evaluate_split_holdout_after_train(
            args, max_p=max_p, min_p=min_p, split_dir=split_dir, project_root=project_root,
        )
