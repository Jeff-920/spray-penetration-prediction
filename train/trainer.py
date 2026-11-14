"""Pure experimental-data FNO training entry point.

Replicates the original ``final_train.py --mode train --use_split`` pipeline.
Uses model definitions from ``model.fno`` / ``model.vae`` and the pre-trained
VAEClassify from ``train.classifier``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.processor import (
    SpraySet,
    compute_sample_weights,
    data_enhance,
    fit_and_augment_data,
    normalize_time,
    prepare_test_holdout,
    _load_rule_map,
)
from data.reader import process_file_list, process_folder
from model.fno import SprayModel, set_seed
from inference.predictor import (
    compute_batch_curve_metrics,
    evaluate_split_holdout_after_train,
    print_test_metrics,
    run_holdout_test,
    safe_state_dict,
    torch_load_compat,
)
from train.classifier import VAEClassify


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def parse_n_modes(value) -> int | tuple[int, ...]:
    """Parse FNO n_modes from int/str/tuple."""
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        nums = [int(v) for v in value]
        return nums[0] if len(nums) == 1 else tuple(nums)
    if value is None:
        return 16
    s = str(value).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return 16
    nums = [int(p) for p in parts]
    return nums[0] if len(nums) == 1 else tuple(nums)


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor | None = None,
    alpha: float = 0.0,
    delta: float = 0.03,
) -> torch.Tensor:
    """SmoothL1(Huber) + optional time weight:  weight = 1 + alpha * t."""
    loss = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    if t is not None and alpha > 0:
        weights = 1.0 + alpha * t
        loss = loss * weights
    return loss.mean()


def R_2(prediction: torch.Tensor, penetration: torch.Tensor) -> float:
    """Batch-wise R² (flattened across time)."""
    prediction_np = prediction.detach().cpu().numpy()
    penetration_np = penetration.detach().cpu().numpy()
    prediction_flat = prediction_np.reshape(-1)
    penetration_flat = penetration_np.reshape(-1)
    return float(r2_score(penetration_flat.reshape(-1, 1), prediction_flat.reshape(-1, 1)))


def build_log_path(log_dir: str, save_path: str = "", log_name: str = "") -> str:
    os.makedirs(log_dir, exist_ok=True)
    if log_name:
        filename = log_name if log_name.endswith(".csv") else f"{log_name}.csv"
    else:
        base = Path(save_path).stem if save_path else "fno"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fno_{base}_{ts}.csv"
    return os.path.join(log_dir, filename)


def init_epoch_log(log_path: str) -> None:
    headers = ["epoch", "train_loss", "test_loss", "test_r2", "lr", "best_r2", "best_epoch"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(headers)


def append_epoch_log(
    log_path: str, epoch: int, train_loss: float, test_loss: float,
    test_r2: float, lr: float, best_r2: float, best_epoch: int,
) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([epoch, train_loss, test_loss, test_r2, lr, best_r2, best_epoch])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_validation_1d(
    train_set: DataLoader,
    test_set: DataLoader,
    epochs: int,
    lr: float,
    hidden_channels: int,
    num_frq: int,
    n_modes: tuple | int,
    num_layer: int,
    encoder_layer_sizes: list,
    decoder_layer_sizes: list,
    latent_size: int,
    class_save_path: str,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    save_path: str = "",
    early_stop_patience: int = 50,
    time_loss_weight: float = 0.5,
    log_path: str = "",
    fno_mono_weight: float = 0.05,
):
    """Train the FNO SprayModel with a frozen VAEClassify providing evaporation probability."""

    model = SprayModel(
        hidden_channels, num_frq, n_modes, num_layer,
        encoder_layer_sizes, decoder_layer_sizes, latent_size, device=device,
    )
    class_model = VAEClassify().to(device)
    checkpoint = torch.load(class_save_path, map_location=device)
    class_model.load_state_dict(safe_state_dict(checkpoint))
    class_model.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=20, min_lr=1e-6)
    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")
    model.to(device_obj)

    max_r2 = 0.0
    best_epoch = 0
    save_path_fno = os.path.join(save_dir, save_path if save_path else "Spray_model.pth")
    total_train_iters = 0
    total_train_time_s = 0.0

    if log_path:
        init_epoch_log(log_path)
        print(f"[log] Training log -> {log_path}")

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0
        train_batch_num = 0
        epoch_train_start = time.perf_counter()
        pbar = tqdm(train_set, total=len(train_set), desc=f"Epoch {epoch + 1}/{epochs}[Training]")

        for times, params, penetration in pbar:
            times, params, penetration = times.to(device_obj), params.to(device_obj), penetration.to(device_obj)
            raw_params = params[:, :4]

            with torch.no_grad():
                _, _, _, class_logit = class_model(raw_params)
                pred_class = torch.sigmoid(class_logit).detach()

            prediction = model(params, times, pred_class)
            opt.zero_grad()

            loss_pred = weighted_huber_loss(prediction, penetration, times, alpha=time_loss_weight, delta=0.03)
            mono_penalty = F.relu(-(prediction[:, 1:] - prediction[:, :-1])).mean()
            total_loss = loss_pred + fno_mono_weight * mono_penalty

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_total_loss += total_loss.item()
            train_batch_num += 1
            pbar.set_postfix({"Batch Loss": f"{total_loss.item():.6f}"})

        epoch_train_elapsed = time.perf_counter() - epoch_train_start
        total_train_iters += train_batch_num
        total_train_time_s += max(epoch_train_elapsed, 1e-12)
        train_avg_loss = train_total_loss / train_batch_num if train_batch_num > 0 else 0.0

        # --- validation ---
        model.eval()
        test_total_loss = 0.0
        test_total_r2 = 0.0
        test_batch_num = 0

        with torch.no_grad():
            for times, params, penetration in test_set:
                times, params, penetration = times.to(device_obj), params.to(device_obj), penetration.to(device_obj)
                raw_params = params[:, :4]
                _, _, _, class_logit = class_model(raw_params)
                pred_class = torch.sigmoid(class_logit).detach()
                prediction = model(params, times, pred_class)

                loss_pred = weighted_huber_loss(prediction, penetration, times, alpha=time_loss_weight, delta=0.03)
                mono_penalty = F.relu(-(prediction[:, 1:] - prediction[:, :-1])).mean()
                total_loss = loss_pred + fno_mono_weight * mono_penalty
                batch_r2 = R_2(prediction, penetration)

                test_total_loss += total_loss.item()
                test_total_r2 += batch_r2
                test_batch_num += 1

            test_avg_loss = test_total_loss / test_batch_num if test_batch_num > 0 else 0.0
            test_avg_r2 = test_total_r2 / test_batch_num if test_batch_num > 0 else 0.0
            tqdm.write(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_avg_loss:.6f} | Test Loss: {test_avg_loss:.6f} | Test R2: {test_avg_r2:.6f}")

        try:
            scheduler.step(test_avg_loss)
        except Exception as e:
            print(f"Scheduler step error: {e}")

        if test_avg_r2 > 0 and test_avg_r2 >= max_r2:
            max_r2 = test_avg_r2
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path_fno)
            tqdm.write(f"Model saved to {save_path_fno} (R2: {test_avg_r2:.6f}, Best R2: {max_r2:.6f})")

        if log_path:
            cur_lr = opt.param_groups[0]["lr"]
            append_epoch_log(log_path, epoch + 1, train_avg_loss, test_avg_loss, test_avg_r2, cur_lr, max_r2, best_epoch)

        if early_stop_patience > 0 and best_epoch > 0 and (epoch + 1 - best_epoch) >= early_stop_patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
            break

    if os.path.exists(save_path_fno):
        ckpt = torch_load_compat(save_path_fno, map_location=device)
        model.load_state_dict(safe_state_dict(ckpt))

    avg_its = total_train_iters / total_train_time_s if total_train_time_s > 0 else 0.0
    print(f"[train] Average it/s: {avg_its:.2f}")
    print(f"The model has been saved to {save_dir}/{save_path or 'Spray_model.pth'}, with best R2: {max_r2:.6f}")

    try:
        stem = Path(save_path).stem if save_path else "Spray_model"
        stats_path = os.path.join(save_dir, f"{stem}_train_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {"average_speed_its": float(avg_its), "best_val_r2": float(max_r2), "best_epoch": int(best_epoch)},
                f, indent=2,
            )
        print(f"[train] Wrote {stats_path}")
    except Exception as ex:
        print(f"[train] Warning: could not write train stats json: {ex}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO on pure experimental data.")
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_path", type=str, default="Final_model_fno_exp_r128_1.pth")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--n_modes", type=str, default="20")
    parser.add_argument("--num_layer", type=int, default=6)
    parser.add_argument("--hidden_channels", type=int, default=112)
    parser.add_argument("--encoder_layer_sizes", type=int, nargs="+", default=[5, 16, 8])
    parser.add_argument("--decoder_layer_sizes", type=int, nargs="+", default=[8, 16, 5])
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--class_path", type=str, default="model_weight/vae_model/vae_classify.pth")
    parser.add_argument("--num_frq", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=120)
    parser.add_argument("--time_loss_weight", type=float, default=0.12)
    parser.add_argument("--fno_mono_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_name", type=str, default="fno_exp_r128_1")
    parser.add_argument("--denoise_method", type=str, default="rule", choices=["fit", "filter", "hybrid", "rule"])
    parser.add_argument("--rule_fallback_mode", type=str, default="filter", choices=["fit", "filter", "hybrid"])
    parser.add_argument("--filter_type", type=str, default="savgol", choices=["linear", "savgol", "moving_avg"])
    parser.add_argument("--filter_window", type=int, default=15)
    parser.add_argument("--filter_polyorder", type=int, default=1)
    parser.add_argument("--rule_json", type=str, default="new/filter_effect_rules.json")
    parser.add_argument("--default_tail_policy", type=str, default="auto_drop_abnormal_tail")
    parser.add_argument("--fit_only_if_short_ms", type=float, default=1.0)
    parser.add_argument("--splice_time_ms", type=float, default=0.5)
    parser.add_argument("--blend_width_ms", type=float, default=0.12)
    parser.add_argument("--post_blend_ma_window", type=int, default=11)
    parser.add_argument("--plateau_flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plateau_slope_eps", type=float, default=0.15)
    parser.add_argument("--plateau_min_duration_ms", type=float, default=0.15)
    parser.add_argument("--plateau_blend_ms", type=float, default=0.12)
    parser.add_argument("--use_split", action="store_true", default=True)
    parser.add_argument("--split_dir", type=str, default="new/split")
    parser.add_argument("--fit_resolution", type=int, default=128)
    parser.add_argument("--augment_factor", type=int, default=1, choices=[1, 3])
    parser.add_argument("--augment_windows", type=float, nargs="+", default=[0.3, 0.6])
    parser.add_argument("--skip_time_augment", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--augment_num_bins", type=int, default=10)
    parser.add_argument("--augment_min_weight", type=float, default=0.7)
    parser.add_argument("--augment_max_weight", type=float, default=2.0)
    parser.add_argument("--test_path", type=str, default="test/txt_s")
    parser.add_argument("--test_fig_path", type=str, default="test/fig_s")
    return parser.parse_args()


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    args.n_modes = parse_n_modes(args.n_modes)
    set_seed(int(getattr(args, "seed", 42)))
    sample_rule_map = _load_rule_map(getattr(args, "rule_json", ""))

    project_root = Path(__file__).resolve().parent.parent
    split_dir = Path(args.split_dir)
    use_split = getattr(args, "use_split", False)

    if use_split and (split_dir / "split_config.json").exists():
        with open(split_dir / "split_config.json", encoding="utf-8") as f:
            split_cfg = json.load(f)
        train_paths = [p.replace("\\", "/") for p in split_cfg["train_files"]]
        val_paths = [p.replace("\\", "/") for p in split_cfg["val_files"]]
        base = str(project_root / "new") if train_paths[0].startswith("data_raw") else str(project_root)
        train_raw = process_file_list(base, train_paths)
        val_raw = process_file_list(base, val_paths)
        print(f"[split] Train(raw): {len(train_raw)}, Val(raw): {len(val_raw)}, 40 holdout test")

        _fit_res = int(getattr(args, "fit_resolution", 128))
        train_fitted, max_p, min_p = fit_and_augment_data(
            train_raw,
            denoise_method=args.denoise_method,
            filter_type=args.filter_type,
            filter_window=args.filter_window,
            filter_polyorder=args.filter_polyorder,
            sample_rule_map=sample_rule_map,
            default_tail_policy=args.default_tail_policy,
            fit_only_if_short_ms=args.fit_only_if_short_ms,
            splice_time_ms=args.splice_time_ms,
            blend_width_ms=args.blend_width_ms,
            post_blend_ma_window=args.post_blend_ma_window,
            plateau_flatten=args.plateau_flatten,
            plateau_slope_eps=args.plateau_slope_eps,
            plateau_min_duration_ms=args.plateau_min_duration_ms,
            plateau_blend_ms=args.plateau_blend_ms,
            resolution=_fit_res,
            rule_fallback=args.rule_fallback_mode,
        )

        n_train_before_enhance = len(train_fitted)
        if getattr(args, "skip_time_augment", False):
            train_enhanced = train_fitted
        else:
            train_enhanced = data_enhance(train_fitted, max_p, min_p, args.augment_factor, args.augment_windows, resolution=_fit_res)
        train_subset = normalize_time(train_enhanced)

        val_subset = prepare_test_holdout(
            val_raw, max_p, min_p, resolution=_fit_res,
            denoise_method=args.denoise_method, filter_type=args.filter_type,
            filter_window=args.filter_window, filter_polyorder=args.filter_polyorder,
            sample_rule_map=sample_rule_map, default_tail_policy=args.default_tail_policy,
            fit_only_if_short_ms=args.fit_only_if_short_ms,
            splice_time_ms=args.splice_time_ms, blend_width_ms=args.blend_width_ms,
            post_blend_ma_window=args.post_blend_ma_window,
            plateau_flatten=args.plateau_flatten, plateau_slope_eps=args.plateau_slope_eps,
            plateau_min_duration_ms=args.plateau_min_duration_ms,
            plateau_blend_ms=args.plateau_blend_ms, rule_fallback=args.rule_fallback_mode,
        )

        print(f"[split] Train: {len(train_subset)} (from {n_train_before_enhance}), Val: {len(val_subset)}")

        if getattr(args, "use_weighted_sampler", False):
            train_weights = compute_sample_weights(
                train_subset,
                num_bins=getattr(args, "augment_num_bins", 10),
                min_weight=getattr(args, "augment_min_weight", 0.7),
                max_weight=getattr(args, "augment_max_weight", 2.0),
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                [float(w) for w in train_weights], num_samples=len(train_subset), replacement=True,
            )
            train_loader = DataLoader(SpraySet(train_subset), batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(
                SpraySet(train_subset), batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(int(getattr(args, "seed", 42))),
            )
        val_loader = DataLoader(SpraySet(val_subset), batch_size=args.batch_size, shuffle=False)

        os.makedirs(args.save_dir, exist_ok=True)
        log_path = build_log_path(args.log_dir, args.save_path, args.log_name)

        train_validation_1d(
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

        if (split_dir / "test_files.txt").exists():
            evaluate_split_holdout_after_train(
                args, max_p=max_p, min_p=min_p, split_dir=split_dir, project_root=project_root,
            )
    else:
        # fallback: random 80/20 split on whole data_raw
        datasets = process_folder(r"new\data_raw")
        _fit_res = int(getattr(args, "fit_resolution", 128))
        dataset_fitted, max_p, min_p = fit_and_augment_data(
            datasets,
            denoise_method=args.denoise_method, filter_type=args.filter_type,
            filter_window=args.filter_window, filter_polyorder=args.filter_polyorder,
            sample_rule_map=sample_rule_map,
            default_tail_policy=args.default_tail_policy,
            fit_only_if_short_ms=args.fit_only_if_short_ms,
            splice_time_ms=args.splice_time_ms, blend_width_ms=args.blend_width_ms,
            post_blend_ma_window=args.post_blend_ma_window,
            plateau_flatten=args.plateau_flatten, plateau_slope_eps=args.plateau_slope_eps,
            plateau_min_duration_ms=args.plateau_min_duration_ms,
            plateau_blend_ms=args.plateau_blend_ms,
            resolution=_fit_res, rule_fallback=args.rule_fallback_mode,
        )

        n_orig = len(dataset_fitted)
        if getattr(args, "skip_time_augment", False):
            dataset_enhance = dataset_fitted
        else:
            dataset_enhance = data_enhance(dataset_fitted, max_p, min_p, args.augment_factor, args.augment_windows, resolution=_fit_res)
            print(f"[data] Augmentation: {n_orig} -> {len(dataset_enhance)} samples")
        dataset_fit = normalize_time(dataset_enhance)

        seed = int(getattr(args, "seed", 42))
        generator = torch.Generator().manual_seed(seed)
        train_size = int(0.8 * len(dataset_fit))
        test_size = len(dataset_fit) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset_fit, [train_size, test_size], generator=generator,
        )

        if getattr(args, "use_weighted_sampler", False):
            all_weights = compute_sample_weights(
                dataset_fit, num_bins=getattr(args, "augment_num_bins", 10),
                min_weight=getattr(args, "augment_min_weight", 0.7),
                max_weight=getattr(args, "augment_max_weight", 2.0),
            )
            train_indices = train_dataset.indices if hasattr(train_dataset, "indices") else list(range(train_size))
            train_weights = [float(all_weights[i]) for i in train_indices]
            sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(train_indices), replacement=True)
            train_loader = DataLoader(SpraySet(train_dataset), batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(SpraySet(train_dataset), batch_size=args.batch_size, shuffle=True, generator=generator)
        test_loader = DataLoader(SpraySet(test_dataset), batch_size=args.batch_size, shuffle=False)

        os.makedirs(args.save_dir, exist_ok=True)
        log_path = build_log_path(args.log_dir, args.save_path, args.log_name)

        train_validation_1d(
            train_loader, test_loader, args.epochs, args.lr, args.hidden_channels, args.num_frq,
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
