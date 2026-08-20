# Spray Penetration Prediction

The source code will be made publicly available upon acceptance of this paper.
Fourier Neural Operator (FNO) for spray tip penetration prediction. Two training modes: pure experimental data, and experimental + simulation mixed data.

## Repository Structure

```
├── model/                    # Model definitions
│   ├── fno.py                # SprayModel — FNO backbone
│   └── vae.py                # VAE / Encoder / Decoder
├── model_weight/             # Pre-trained weights
│   ├── fno_model/
│   │   ├── trained_with_experimental_data.pth   # Trained on experimental data only
│   │   └── trained_with_fused_data.pth          # Trained on experimental + simulation data
│   └── vae_model/
│       └── vae_classify.pth                     # VAE classifier
├── data/                     # Data processing
│   ├── reader.py             # Raw .dat/.txt file parser
│   ├── processor.py          # Denoising, curve fitting, normalisation, augmentation
│   └── test_data/            # Placeholder for test samples (not provided)
├── train/                    # Training
│   ├── classifier.py         # VAEClassify training
│   ├── trainer.py            # FNO training (experimental data only)
│   └── mixed_trainer.py      # FNO training (experimental + simulation)
├── inference/
│   └── predictor.py          # Inference and evaluation metrics
├── run_inference.py          # Inference entry point
└── requirements.txt
```

## Model

### Input

Four normalised physical parameters plus normalised cutoff time:

| Parameter | Normalisation |
|-----------|---------------|
| Ambient density ρ | `(ρ − 1.93×10⁻²) / 1.34×10⁻²` |
| Ambient temperature T | `(T − 600) / 577` |
| Injection pressure P | `(P − 100) / 84.25` |
| Nozzle radius r | `(r − 0.009) / 0.0055` |

The VAEClassify model provides an evaporation probability (sigmoid) as the 5th conditioning channel for the FNO.

### Architecture

- **VAEClassify**: VAE encoder → latent space → binary classifier head. Trained on labelled `sample_params_standard.csv` to predict whether a spray condition is evaporative.
- **SprayModel**: Sinusoidal embedding of physical params + time → FNO layers → penetration curve (128 points).

## Environment

```bash
pip install -r requirements.txt
```

## Training

> **Warning:** The `--save_dir` and `--save_path` defaults may point to `model_weight/`. Change them to avoid overwriting the provided pre-trained weights.

Training requires the raw experimental dataset and split configuration (not included in this repository).

### 1. VAEClassify (evaporation probability classifier)

Trains the VAE + classifier that provides the evaporation probability conditioning input to the FNO.

```bash
python train/classifier.py --device cuda --epochs 200 --batch_size 32 --lr 1e-3 --save_path model_weight/vae_model/vae_classify.pth
```

Requires `new/data/sample_params_standard.csv` with columns: `ambient_density`, `ambient_temperature`, `injection_pressure`, `nozzle_radius`, `if_evaporative`.

### 2. FNO — experimental data only

```bash
python train/trainer.py --use_split --split_dir new/split --class_path model_weight/vae_model/vae_classify.pth --save_dir checkpoints --save_path Final_model_fno_exp_r128_1.pth --epochs 800 --early_stop_patience 120 --batch_size 12 --lr 1.2e-4 --hidden_channels 112 --num_layer 6 --n_modes 20 --num_frq 8 --time_loss_weight 0.12 --fno_mono_weight 0.05 --denoise_method rule --rule_json new/filter_effect_rules.json --fit_resolution 128
```

Key hyperparameters: `n_modes=20`, `hidden_channels=112`, `num_layer=6`, `lr=1.2e-4`, `batch_size=12`, `time_loss_weight=0.12`, `fno_mono_weight=0.05`.

Requires `new/split/` (split_config.json with train/val/test splits), `new/data_raw/`, and `new/filter_effect_rules.json`.

### 3. FNO — experimental + simulation

```bash
python train/mixed_trainer.py --seed 42 --split_dir new/split --class_path model_weight/vae_model/vae_classify.pth --save_dir checkpoints --save_path Final_model_fno_sim_dynw_2.pth --epochs 800 --early_stop_patience 120 --batch_size 16 --lr 1.2e-4 --hidden_channels 112 --num_layer 6 --n_modes 16 --num_frq 8 --time_loss_weight 0.12 --sim_data_dir new/random_penetration --sim_add_count 300 --augment_factor 1 --augment_windows 0.3 0.6 --sim_weight_schedule exp --sim_weight_start 0.2 --sim_weight_end 0.001 --denoise_method rule --rule_json new/filter_effect_rules.json --sim_rule_json new/sim_filter_effect_rules.json --fit_resolution 128
```

Key additions over exp-only: `n_modes=16`, `sim_add_count=300`, `sim_weight_schedule=exp`, `sim_weight_start=0.2`, `sim_weight_end=0.001`. Simulation samples are injected into the training set only; validation and test sets remain pure experimental data.

Requires `new/random_penetration/` (simulation .dat files), `new/sim_filter_effect_rules.json`.

## Inference

Place test `.dat` files in `data/test_data/`, then run:

```bash
python run_inference.py --weight exp    # experimental-only model
python run_inference.py --weight fused  # mixed-data model
```

Or programmatically:

```python
from inference.predictor import run_inference

metrics, scatter = run_inference(
    test_data_dir="data/test_data",
    weight_path="model_weight/fno_model/trained_with_fused_data.pth",
    class_path="model_weight/vae_model/vae_classify.pth",
    denoise_method="fit",
    n_modes=16,
)
```

## Notes

This repository was restructured from the original research codebase with the assistance of AI. The inference pipeline is verified against the original results. Some redundancy remains in the code and will be cleaned up in a future revision.
