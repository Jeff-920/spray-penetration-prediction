"""One-click inference on 40 holdout test samples.

Usage:
    cd spray_penetration_prediction
    python run_inference.py              # experimental-only model
    python run_inference.py --weight fused  # mixed-data model
"""
import sys
sys.path.insert(0, ".")

from inference.predictor import run_inference

CONFIGS = {
    "exp": {
        "weight_path": "model_weight/fno_model/trained_with_experimental_data.pth",
        "n_modes": 20,
    },
    "fused": {
        "weight_path": "model_weight/fno_model/trained_with_fused_data.pth",
        "n_modes": 16,
    },
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=str, default="exp", choices=["exp", "fused"])
args = parser.parse_args()

cfg = CONFIGS[args.weight]
output_dir = f"fig/{args.weight}"

print(f"\n{'='*60}")
print(f"Model: {'Experimental-only' if args.weight == 'exp' else 'Experimental + Simulation'}")
print(f"Weights: {cfg['weight_path']}")
print(f"Output:  {output_dir}/")
print(f"{'='*60}\n")

metrics, scatter_png = run_inference(
    test_data_dir="data/test_data",
    weight_path=cfg["weight_path"],
    class_path="model_weight/vae_model/vae_classify.pth",
    output_dir=output_dir,
    denoise_method="rule",
    rule_json="config/test_filter_rules.json",
    resolution=128,
    batch_size=8,
    hidden_channels=112,
    num_frq=8,
    n_modes=cfg["n_modes"],
    num_layer=6,
)

print(f"\nDone. Curves saved to {output_dir}/, scatter: {scatter_png}")
