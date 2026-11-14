"""VAEClassify model: VAE + binary classifier for evaporative spray probability.

This classifier must be trained first; its output (spray probability) is then fed
as an additional input channel to the FNO model.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model.vae import Encoder, Decoder


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class Classification(nn.Module):
    def __init__(self, layer_sizes: list[int], class_num: int = 2):
        super().__init__()
        self.class_num = 1 if class_num == 2 else class_num
        self.encode = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.encode.add_module(f"L{i}", nn.Conv1d(in_size, out_size, kernel_size=1))
            self.encode.add_module(f"B{i}", nn.BatchNorm1d(out_size))
            self.encode.add_module(f"A{i}", nn.ReLU())
        self.class_layer = nn.Sequential(nn.Linear(layer_sizes[-1], self.class_num))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x = self.encode(x)
        return self.class_layer(x.squeeze(-1))


class VAEClassify(nn.Module):
    """Joint VAE + binary classifier.

    During training: samples latent z via reparameterisation.
    During eval:    uses deterministic mean.
    """

    def __init__(
        self,
        encoder_layer_sizes: list[int] | None = None,
        decoder_layer_sizes: list[int] | None = None,
        classification_size: list[int] | None = None,
        latent_size: int = 32,
        class_num: int = 2,
    ):
        super().__init__()
        if encoder_layer_sizes is None:
            encoder_layer_sizes = [4, 64, 32]
        if decoder_layer_sizes is None:
            decoder_layer_sizes = [32, 64, 32]
        if classification_size is None:
            classification_size = [32, 16]

        self.latent_size = latent_size
        classification_size = list(classification_size)
        classification_size.insert(0, latent_size)

        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, params_num=encoder_layer_sizes[0])
        self.classification = Classification(classification_size, class_num=class_num)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        means, log_var = self.encoder(x)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size], device=means.device)
            z = eps * std + means
        else:
            z = means
        recon_x = self.decoder(z)
        class_x = self.classification(z)
        return recon_x, means, log_var, class_x


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloader(feature_dim: int, batch_size: int):
    """Load labelled parameter data and split into train/val/test."""
    df = pd.read_csv(r"new\data\sample_params_standard.csv")
    feature_cols = ["ambient_density", "ambient_temperature", "injection_pressure", "nozzle_radius"]
    label_col = "if_evaporative"

    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    val_data = torch.tensor(X_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)

    print(f"Data split: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    train_loader, val_loader, _ = get_dataloader(args.feature_dim, args.batch_size)

    encoder_layers = [args.feature_dim] + args.encoder_hidden
    decoder_layers = [args.latent_dim] + args.decoder_hidden
    class_layers = args.class_hidden

    model = VAEClassify(encoder_layers, decoder_layers, class_layers, args.latent_dim, class_num=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss: list[float] = []
        train_preds: list[int] = []
        train_labels_list: list[int] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_x, means, log_var, class_logit = model(data)

            recon_loss = F.mse_loss(recon_x, data)
            kl_loss = -0.5 * torch.mean(1 + log_var - means.pow(2) - log_var.exp())
            class_loss = F.binary_cross_entropy_with_logits(class_logit, labels)
            total_loss = recon_loss + 0.1 * kl_loss + class_loss

            total_loss.backward()
            optimizer.step()

            train_loss.append(total_loss.item())
            preds = (torch.sigmoid(class_logit) > 0.5).int().cpu().numpy().flatten()
            train_preds.extend(preds.tolist())
            train_labels_list.extend(labels.int().cpu().numpy().flatten().tolist())
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        train_acc = np.mean(np.array(train_preds) == np.array(train_labels_list))

        # validation
        model.eval()
        val_loss: list[float] = []
        val_preds: list[int] = []
        val_labels_list: list[int] = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                recon_x, means, log_var, class_logit = model(data)
                recon_loss = F.mse_loss(recon_x, data)
                kl_loss = -0.5 * torch.mean(1 + log_var - means.pow(2) - log_var.exp())
                class_loss = F.binary_cross_entropy_with_logits(class_logit, labels)
                val_loss.append((recon_loss + 0.1 * kl_loss + class_loss).item())
                preds = (torch.sigmoid(class_logit) > 0.5).int().cpu().numpy().flatten()
                val_preds.extend(preds.tolist())
                val_labels_list.extend(labels.int().cpu().numpy().flatten().tolist())

        avg_val_loss = float(np.mean(val_loss))
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels_list))
        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {np.mean(train_loss):.4f} "
            f"Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            tqdm.write(f"  -> saved best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

    print(f"Training finished. Best val_loss={best_val_loss:.4f}, best val_acc={best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAEClassify for evaporative probability.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--feature_dim", type=int, default=4)
    parser.add_argument("--encoder_hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--decoder_hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--class_hidden", type=int, nargs="+", default=[16])
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="models/vae_classify.pth")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
