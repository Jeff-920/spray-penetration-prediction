import random
import torch
import torch.nn as nn
import numpy as np
from neuralop import FNO
from neuralop.layers.embeddings import SinusoidalEmbedding
import torch.nn.functional as F 


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SprayModel(nn.Module):
    def __init__(self, hidden_channels: int, num_frq: int, n_modes: tuple | int, num_layer: int, 
                 encoder_layer_sizes: list, decoder_layer_sizes: list, latent_size: int, device='cuda'):
        super().__init__()
        # params shape: (B, 5) = 4 physical params + normalized cutoff_time
        # pred_class shape: (B, 1) = evaporative probability in [0, 1]
        if isinstance(n_modes, int):
            n_modes = (n_modes, )

        self.const_encoder = SinusoidalEmbedding(
            in_channels=6, 
            num_frequencies=num_frq,
            embedding_type='transformer',
        ).to(device=device)

        self.time_encoder = SinusoidalEmbedding(
            in_channels=1,
            num_frequencies=num_frq,
            embedding_type='transformer',
        ).to(device=device)

        self.fno = FNO(
            n_modes=n_modes,
            in_channels=6 * num_frq * 2 + 1 * num_frq * 2,
            out_channels=1,
            hidden_channels=hidden_channels,
            n_layers=num_layer,
            non_linearity=F.gelu,
            domain_padding=0.05,
            fno_block_precision='full',
            channel_mlp_dropout=0.1,
            fno_skip='soft-gating',
        ).to(device=device)

    def forward(self, params, time, pred_class):
        # params shape: (batch_size, params_num)
        # time shape: (batch_size, sequence_len)
        # pred_class: (batch_size, 1)
        # pred_class should be a probability (sigmoid output) and detached from classifier graph
        params = torch.concat((params, pred_class), dim=-1)
        batch_size = time.shape[0]
        len_seq = time.shape[1]
        params_rep = params.unsqueeze(1).repeat(1, len_seq, 1)  # (B, L, 6)
        encode_params = self.const_encoder(params_rep)  # (B, L, param_embed_dim)
        time = time.unsqueeze(-1)  # shape (B, L, 1)
        encode_time = self.time_encoder(time) # shape (B, L, time_embed_dim)
        t_p_features = torch.concat((encode_time, encode_params), dim=-1)  # shape (B, L, time_embed_dim + param_embed_dim)
        t_p_features = t_p_features.permute(0, 2, 1)  # shape (B, C, L)
        out = self.fno(t_p_features)  # shape (B, 1, L)
        return out.squeeze(1)  # shape (B, L)