import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, decoder_layer_sizes, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.params_num = encoder_layer_sizes[0]
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, params_num=encoder_layer_sizes[0])

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        means, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means
        recon_x = self.decoder(z)
    
        return recon_x, means, log_var


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(f"L{i}", nn.Conv1d(in_size, out_size, kernel_size=1))
            self.MLP.add_module(f"B{i}", nn.BatchNorm1d(out_size))
            self.MLP.add_module(f"A{i}", nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)
        x = x.squeeze(-1)
        return self.linear_means(x), self.linear_log_var(x)


class Decoder(nn.Module):
    def __init__(self, layer_sizes, params_num):
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(f"L{i}", nn.Conv1d(in_size, out_size, kernel_size=1))
            self.MLP.add_module(f"B{i}", nn.BatchNorm1d(out_size))
            self.MLP.add_module(f"A{i}", nn.ReLU())

        self.reconduct = nn.Linear(layer_sizes[-1], params_num)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.MLP(x)
        return self.reconduct(x.squeeze(-1))