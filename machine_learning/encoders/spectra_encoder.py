import torch.nn as nn
import torch

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_dim: int,
                 out_channels: int,
                 stride: int = 1):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        if (out_channels != in_channels) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels),

            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        return self.bottleneck(x) + self.downsample(x)


class FilmLayer(nn.Module):
    def __init__(self, feature_dim: int, meta_embed_dim: int):
        super().__init__()
        self.gamma_linear = nn.Linear(meta_embed_dim, feature_dim)
        self.beta_linear = nn.Linear(meta_embed_dim, feature_dim)
        self.batch1d = nn.BatchNorm1d(feature_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, meta_embed):
        gamma = self.gamma_linear(meta_embed).unsqueeze(-1)
        beta = self.beta_linear(meta_embed).unsqueeze(-1)
        return self.relu(
            self.batch1d(gamma * x + beta)
        )


class SpectraEncoder(nn.Module):
    def __init__(self, input_size: int, output_dim: int,
                 hidden_dims: list = [32, 64, 128, 256],
                 num_blocks_per_layer: list = [2, 2, 2, 2],
                 meta_embed_dim: int = 3,
                 ):
        super().__init__()

        self.meta_embed_dim = meta_embed_dim
        self.input_size = input_size
        in_channels = hidden_dims[0]

        self.init_compress = nn.Sequential(
            nn.Conv1d(1, in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for i, (out_channels, num_blocks) in enumerate(zip(hidden_dims, num_blocks_per_layer)):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(in_channels, out_channels, num_blocks, stride)
            self.layers.append(layer)
            film_layer = FilmLayer(out_channels, meta_embed_dim)
            self.film_layers.append(
                film_layer
            )
            in_channels = out_channels

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):

        layers = []
        print()
        layers.append(ResidualBlock1D(in_channels, in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, spec: torch.Tensor, meta_embed: torch.Tensor):
        """
        :param spec: torch.Tensor with shape [batch_shape, input_size]
        """
        x = spec.reshape(-1, self.input_size).unsqueeze(1)
        x = self.init_compress(x)
        for layer, film_layer in zip(self.layers, self.film_layers):
            x = layer(x)
            x = film_layer(x, meta_embed)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
