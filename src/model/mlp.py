import torch.nn as nn


def build_mlp(input_size: int, output_size: int, n_layers: int, size: int) -> nn.Module:
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(input_size, size))
        layers.append(nn.ReLU())
        input_size = size

    layers.append(nn.Linear(input_size, output_size))

    model = nn.Sequential(*layers)

    return model
