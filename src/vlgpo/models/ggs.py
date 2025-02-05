from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv1d(nn.Conv1d):
    """
    A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1d, with padding set automatically.

    Shape:
        Input: (N, L, in_channels)
        input_mask: (N, L, 1), optional
        Output: (N, L, out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        activations = {
            'swish': lambda x: x * torch.sigmoid(100.0 * x),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
            'relu': F.relu
        }
        self.act_fn = activations.get(activation)
        if self.act_fn is None:
            raise NotImplementedError(f"Activation '{activation}' is not implemented.")

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        return torch.max(x, dim=1)[0]


class BaseCNN(nn.Module):
    def __init__(self, n_tokens=20, kernel_size=5, input_size=256, dropout=0.0, 
                 make_one_hot=True, activation='relu', linear=True, **kwargs):
        super().__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size * 2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size * 2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self._make_one_hot = make_one_hot

    def forward(self, x):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        return self.decoder(x).squeeze(1)

    def forward_soft(self, x):
        x = torch.softmax(x, -1)
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        return self.decoder(x).squeeze(1)