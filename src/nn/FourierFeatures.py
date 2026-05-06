import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Deterministic Fourier feature mapping.

    For each input dimension x, returns:
      [x, sin(2*pi*2^k*x), cos(2*pi*2^k*x)] for k = 0..num_bands-1
    """

    def __init__(self, input_dim: int, num_bands: int = 4, include_input: bool = True):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")
        self.input_dim = int(input_dim)
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)

        # Frequencies: 1,2,4,8,... in powers of two.
        bands = torch.tensor(
            [2.0 ** k for k in range(self.num_bands)], dtype=torch.float32
        )
        self.register_buffer("bands", bands, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + (2 * self.num_bands * self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")

        # x: [B, D], scaled: [B, D, K]
        scaled = x.unsqueeze(-1) * self.bands.view(1, 1, -1) * (2.0 * math.pi)
        sin_feats = torch.sin(scaled).reshape(x.shape[0], -1)
        cos_feats = torch.cos(scaled).reshape(x.shape[0], -1)

        outputs = []
        if self.include_input:
            outputs.append(x)
        outputs.extend([sin_feats, cos_feats])
        return torch.cat(outputs, dim=1)
