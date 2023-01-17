from typing import Union
import math

import torch
import torch.nn as nn


class TripleSigmoid(nn.Module):
    """Triple-Sigmoid Activation Function.
    Paper: https://ieeexplore.ieee.org/document/9833503/
    """
    def __init__(
        self,
        w1: float = 0.005,
        w2: float = 0.1,
        w3: float = 0.001,
        alpha: float = 0,
        beta: float = 500,
        gamma: float = 0,
        delta: float = 1.5,
        dtype = torch.float32,
        device: Union[str, torch.device] = 'cpu',
    ):
        """Initialize Triple-Sigmoid function.

        Args:
            w1 (float, optional): w1. Defaults to 0.005.
            w2 (float, optional): w2. Defaults to 0.1.
            w3 (float, optional): w3. Defaults to 0.001.
            alpha (float, optional): alpha. Defaults to 0.
            beta (float, optional): beta. Defaults to 500.
            gamma (float, optional): gamma. Defaults to 0.
            delta (float, optional): delta. Defaults to 1.5.
            dtype (_type_, optional): dtype. Defaults to torch.float32.
            device (Union[str, torch.device], optional): device. Defaults to 'cpu'.
        """
        super().__init__()

        self.dtype = dtype
        self.device = torch.device(device)

        self.w1 = torch.tensor(w1, dtype=self.dtype, device=self.device)
        self.w2 = torch.tensor(w2, dtype=self.dtype, device=self.device)
        self.w3 = torch.tensor(w3, dtype=self.dtype, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)
        self.beta = torch.tensor(beta, dtype=self.dtype, device=self.device)

        self.b0 = torch.tensor(delta, dtype=self.dtype, device=self.device)
        self.b1 = torch.tensor((w2 * gamma + (w1 - w2) * alpha) / w1, dtype=self.dtype, device=self.device)
        self.b2 = torch.tensor(gamma)
        self.b3 = beta

        _temp = math.exp(-delta)
        self.t_beta = torch.tensor(1 / (1 + math.exp(-w2 * (beta - gamma) - delta)) - _temp / (1 + _temp), dtype=self.dtype, device=self.device)

    def _calc_part(self, h: torch.Tensor, w: float, b: float) -> torch.Tensor:
        """Calculate `e^(-w * (h - b) - b0)`.

        Args:
            h (torch.Tensor): h
            w (float): w
            b (float): b

        Returns:
            torch.Tensor: Output of `e^(-w * (h - b) - b0)`
        """
        return torch.exp(-w * (h - b) - self.b0)

    def eq_1_1(self, h: torch.Tensor) -> torch.Tensor:
        """Calculate equation 1 when h < alpha.

        Args:
            h (torch.Tensor): h

        Returns:
            torch.Tensor: Output of t(h)
        """
        return 1 / (1 + self._calc_part(h, self.w1, self.b1))

    def eq_1_2(self, h: torch.Tensor) -> torch.Tensor:
        """Calculate equation 2 when alpha <= h < beta.

        Args:
            h (torch.Tensor): h

        Returns:
            torch.Tensor: Output of t(h)
        """
        return 1 / (1 + self._calc_part(h, self.w2, self.b2))

    def eq_1_3(self, h: torch.Tensor) -> torch.Tensor:
        """Calculate equation 3 when beta <= h.

        Args:
            h (torch.Tensor): h

        Returns:
            torch.Tensor: Output of t(h)
        """
        _temp = self._calc_part(h, self.w3, self.b3)
        return self.t_beta + _temp / (1 + _temp)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Calculate Triple-Sigmoid.

        Args:
            h (torch.Tensor): Input

        Returns:
            torch.Tensor: Output of Triple-Sigmoid
        """
        h1 = torch.where(h < self.alpha, self.eq_1_1(h), 0.)
        h2 = torch.where(torch.logical_and(self.alpha <= h, h < self.beta), self.eq_1_2(h), 0.)
        h3 = torch.where(self.beta <= h, self.eq_1_3(h), 0.)

        return h1 + h2 + h3
