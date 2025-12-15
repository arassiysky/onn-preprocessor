from __future__ import annotations

from dataclasses import dataclass
import torch


def thue_morse_bitparity(n: torch.Tensor) -> torch.Tensor:
    """
    Thue–Morse parity: parity(popcount(n)).
    Returns 0/1 tensor, same shape as n.
    """
    # popcount parity via XOR-folding
    x = n.clone()
    x ^= x >> 1
    x ^= x >> 2
    x = (x & 0x11111111) * 0x11111111
    parity = (x >> 28) & 1
    return parity.to(torch.int64)


def sierpinski_like(n: torch.Tensor) -> torch.Tensor:
    """
    Simple Sierpinski-like pattern from bitwise structure.
    Not the full triangle, but a deterministic fractal-ish mask:
      mask = 1 if (n & (n >> 1)) == 0 else 0
    """
    return (((n & (n >> 1)) == 0).to(torch.int64))


@dataclass
class L4HybridMask:
    """
    L4 = global/positional structure provider.

    Produces a deterministic positional mask m in {0,1} of length S.
    Then provides a weight w = 1 + alpha*m that can modulate feature magnitudes
    (or gating scores) without changing shapes.

    This is a safe first implementation:
    - deterministic
    - GPU-friendly
    - no dependence on token IDs (only positions)
    """

    alpha: float = 0.5  # how strongly to emphasize masked positions
    mix_tm: float = 0.5  # mixing weight for Thue–Morse
    mix_sp: float = 0.5  # mixing weight for Sierpinski-like

    def mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns m: (S,) in {0,1}, int64
        """
        idx = torch.arange(seq_len, device=device, dtype=torch.int64)

        tm = thue_morse_bitparity(idx)  # 0/1
        sp = sierpinski_like(idx)       # 0/1

        # Mix and threshold into {0,1}
        mix = self.mix_tm * tm.to(torch.float32) + self.mix_sp * sp.to(torch.float32)
        m = (mix >= 0.5).to(torch.int64)
        return m

    def weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns w: (S,) float32, where w = 1 + alpha*m.
        """
        m = self.mask(seq_len, device).to(torch.float32)
        w = 1.0 + float(self.alpha) * m
        return w