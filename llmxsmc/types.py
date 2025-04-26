from typing import List, Protocol

import torch


class RewardFn(Protocol):
    """A protocol for reward functions."""

    def __call__(self, texts: List[str]) -> torch.Tensor: ...
