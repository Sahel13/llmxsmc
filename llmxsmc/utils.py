from typing import Optional

import torch


def ess(log_weights: torch.Tensor) -> torch.Tensor:
    """Computes the effective sample size."""
    log_ess = 2 * torch.logsumexp(log_weights, 0) - torch.logsumexp(2 * log_weights, 0)
    return torch.exp(log_ess)


def systematic_resampling(
    weights: torch.Tensor, num_samples: Optional[int] = None
) -> torch.Tensor:
    """Perform systematic resampling of particles based on their weights."""
    device = weights.device
    n = weights.shape[0]
    if num_samples is None:
        num_samples = n

    u = torch.rand(1, device=device)
    cumsum = torch.cumsum(weights, dim=0)
    linspace = (
        torch.arange(num_samples, dtype=weights.dtype, device=device) + u
    ) / num_samples
    indices = torch.searchsorted(cumsum, linspace, out_int32=True)
    return torch.clamp(indices, max=n - 1)
