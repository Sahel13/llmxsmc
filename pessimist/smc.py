import torch
from typing import List, Optional, Tuple
from collections.abc import Callable
from transformers import PreTrainedModel, PreTrainedTokenizer


def sampler(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_fn: Callable,
    prompt: str,
    num_particles: int,
    tempering: float,
    num_tokens: int,
    device: str,
) -> Tuple[List[str], torch.Tensor]:
    """Samples from the model using the provided parameters."""
    # Initialize particles
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    input_ids = torch.tile(input_ids, (num_particles, 1))

    log_weights = torch.zeros(num_particles, dtype=torch.float32, device=device)
    weights = torch.softmax(log_weights, dim=0)
    resampling_indices = torch.arange(num_particles, dtype=torch.int32, device=device)

    for _ in range(num_tokens):
        # Resample
        if ess(log_weights) < 0.75 * num_particles:
            resampling_indices = systematic_resampling(weights)
            input_ids = input_ids[resampling_indices]
            log_weights = torch.zeros(num_particles, dtype=torch.float32, device=device)

        # Mutate
        input_ids = mutate_fn(model, input_ids)

        # Reweight
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        log_weights += log_potential_fn(reward_fn, texts, tempering)
        weights = torch.softmax(log_weights, dim=0)

    weights = torch.softmax(log_weights, dim=0)
    return texts, weights


def mutate_fn(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Mutates the input_ids by sampling the next tokens from the model."""
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat([input_ids, next_tokens], dim=-1)
    return output_ids


def log_potential_fn(
    reward_fn: Callable, texts: List[str], tempering: float = 1.0
) -> torch.Tensor:
    """The log potential function that computes the reward for each text."""
    rewards = reward_fn(texts)
    return rewards * tempering


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
