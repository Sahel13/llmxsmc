import jax
from jax import random
import jax.numpy as jnp


def generate(
    model, tokenizer, prompt, num_particles=20, max_length=30, top_k=5, reward_fn=None
):
    def step(sequences, key):
        # 1. Get the logits for the next token.
        outputs = model(input_ids=sequences)
        logits = outputs.logits[:, -1, :]

        # 2. Top-k sampling
        topk_logits, topk_indices = jax.lax.top_k(logits, top_k)
        top_k_key, resampling_key = random.split(key)
        sampled_idx = random.categorical(top_k_key, topk_logits, axis=-1)
        next_tokens = jnp.take_along_axis(topk_indices, sampled_idx[:, None], axis=-1)

        # 3. Append next tokens to sequences
        sequences = jnp.concatenate([sequences, next_tokens], axis=1)

        # 4. Score sequences using the reward function.
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        scores = jnp.array([reward_fn(text) for text in texts])

        # 5. Update log weights and normalize
        log_weights = jnp.log(scores + 1e-6)
        weights = jax.nn.softmax(log_weights)

        # 6. Resample particles based on weights
        indices = random.choice(
            resampling_key, num_particles, shape=(num_particles,), p=weights
        )
        sequences = sequences[indices]
        return sequences

    input_ids = tokenizer(prompt, return_tensors="jax")["input_ids"]
    input_ids = jnp.tile(input_ids, (num_particles, 1))

    sequences = input_ids
    key = random.key(0)

    for _ in range(max_length):
        key, step_key = random.split(key)
        sequences = step(sequences, step_key)
        print("One step done")

    idx = random.choice(key, num_particles)
    return tokenizer.decode(sequences[idx], skip_special_tokens=True)
