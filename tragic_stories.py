import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from llmxsmc import sampler

# === Command line arguments ===
parser = argparse.ArgumentParser(
    description="Generate tragic sentences using SMC-steering."
)
parser.add_argument(
    "--prompt", type=str, required=True, help="The initial prompt for text generation."
)
parser.add_argument(
    "--num_tokens", type=int, required=True, help="The number of tokens to generate."
)
parser.add_argument(
    "--seed", type=int, default=0, help="The random seed for reproducibility."
)
args = parser.parse_args()

torch.manual_seed(args.seed)

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
num_particles = 50
num_tokens = args.num_tokens
prompt = args.prompt

# === LLM setup ===
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").to(device)
model.eval()

# === Reward model ===
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0 if device == "cuda" else -1,
)
target_label = "sadness"


def reward_fn(texts: List[str]) -> torch.Tensor:
    """Computes the reward for each text based on the score of the target label."""
    results = classifier(texts)
    rewards = [
        next(r["score"] for r in result if r["label"].lower() == target_label)
        for result in results
    ]
    return torch.tensor(rewards).to(device)


# Run the sampler
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
input_ids = torch.tile(input_ids, (num_particles, 1))
outputs, weights = sampler(
    model,
    tokenizer,
    reward_fn,
    prompt,
    num_particles,
    1.0,
    num_tokens,
    device,
)

# Output the trajectory with the highest weight
best_idx = torch.argmax(weights)
best_output = outputs[best_idx]
print("\nModel output: ", best_output)
