import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List
from pessimist import sampler
import argparse

# === Argument Parsing ===
parser = argparse.ArgumentParser(
    description="Generate tragic sentences using SMC-steering."
)
parser.add_argument(
    "--prompt", type=str, required=True, help="The initial prompt for text generation."
)
parser.add_argument(
    "--num_tokens", type=int, required=True, help="The number of tokens to generate."
)
args = parser.parse_args()

# Set the seed
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
num_particles = 50
num_tokens = args.num_tokens
prompt = args.prompt

# === GPT-2 setup ===
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
gpt2_model.eval()

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


# === Initialize particles ===
input_ids = gpt2_tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
input_ids = torch.tile(input_ids, (num_particles, 1))
outputs, weights = sampler(
    gpt2_model,
    gpt2_tokenizer,
    reward_fn,
    prompt,
    num_particles,
    1.0,
    num_tokens,
    device,
)

# Output the trajectory with the highest weight.
best_idx = torch.argmax(weights)
best_output = outputs[best_idx]
print("\nModel output: ", best_output)
