from pessimist.smc_sampler import generate
from pessimist.reward import tragicness_score

import argparse
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer


def load_gpt2_model():
    """Load the model and the tokenizer."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using GPT-2")
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt text to generate from",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum length of the generated text",
    )
    args = parser.parse_args()
    model, tokenizer = load_gpt2_model()

    generated_text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        num_particles=30,
        max_length=args.max_length,
        top_k=5,
        reward_fn=tragicness_score,
    )

    print(f"Completed (tragic) text: {generated_text}")
