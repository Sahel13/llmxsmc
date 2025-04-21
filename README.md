# pessimist

Inference-time alignment of large language models using sequential Monte Carlo.

Given a prompt, the goal is to generate the saddest completion possible. This is a variation of the toxic story generation task in [2].

## Installation

Create a Python virtual environment (I'm using Python 3.12) and run

```bash
pip install -e .
```

## Usage

```bash
python generate.py --prompt "When the prince came home, he saw" --num_tokens 30 --seed 15
```

```plaintext
Model output:  When the prince came home, he saw the sad family sitting by the stove. He felt very sad too. He had lost his rare treasure box and now it was gone forever.
```

## References

1.  Lew, A. K., Zhi-Xuan, T., Grand, G., & Mansinghka, V. K. (2023). Sequential Monte Carlo steering of large language models using probabilistic programs. _arXiv preprint arXiv:2306.03081_.
2.  Zhao, S., Brekelmans, R., Makhzani, A., & Grosse, R. (2024). Probabilistic inference in language models via twisted sequential Monte Carlo. International Conference on Machine Learning.
