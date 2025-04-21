# pessimist

An attempt to use sequential Monte Carlo for inference-time alignment of large language models. The goal is to steer the model to generate tragic sentences given a prompt.

**Note**: This is a work in progress and not yet fully functional.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python main.py --prompt "When the prince came home, he saw" --num_tokens 30 --seed 15
```

```plaintext
Model output: When the prince came home, he saw everyone weeping. He asked his mother “why are our people sad?” His mother told him noblemen had been killed and cities burned. The prince
```

## References

1.  Lew, A. K., Zhi-Xuan, T., Grand, G., & Mansinghka, V. K. (2023). Sequential Monte Carlo steering of large language models using probabilistic programs. _arXiv preprint arXiv:2306.03081_.
2.  Zhao, S., Brekelmans, R., Makhzani, A., & Grosse, R. (2024). Probabilistic inference in language models via twisted sequential Monte Carlo. International Conference on Machine Learning.
