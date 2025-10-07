# LLM Project

This is a project for the course 'Data Mining'

To run the project uv must be installed.

The project dependencies are tracked with Nix, run the following command to install dependencies

```sh
uv run main.py
```

## Dataset

A good dataset found at: https://huggingface.co/datasets/bbunzeck/phoneme-babylm-10M

To load it:

```py
from datasets import load_dataset

ds = load_dataset("bbunzeck/phoneme-babylm-10M")
```
