# tiny-llm

## Overview

The `tiny-llm` project is designed to provide a hands-on learning experience with Large Language Models (LLMs) and their architecture, specifically focusing on the transformer model as introduced in the paper “Attention Is All You Need”. This project primarily utilizes the transformer decoder blocks, following the GPT-2 architecture.

**Note:** The code in this project is intended for educational purposes, is not optimized and does not adhere to the latest standards in LLMs.

## Features

- **Model Architecture:** Based on GPT-2, utilizing transformer decoder blocks. Implements Pre-Layer Normalization (Pre-LN) for improved stability over Post-Layer Normalization.
- **Training:** Easy setup and training of small transformer models.
- **Evaluation:** Evaluate model performance with provided scripts.
- **Cloud Training:** Supports training on cloud GPUs via Runpod.io.

## Getting Started

### Prerequisites

- Python 3.10 or later (older versions might work but are not guaranteed).

### Installation

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install the project dependencies:**

   ```bash
   pip install .
   ```

### Setup example for the tiny-stories dataset

1. **Configure and Train the Tokenizer:**

   Run the following command to train the tokenizer on the tiny-stories dataset. This will generate a `tokenizer.json` file.

   ```bash
   python configurations/tiny-stories/setup.py train
   ```

2. **Tokenize the Dataset:**

   Use the following command to tokenize the dataset, which will create an HDF5 file with the prepared data.

   ```bash
   python configurations/tiny-stories/setup.py tokenize
   ```

   This step may take some time depending on the dataset size.

### Training

To train the model, use the following command where `<model-name>` refers to the folder name within the `configurations` directory (e.g., `tiny-stories`):

```bash
python src/training.py <model-name>
```

Checkpoints will be saved in the `configurations` folder based on the configured frequency (only those with improved validation loss will be saved).

To resume training from a checkpoint, use:

```bash
python src/training.py <model-name> --checkpoint <path>
```

Refer to the help message (`--help`) for additional parameters.

### Evaluation

To evaluate the model's performance, run:

```bash
python src/evaluation.py <checkpoint-path>
```

### Example

In this [run](https://api.wandb.ai/links/quentin-wendegass-bitmovin/2z44b4ck) the model was trained on the tiny-stories dataset for only one and a half hours on a Nvidia A40 GPU.

Model output:
> The lady smiled too. "That was so nice!" she said. "No, thank you. I was so persistent and I'm glad my friends didn't love him."
The woman said, "That's very kind of you today. Just remember, even if you want to be good friends. Do you want to share?"
So, they went to the tree and ate the apples with all of the berries.

## Cloud Training

For training with cloud GPUs on [runpod.io](https://www.runpod.io), use the `runpod_deploy.py` script. Ensure to configure your `runpod.ini` file with your API token beforehand.

## Improvement Ideas

- **Positional Encoding:** Consider using trainable positional embeddings (e.g., Rotary Positional Embedding) instead of sinusoidal encodings.
- **Grouped Query Attention (GQA):** Implement GQA to potentially speed up training and enhance performance in lower-resource environments.
- **Weight Initialization:** Explore customized weight initialization strategies, such as those used in GPT-2 for residual layers.
- **Weight Decay:** Implement weight decay as described in the GPT-3 paper to potentially improve model performance.
- **Distributed Training:** Investigate the use of Distributed Data Parallel (DDP) to enable training on multiple GPUs.

## Things to Read
* Transformer Architecture https://arxiv.org/pdf/1706.03762
* Positional Encoding https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
* Gaussian Error Linear Units (GELU) https://arxiv.org/pdf/1606.08415
* Large Batch Training https://arxiv.org/pdf/1812.06162
* Gradient Accumulation https://arxiv.org/pdf/2106.02679
* Transformer Training https://www.borealisai.com/research-blogs/tutorial-17-transformers-iii-training/
* Weight tying https://paperswithcode.com/method/weight-tying
* Processing data at scale https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
* Post-LN vs Pre-LN https://arxiv.org/pdf/2206.00330v1#:~:text=There%20are%20currently%20two%20major,normalization%20after%20each%20residual%20connection.
* GPT-2 paper https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* GPT-3 paper https://arxiv.org/pdf/2005.14165

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
