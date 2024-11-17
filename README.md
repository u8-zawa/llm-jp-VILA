# VILA-ja

This repository contains the code for training [llm-jp/llm-jp-3-vila-14b](https://huggingface.co/llm-jp/llm-jp-3-vila-14b), modified from [VILA repository](https://github.com/NVlabs/VILA/tree/48aadd55c450b182f82f88ad340800428fa3a161).

## Installation

Python version: 3.10.12

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install --upgrade pip
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"
```

```
pip install git+https://github.com/huggingface/transformers@v4.36.2
cp -rv ./llava/train/transformers_replace/* ./venv/lib/python3.10/site-packages/transformers/
```
<!-- # Disable initialization of Eva parameters
cp -rv ./llava/train/timm_replace/* ./venv/lib/python3.10/site-packages/timm/
``` -->

## Training

### 1. Prepare dataset

See [data_prepare/README.md](data_prepare/README.md) and prepare the training datasets for each step.

### 2. Run scripts

There are three stages in the model training.

#### step-0

Tuning the parameters of Projector using English and Japanese image-text pairs datasets.
This takes about 14-15 hours on 8xA100 (40G).

script: [scripts/mdx/release/1_train_step0.sh](scripts/mdx/release/1_train_step0.sh)

#### step-1
Perform multimodal continual pre-training using a relatively large-scale dataset.
This takes about 130 hours on 8x8xA100 (40G).

script: [scripts/mdx/release/2_train_step1.sh](scripts/mdx/release/2_train_step1.sh)

#### step-2

Fine-tune the model with multimodal instruction data in both English and Japanese.
This takes about 11 hours on 4x8xA100 (40G).

script: [scripts/mdx/release/3_train_step2.sh](scripts/mdx/release/3_train_step2.sh)

## Evaluations

We used [llm-jp-eval-mm](https://pypi.org/project/eval-mm/) for evaluation.
Please note that this is currently in beta version.

## Inference

```bash
python -W ignore scripts/mdx/eval/run_inference_ja.py \
    --model-path llm-jp/llm-jp-3-vila-14b \
    --query "<image>\nこの画像について説明してください。" \
    --image-file path/to/image
```

## Model weights

[llm-jp-3-vila-14b](https://huggingface.co/llm-jp/llm-jp-3-vila-14b)

## License
The code is released under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

# Acknowledgement
This codebase is built upon the following projects:

- [LLaVA](https://github.com/haotian-liu/LLaVA) 
- [VILA](https://github.com/NVlabs/VILA)
