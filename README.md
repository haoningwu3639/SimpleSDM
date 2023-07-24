# SimpleSDM
This repository contains a simple and flexible PyTorch implementation of StableDiffusion based on diffusers.

## Prepartion
- You should download the checkpoints of SDM-1.5, from [SDM-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main), including scheduler, text_encoder, tokenizer, unet, and vae. Then put it in the ckpt folder.

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/)
- xformers == 0.0.13
- diffusers == 0.13.1
- accelerate == 0.17.1
- transformers == 4.27.4

A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

## Dataset Preparation
- You need write a DataLoader suitable for your own Dataset, because we just provide a simple example to test the code.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --multi_gpu train.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python inference.py --prompt "A cat is running in the rain."
```

## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers).

Ongoing update...