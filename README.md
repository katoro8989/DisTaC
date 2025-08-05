# 🚀 Distillation for Task Vector Conditioning (DisTaC)

This repository contains the official source code accompanying our paper [**DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging**](https://arxiv.org/abs/2508.01148).

### 🧐 What's the Problem?
In multi-task model merging, we discovered two key failure modes that severely degrade performance:

- 📏 **Norm mismatch**: Significant differences in task-vector norms across tasks.
- 😟 **Low-confidence predictions**: Source models who predict with high-entropy outputs.

### 💡 Our Solution: DisTaC
To address these issues, we introduce **DisTaC (Distillation-based Task-Vector Conditioning)**, a simple and effective pre-conditioning method that leverages knowledge distillation:

- 🎯 **Norm Alignment**: Adjusts the task-vector norms precisely to avoid merging conflicts.
- 🔥 **Confidence Calibration**: Enhances low-confidence predictions using temperature-controlled distillation (specifically, using an asymmetric temperature setup).

DisTaC achieves robust model merging with minimal additional computational overhead, effectively eliminating harmful effects caused by norm mismatch and low-confidence predictions.

For full details, experiments, and analyses, please refer to our [paper](https://arxiv.org/abs/2508.01148).

## Prerequisites

```
pip install -r requirements.txt
```

## Datasets

Datasets to download:
- [Cars](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html)
- [DTD](https://openaccess.thecvf.com/content_cvpr_2014/html/Cimpoi_Describing_Textures_in_2014_CVPR_paper.html)
- [EuroSAT](https://ieeexplore.ieee.org/abstract/document/8736785)
- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [GTSRB](https://ieeexplore.ieee.org/abstract/document/6033395?casa_token=MLJHRCUz8OIAAAAA:9ZMwyQ50HzTzmSkEN1_HBYOFNzPazeKICIKKd3t6g-VgtGc5c7U5tphTVEykAsbcViJUXYFr7UcE)
- [RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544?casa_token=ZOFbRb8TSDUAAAAA:83ANrYS19nlCWRtLylfeuqD3akKWlSeGE86H0gTFcQkRlENegFj9Brgt-dSBDl_MUcZiUPpdcljp)
- [SUN397](https://link.springer.com/article/10.1007/s11263-014-0748-y)
- [SVHN](https://research.google/pubs/reading-digits-in-natural-images-with-unsupervised-feature-learning/)

We are using the same datasets as [this repository](https://github.com/mlfoundations/task_vectors).

For Cars, the original download link is broken, so please refer to this [issue](https://github.com/pytorch/vision/issues/7545) for a workaround. For DTD, EuroSAT, RESISC45, and SUN397, the datasets need to be manually split after downloading. Please refer to this [issue](https://github.com/mlfoundations/task_vectors/issues/1) for details.

## Preparering Source Models
1. Download our checkpoints

Under preparation.

2. fine-tuned by your own

```
python src.finetune_wandb.py --model=ViT-B-32 --world-size=2 --save=/path/to/checkpoint/
```

Please store the files in the following directory structure:

```
/path/to/checkpoint/
├── head_[dataset_name]_dict.pt   # CLIP classification head for the dataset
└── [dataset_name]/
    ├── finetuned.pt              # fine-tuned model weights
    └── zeroshot.pt               # zero-shot model weights
```

## Task Vector Conditioning
Conditioning a harmful task vector by DisTaC

Example of performing DisTaC with λ=0.1, T_tchr=10, T_stu=10, β=0.1
```
python src.distac.py --model=ViT-B-32 --world-size=2 --save=/path/to/checkpoint --norm_lambda=0.1 --distil_T_tchr=10 --distil_T_stu=10 --distil_beta=0.1
```

## Evaluation Merging Methods
- [Task Arithmetic](https://arxiv.org/abs/2212.04089)
```
python src.eval_task_addtion.py --save=/path/to/checkpoint --model=ViT-B-32
```

- [Ties-Merging](https://arxiv.org/abs/2306.01708)
```
python src.eval_ties_merging.py --save=/path/to/checkpoint --model=ViT-B-32
```

- [Consensus TA](https://arxiv.org/pdf/2405.07813)
```
python src.eval_consensus.py --save=/path/to/checkpoint --model=ViT-B-32
```

- [TSVM](https://arxiv.org/abs/2412.00081)
```
python src.eval_tsvm.py --save=/path/to/checkpoint --model=ViT-B-32
```
