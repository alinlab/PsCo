# Unsupervised Meta-learning via Few-shot Pseudo-supervised Contrastive Learning

PyTorch implementation for "[Unsupervised Meta-learning via Few-shot Pseudo-supervised Contrastive Learning](https://arxiv.org/abs/2303.00996)" (accepted Spotlight presentation in ICLR 2023)

<img width="100%" src="https://user-images.githubusercontent.com/69646951/220236285-73966fae-da69-4416-b088-b6373df9373e.png"/>

**TL;DR:** Constructing online pseudo-tasks via momentum representations and applying contrastive learning improves the pseudo-labeling strategy progressively for meta-learning.

## Install

```bash
conda create -n unsup_meta python=3.9
conda activate unsup_meta
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install ignite -c pytorch
pip install packaging tensorboard sklearn
```

## Meta-Training PsCo

### Omniglot
```bash
python train.py --model psco --backbone conv4 --prediction --num-shots 1 \
    --dataset omniglot --datadir DATADIR \
    --logdir logs/omniglot/psco
```

### miniImageNet
```bash
python train.py --model psco --backbone conv5 --prediction --num-shots 4 \
    --dataset miniimagenet --datadir DATADIR \
    --logdir logs/miniimagenet/psco
```

## Meta-Testing PsCo

### Standard few-shot classification (Table 1)
- For Omniglot
```bash
python test.py --model psco --backbone conv4 --prediction --num-shots 1 \
    --ckpt logs/omniglot/psco/last.pth \
    --pretrained-dataset omniglot \
    --dataset omniglot --datadir [DATADIR] \
    --N 5 --K 1 --num-tasks 2000 \
    --eval-fewshot-metric supcon
```
- For miniImageNet
```bash
python test.py --model psco --backbone conv5 --prediction --num-shots 4 \
    --ckpt logs/miniimagenet/psco/last.pth \
    --pretrained-dataset miniimagenet \
    --dataset miniimagenet --datadir [DATADIR] \
    --N 5 --K 1 --num-tasks 2000 \
    --eval-fewshot-metric supcon
```

### Cross-domain few-shot classification with miniImageNet pretrained (Table 2)
- miniImageNet to [DATASET]
```bash
python test.py --model psco --backbone conv5 --prediction --num-shots 4 \
    --ckpt logs/miniimagenet/psco/last.pth \
    --pretrained-dataset miniimagenet \
    --dataset [DATASET] --datadir [DATADIR] \
    --N 5 --K 5 --num-tasks 2000 \
    --eval-fewshot-metric ft-supcon
```

- [DATASET] list
    - cub200       (For CUB200)
    - cars         (For Cars)
    - places       (For Places)
    - plantae      (For Plantae)
    - cropdiseases (For CropDiseases)
    - eurosat      (For EuroSAT)
    - isic         (For ISIC)
    - chestx       (For ChestX)