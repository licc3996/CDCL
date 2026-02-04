
## Abstract
Person search is a challenging computer vision task that aims to simultaneously detect and re-identify individuals from uncropped gallery images. However, most existing approaches are limited by restricted receptive ffelds, leading to distorted local feature representations under occlusions or complex poses. Additionally, scale variations hinder model generalization in real-world scenarios. To address these limitations, we introduce a novel E-Bike Rider Search (EBRS) dataset, which comprises 27,501 images capturing 963 distinct IDs across 8 camera views at a large urban intersection in a Chinese city. Furthermore, we propose a Context-aware Dynamic Contrastive Learning (CDCL) framework that dynamically
adjusts convolutional weights and performs hard sample mining based on contextual cues, thereby improving discriminative capability for both local details and global features. Extensive experiments show our method achieves state-ofthe-art
 performance on CUHK-SYSU and PRW benchmarks, with competitive results on the challenging EBRS dataset, demonstrating its effectiveness.


## Installation

Create the environment using yml  `conda env create -f cdcl.yml` in the root directory of the project.

## Quick Start

Let's say `$ROOT` is the root directory.

1. Download [EBRS](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af)  datasets, and unzip them to `$ROOT/data`
```
$ROOT/data
├── EBRS
└── PRW
└── CUHK-SYSU
```
2. Run an inference demo by specifying the paths of the checkpoint and corresponding configuration file. `python train.py --cfg $ROOT/exp_cuhk/config.yaml --ckpt $ROOT/exp_cuhk/best_cuhk.ph` You can checkout the result in `demo_imgs` directory.

![demo.jpg](./demo_imgs/demo.jpg)

## Training

Pick one configuration file you like in `$ROOT/configs`, and run with it.

```
python train.py --cfg configs/cuhk_sysu.yaml
```

**Note**: At present, our script only supports single GPU training, but distributed training will be also supported in future. By default, the batch size and the learning rate during training are set to 3 and 0.003 respectively. If your GPU cannot provide the required memory, try smaller batch size and learning rate (*performance may degrade*). Specifically, your setting should follow the [*Linear Scaling Rule*](https://arxiv.org/abs/1706.02677): When the minibatch size is multiplied by k, multiply the learning rate by k. For example:

```
python train.py --cfg configs/cuhk_sysu.yaml INPUT.BATCH_SIZE_TRAIN 3 SOLVER.BASE_LR 0.0003
```

**Tip**: If the training process stops unexpectedly, you can resume from the specified checkpoint.

```
python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /path/to/your/checkpoint
```

## Test

Suppose the output directory is `$ROOT/exp_cuhk`. Test the trained model:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_xx.pth 
```

Test the upper bound of the person search performance by using GT boxes:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_xx.pth EVAL_USE_GT True
```



<hr />

## References
Our code is based on [SeqNet](https://github.com/serend1p1ty/SeqNet) and [COAT](https://github.com/Kitware/COAT)  repositories. 
We thank them for releasing their baseline code.


## Citation

```
@inproceedings{li2021sequential,
  title={Sequential end-to-end network for efficient person search},
  author={Li, Zhengjia and Miao, Duoqian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2011--2019},
  year={2021}
}
@inproceedings{yu2022cascade,
  title={Cascade transformers for end-to-end person search},
  author={Yu, Rui and Du, Dawei and LaLonde, Rodney and Davila, Daniel and Funk, Christopher and Hoogs, Anthony and Clipp, Brian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7267--7276},
  year={2022}
}
