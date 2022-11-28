## One is All: Bridging the Gap Between Neural Radiance Fields Architectures with Progressive Volume Distillation (AAAI 2023)


## WIP: [Project Page](http://sk-fun.fun/PVD/) | [Latest arXiv](..) | [Datasets]() | [ckpts]() |

## Introduction
In this paper, we propose Progressive Volume Distillation (PVD), a systematic distillation method that allows any-to-any conversions between different neural architectures, including MLP(NeRF), sparse(Plenoxels) or low-rank tensors(TensoRF), hash tables(INGP).

## Installation
We recommand using [Anaconda](https://www.anaconda.com/) to setup the environment. Run the following commands:

*Step1*: Create a conda environment named 'pvd'
```
conda create --name pvd python=3.7
conda activate gnerf
pip install -r requirements.txt
```

*Step2*: Install extension modules. (Draw from the great project [torch-ngp](https://github.com/ashawkey/torch-ngp) that we mainly rely on.)
```
bash scripts/install_ext.sh
```

## Datastes & Pretrained-teacher models
You can download the following datasets and put them under ./data.
WIP: [Synthetic-NeRF]() [LLFF]() [Tanks&Temples]()
You can download the pretrained teacher models and put them under ./teacher, or train a teacher model according to the guidance of the next item (it only takes dozens of minutes).
WIP: [Pretrain]()

## Train a teacher
```
python main_just_train_tea.py ./config/CONFIG.yaml --data_dir PATH/TO/DATASET
```

## Distill a student
```
python main_distill.py XXX
```

## Evaluation
- evaluate teacher
```
python eval.py --ckpt PATH/TO/CKPT.pt --ckpt_teacher PATH/TO/GT.json --test_teacher
```
- evaluate student
```
python eval.py --ckpt PATH/TO/CKPT.pt --ckpt_teacher PATH/TO/GT.json --test
```

where you replace PATH/TO/CKPT.pt with your trained model checkpoint, and PATH/TO/GT.json with the json file in NeRF-Synthetic
dataset. Then, just run the  [ATE toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation) on the `evaluation` directory.

## [More detailed parameter description and running commonds]()

## Citation

If you find our code or paper useful, please consider citing
```
@article{pvd2023,
  author    = {Fang, Shuangkang and Xu, Weixin and Wang, Heng and Yang, Yi and Wang, Yufeng and Zhou, Shuchang},
  title     = {One is All: Bridging the Gap Between Neural Radiance Fields Architectures with Progressive Volume Distillation},
  journal   = {AAAI},
  year      = {2023}
}
```

### Acknowledgement
We would like to thank [ingp](https://github.com/NVlabs/instant-ngp),  [torch-ngp](https://github.com/ashawkey/torch-ngp), [TensoRF](https://github.com/apchenstu/TensoRF), [Plenoxels](https://github.com/sxyu/svox2), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)  for their great frameworks!
