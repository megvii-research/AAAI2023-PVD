## One is All: Bridging the Gap Between Neural Radiance Fields Architectures with Progressive Volume Distillation (AAAI 2023 Oral)

# :partying_face: ***New:*** :partying_face: Code for more powerful PVD-AL is now provided [here](https://github.com/megvii-research/AAAI2023-PVD/tree/PVD-AL). (We strongly recommend using PVD-AL with better performance).


### [Project Page](http://sk-fun.fun/PVD/) | [Paper](https://arxiv.org/abs/2211.15977) | [Datasets](https://drive.google.com/drive/folders/1U06KAEsW53PolLI3U8hWUhzzIH74QGaP?usp=sharing) | [Ckpts](https://drive.google.com/drive/folders/1GGJf-FTmpCJjmEn-AF_S9-HrLRkFe5Ud?usp=sharing) | [Chinese tutorial](https://github.com/megvii-research/AAAI2023-PVD/blob/main/tools/%E4%B8%AD%E6%96%87%E4%BB%8B%E7%BB%8D.md) | [zhihu](https://zhuanlan.zhihu.com/p/605121286)|

## Introduction
In this paper, we propose Progressive Volume Distillation (PVD), a systematic distillation method that allows any-to-any conversions between different neural architectures, including MLP(NeRF), sparse(Plenoxels) or low-rank tensors(TensoRF), hash tables(INGP).

## Installation
We recommend using [Anaconda](https://www.anaconda.com/) to setup the environment. Run the following commands:

*Step1*: Create a conda environment named 'pvd'
```
conda create --name pvd python=3.7
conda activate pvd
pip install -r ./tools/requirements.txt
```
*Step2*: Install extension modules. (Draw from the great project [torch-ngp](https://github.com/ashawkey/torch-ngp) that we mainly rely on.)
```
bash ./tools/install_extensions.sh
```

## Datastes & Pretrained-teacher models
You can download Synthetic-NeRF/LLFF/Tanks&Temples datasets from [google](https://drive.google.com/drive/folders/1U06KAEsW53PolLI3U8hWUhzzIH74QGaP?usp=sharing), or from [baidu](https://pan.baidu.com/s/1ky_TWrbUZG_MpHTBhncAKA?pwd=4h2h).

And download pretrained-teacher-models from [google](https://drive.google.com/drive/folders/1GGJf-FTmpCJjmEn-AF_S9-HrLRkFe5Ud?usp=sharing), or from [baidu](https://pan.baidu.com/s/1LGLXwLGusX60GpAywLwosg?pwd=34k8).

You can also train a teacher model according to the follow guidance.

## Train a teacher
```
# train a hash-based(INGP) teacher
python main_just_train_tea.py ./data/nerf_synthetic/chair --model_type hash --data_type synthetic  --workspace ./log/train_teacher/hash_chair

# train a sparse-tensor-based(TensoRF VM-decomposion) teacher
python main_just_train_tea.py ./data/nerf_synthetic/chair --model_type vm --data_type synthetic  --workspace ./log/train_teacher/vm_chair

# train a MLP-based(NeRF) teacher
python main_just_train_tea.py ./data/nerf_synthetic/chair --model_type mlp --data_type synthetic  --workspace ./log/train_teacher/mlp_chair

# train a tensors-based(Plenoxels) teacher
python main_just_train_tea.py ./data/nerf_synthetic/chair --model_type tensors --data_type synthetic  --workspace ./log/train_teacher/tensors_chair

```

## Distill a student
```
# teacher: hash(INGP),  student: vm(tensoRF)
python3 main_distill_mutual.py  ./data/nerf_synthetic/chair \
                    --data_type synthetic \
                    --teacher_type hash \
                    --ckpt_teacher ./log/train_teacher/hash_chair/checkpoints/XXX.pth \
                    --model_type vm \
                    --workspace ./log/distill_student/hash2vm/chair
                    
# teacher: MLP(NeRF),  student: tensors(Plenoxels)
python3 main_distill_mutual.py  ./data/nerf_synthetic/chair \
                    --data_type synthetic \
                    --teacher_type mlp \
                    --ckpt_teacher ./log/train_teacher/mlp_chair/checkpoints/XXX.pth \
                    --model_type tensors \
                    --workspace ./log/distill_student/mlp2tensors/chair
                   
```

## Evaluation

```
# evaluate a hash teacher
python main_distill_mutual.py ./data/nerf_synthetic/chair  --teacher_type hash --ckpt_teacher PATH/TO/CKPT.pth --test_teacher --data_type synthetic --workspace ./log/eval_teacher/hash_chair

# evaluate a mlp student
python main_distill_mutual.py ./data/nerf_synthetic/chair --model_type mlp --ckpt PATH/TO/CKPT.pth --test --data_type synthetic --workspace ./log/eval_student/mlp_chair
```

## More detailed parameter description and running commonds
Please refer to [more running description](https://github.com/megvii-research/AAAI2023-PVD/blob/main/tools/details.md) for details of training different types of datasets, parameter adjustment, key settings, etc.

## Citation

If you find our code or paper useful, please consider citing
```
@article{fang2022one,
  title={One is All: Bridging the Gap Between Neural Radiance Fields Architectures with Progressive Volume Distillation},
  author={Fang, Shuangkang and Xu, Weixin and Wang, Heng and Yang, Yi and Wang, Yufeng and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2211.15977},
  year={2022}
}
```

### Acknowledgement
We would like to thank [ingp](https://github.com/NVlabs/instant-ngp),  [torch-ngp](https://github.com/ashawkey/torch-ngp), [TensoRF](https://github.com/apchenstu/TensoRF), [Plenoxels](https://github.com/sxyu/svox2), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)  for their great frameworks!

Also check out [Arch-Net](https://github.com/megvii-research/Arch-Net) for more on general progressive distillation.
