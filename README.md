## PVD-AL: Progressive Volume Distillation with Active Learning for Efficient Conversion Between Different NeRF Architectures


## [Project Page](http://sk-fun.fun/PVD-AL/) | [Paper](https://arxiv.org/abs/2304.04012) | [Datasets](https://drive.google.com/drive/folders/1U06KAEsW53PolLI3U8hWUhzzIH74QGaP?usp=sharing) | [Ckpts](https://drive.google.com/drive/folders/1GGJf-FTmpCJjmEn-AF_S9-HrLRkFe5Ud?usp=sharing) |



<img width="1005" alt="image" src="https://user-images.githubusercontent.com/34268707/231034579-a1beb97a-2aa4-469f-9bcd-36d3a83bfd7b.png">


### This code is based on the [PVD](https://github.com/megvii-research/AAAI2023-PVD), in which we additionally introduce an active learning strategy to take the PVD performance one step further, which provides a more comprehensive and deeper understanding of distillation between different architectures.


### WHAT PVD-AL CAN DO？
- PVD-AL allows conversions between different NeRF architectures, including MLP, sparse Tensors, low-rank Tensors and hashtables, breaking down the barrier of independent research between them and reaching their full potential.
- With PVD-AL, an MLP-based NeRF model can be distilled from a hashtable-based Instant-NGP model at a 10X ~ 20X faster speed than being trained the original NeRF from scratch, with smaller model parameters.

- With the help of PVD-AL, using a high-performance teacher, such as hashtables and VM-decomposition structures, frequently improves student model synthesis quality compared with training the student from scratch.
- PVD-AL allows for the fusion of various properties between different structures. Calling PVD-AL multiple times can obtain models with multiple editing properties. It is also possible to convert a scene under a specific model to another model that runs more efficiently to meet the real-time requirements of downstream tasks.
- The three levels of active learning strategies in PVD-AL are decoupled, flexible, and highly versatile, thus can be also easily applied as plug-in to other distillation tasks that use NeRF-based model as a teacher or student.

<br>

# The complete code will be released in May
