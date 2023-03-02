# custom datasets

Our dataset format is based on the [torch-ngp](https://github.com/ashawkey/torch-ngp/tree/3b066b6cd6ccd3610cb66a56a54f5daaf12a8033), which totally supports custom dataset in form of colmap. 
The specific steps for supporting custom dataset are as follows:

- 1. take a video / many photos from different views 
- 2. put the video under a path like ./data/custom/video.mp4 or the images under ./data/custom/images/*.jpg.
- 3. call the preprocess code: (should install ffmpeg and colmap first! refer to the  [colmap2nerf.py](https://github.com/ashawkey/torch-ngp/blob/3b066b6cd6ccd3610cb66a56a54f5daaf12a8033/scripts/colmap2nerf.py) for more options.)
    - python colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
    - python colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
- 4. it should create the transform.json, and you can train with: (you'll need to try with different scale & bound & dt_gamma to make the object correctly located in the bounding box and render fluently.)


Then you can train a teacher and distill students with various structures according to our introduction：https://github.com/megvii-research/AAAI2023-PVD


# some ways to reduce GPU memory

- 1. Reduce the value of parameter '--num_rays' .
- 2. Try to disable the parameter of '--preload'. When preload=True, it will load all image data into the gpu. For images with large resolution, it will occupy memory seriously (But I haven't tested this parameter, which is inherited from torch-ngp).
- 3. If you don't have too strict requirements for image resolution, you can use the downsampled image for experiment.
- 4. In the distillation process, due to the need to load the student and the teacher network at the same time, it will consume more memory. One solution is to separately inference the teacher network in advance and record the data required for distillation, and use these data to guide the training of the student.
- 5. For different model(INGP/Plenoxels/NeRF/TensoRF), there are different parameters to adjust the model size. For example, you can reduce the number and resolution of hash tables in INGP, reduce the resolution of Plenoxels or tensoRF, and reduce the number of MLP parameters in NeRF, etc.
- 6. The current code does not support multi-GPUs temporarily, but it should be easy to implement. If the above cannot solve your problem, you can try to implement DDP.

