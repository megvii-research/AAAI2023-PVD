import os
import copy
import lpips
import glob
import tqdm
import math
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import imageio

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from IPython import embed
import sys

from packaging import version as pver

device = torch.device("cuda")
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def update_loss_rate(cur_lrate, scale=0.99):
    return cur_lrate * scale


def get_softmax_map_mean(a, b):
    return (F.softmax(a) - F.softmax(b)).abs().mean()


def get_kl(inputs, targets):
    return F.kl_div(F.log_softmax(inputs), F.softmax(targets), reduction="sum")


def nerf_matrix_to_ngp(pose, scale=0.8):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def pose_spherical(theta, phi, radius):
    # for synthetic. it generates sphere random poses
    trans_t = lambda t: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).astype(np.float32)
    rot_phi = lambda phi: np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    rot_theta = lambda th: np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(
            np.float32
        )
        @ c2w
    )
    return c2w


def get_rand_poses(data_type="synthetic", original_loader=None):
    """
    Random sampling. Random origins and directions.
    """
    from scipy.spatial.transform import Slerp, Rotation

    assert data_type in {"synthetic", "llff", "tank"}

    def get_single_syn_pose(ph, rand_radius=False):
        theta1 = -180
        theta2 = 180
        phi1 = -ph
        phi2 = 5 - ph if (5 - ph) <= 0 else 0
        theta = theta1 + np.random.rand() * (theta2 - theta1)
        phi = phi1 + np.random.rand() * (phi2 - phi1)
        if rand_radius:
            radius = np.random.uniform(3, 4)
        else:
            radius = 4
        return pose_spherical(theta, phi, radius)

    def get_syn_poses():
        random_poses = np.array([get_single_syn_pose(8) for _ in range(1)])
        for a in range(0, 80):
            rp = np.array(
                [get_single_syn_pose(a) for _ in range(int(((90 - a) // 15) ** 1 + 1))]
            )
            random_poses = np.concatenate([random_poses, rp], axis=0)
        for i in range(len(random_poses)):
            random_poses[i] = nerf_matrix_to_ngp(random_poses[i])
        print(f"\nlen(train data): {len(random_poses)}\n")
        random_poses = torch.from_numpy(random_poses).cuda()
        return random_poses

    def get_tank_poses():
        random_poses = np.array([get_single_syn_pose(8) for _ in range(1)])
        for a in range(5, 20):
            rp = np.array(
                [
                    get_single_syn_pose(a, True)
                    for _ in range(int(((90 - a) // 15) ** 1 + 1))
                ]
            )
            random_poses = np.concatenate([random_poses, rp], axis=0)
        for i in range(len(random_poses)):
            random_poses[i] = nerf_matrix_to_ngp(random_poses[i])
        print(f"\nlen(train data): {len(random_poses)}\n")
        random_poses = torch.from_numpy(random_poses).cuda()
        return random_poses

    def rand_poses_from_cam_centers(centers):
        def normalize(vectors):
            return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

        size = len(centers)
        forward_vector = -normalize(centers)
        up_vector = (
            torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
        )  # confused at the coordinate system...
        right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

        poses = (
            torch.eye(4, dtype=torch.float, device=device)
            .unsqueeze(0)
            .repeat(size, 1, 1)
        )
        poses[:, :3, :3] = torch.stack(
            (right_vector, up_vector, forward_vector), dim=-1
        )
        poses[:, :3, 3] = centers
        return poses

    def get_llff_poses_rand():
        def get_rand_cam_centers_from_bbox(poses, gen_num=30):
            # use poses to estimate the bbox of the camera
            trasitions = poses[:, :3, 3]
            bbox_max = trasitions.max(axis=0) + 1e-6
            bbox_min = trasitions.min(axis=0) - 1e-6
            rand_xs = np.random.uniform(low=bbox_min[0], high=bbox_max[0], size=gen_num)
            rand_ys = np.random.uniform(low=bbox_min[1], high=bbox_max[1], size=gen_num)
            rand_zs = np.random.uniform(low=bbox_min[2], high=bbox_max[2], size=gen_num)
            centers = np.stack([rand_xs, rand_ys, rand_zs], axis=1)
            return centers.astype(np.float32)

        centers = get_rand_cam_centers_from_bbox(original_loader)
        random_poses = rand_poses_from_cam_centers(torch.from_numpy(centers).cuda())
        random_poses[:, 0, 0] = -random_poses[:, 0, 0]
        return random_poses

    if data_type == "synthetic":
        random_poses = get_syn_poses()
    elif data_type == "llff":
        random_poses = get_llff_poses_rand()
    elif data_type == "tank":
        random_poses = get_tank_poses()
    else:
        raise ValueError("illegal")
    return random_poses


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def compute_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable
    output.
    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned
    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    device = img0.device
    img0 = img0.type(torch.float32)
    img1 = img1.type(torch.float32)
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z,
        filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0],
        groups=num_channels,
    )
    filt_fn2 = lambda z: F.conv2d(
        z,
        filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw],
        groups=num_channels,
    )

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1)
    return ssim_map if return_map else ssim


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    import lpips

    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().cuda()


lpips_fns = {
    "alex": lpips.LPIPS(net="alex", version="0.1").eval().cuda(),
    "vgg": lpips.LPIPS(net="vgg", version="0.1").eval().cuda(),
}


def rgb_lpips(gt, im, net_name):
    assert net_name in ["alex", "vgg"]
    gt = gt.type(torch.float32).permute([0, 3, 1, 2]).contiguous().cuda()
    im = im.type(torch.float32).permute([0, 3, 1, 2]).contiguous().cuda()
    return lpips_fns[net_name](gt, im, normalize=True).item()


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(
                error_map.to(device), N, replacement=False
            )  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = (
                inds_coarse // 128,
                inds_coarse % 128,
            )  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (
                (inds_x * sx + torch.rand(B, N, device=device) * sx)
                .long()
                .clamp(max=H - 1)
            )
            inds_y = (
                (inds_y * sy + torch.rand(B, N, device=device) * sy)
                .long()
                .clamp(max=W - 1)
            )
            inds = inds_x * W + inds_y

            results["inds_coarse"] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f"[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}")

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (
            x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8
        )

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    )  # [S, 3]
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )  # [S, 1] --> [x, y, z]
                    u[
                        xi * S : xi * S + len(xs),
                        yi * S : yi * S + len(ys),
                        zi * S : zi * S + len(zs),
                    ] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = (
        vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
        + b_min_np[None, :]
    )
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0
        self.psnr_list = []

    def clear(self):
        self.V = 0
        self.N = 0
        self.psnr_list = []

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        self.psnr_list.append(psnr)
        self.V += psnr
        self.N += 1
        assert self.N == len(self.psnr_list)

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.6f}"


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        opt,  # extra conf
        model_tea,  # network
        model_stu,
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=10e10,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        self.name = name
        self.opt = opt
        self.args = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        self.model_tea = model_tea.to(device)
        self.model_stu = model_stu.to(device)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model_stu.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            self.optimizer = optimizer(self.model_stu)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            self.ls = lr_scheduler
            self.lr_scheduler = lr_scheduler(self.optimizer)
        if ema_decay is not None and ema_decay > 0:
            self.ema = ExponentialMovingAverage(
                self.model_stu.parameters(), decay=ema_decay
            )
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
        self.log(self.opt)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in model_stu.parameters() if p.requires_grad])}"
        )

        if (
            self.workspace is not None
        ):  # only load state_dict for teacher and share backbone for student
            self.log(f"[INFO] Loading teacher ckpt from {self.opt.ckpt_teacher} ...")
            self.load_teacher_checkpoint()
            self.log(self.model_tea)
            self.load_student_checkpoint()
            self.log(self.model_stu)
            # self.model_tea.reset_extra_state()
            # self.model_stu.reset_extra_state()
        """
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...
        """

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def train(self, train_loader, valid_loader, max_epochs):
        self.hard_rays_pool = [torch.tensor([]).cuda(), torch.tensor([]).cuda()]
        self.is_hard_rays_pool_full = False

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name)
            )

        for p in self.model_tea.parameters():
            p.requires_grad = False
        self.model_tea.eval()

        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        if (
            not self.args.use_real_data_for_train
        ):  # using random poses to calculate max_epochs.
            random_poses = get_rand_poses(
                data_type=self.args.data_type,
                original_loader=copy.deepcopy(
                    train_loader._data.poses.detach().cpu().numpy()
                ),
            )
            self.opt.iters = int(
                (self.opt.iters // len(random_poses)) * len(random_poses)
            )
            max_epochs = np.ceil(self.opt.iters / len(random_poses)).astype(np.int32)
            scheduler = lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.opt.iters * 1, eta_min=7e-5
            )  # update scheduler according to new opt.iters
            self.lr_scheduler = scheduler(self.optimizer)

        self.total_epoch = max_epochs
        self.log(f"\n----------------total epoch:{max_epochs} -----------\n")

        self.real_train_poses = copy.deepcopy(train_loader._data.poses)
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            if not self.args.use_real_data_for_train:
                print(f"\n generate new random poses at epoch{self.epoch}")
                random_poses = get_rand_poses(
                    data_type=self.args.data_type,
                    original_loader=self.real_train_poses.detach().cpu().numpy(),
                )
                train_loader._data.poses = copy.deepcopy(random_poses)
                train_loader._data.images = train_loader._data.images[:1].expand(
                    len(random_poses), -1, -1, -1
                )
                train_loader = train_loader._data.dataloader()
            self.train_one_epoch(train_loader)
            print("\n", self.workspace, "\n")

            if (
                self.workspace is not None
                and self.local_rank == 0
                and self.epoch > max_epochs - 1
            ):
                self.save_checkpoint(full=False, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)  # #  为了节省存储，暂时不存储pth

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def train_one_epoch(self, loader):
        # self.log(
        #    f"tttttttttt> Start Training Epoch {self.epoch}/{self.total_epoch}, len(train_data):{len(loader)} lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        # )

        total_loss = 0
        total_loss_rgb = 0
        total_loss_fea_sc = 0
        total_loss_sigma = 0
        total_loss_color = 0

        psnr_tool = PSNRMeter()
        psnr_tool.clear()
        self.pose_psnr = []  # [(pose1, psnr1), (pose2,psnr2)...]

        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model_stu.train()
        self.model_tea.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        for data in loader:
            # update grid every 16 steps. It shoule be run in just train a teacher, but not when distillting a student
            if (
                self.model_tea.cuda_ray
                and self.global_step % self.opt.update_extra_interval == 0
            ):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.opt.update_stu_extra:
                        self.model_stu.update_extra_state()
                    else:
                        pass

            self.local_step += 1
            self.global_step += 1
            self.args.global_step = self.global_step

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                (
                    preds,
                    truths,
                    loss,
                    loss_rgb,
                    loss_fea_sc,
                    loss_color,
                    loss_sigma,
                ) = self.train_step(data)
                if preds is not None:
                    psnr_tool.update(preds, truths)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            total_loss_rgb += loss_rgb
            total_loss_sigma += loss_sigma
            total_loss_color += loss_color
            total_loss_fea_sc += loss_fea_sc

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/loss_rgb", loss_rgb, self.global_step)
                    self.writer.add_scalar(
                        "train/loss_fea_sc", loss_fea_sc, self.global_step
                    )
                    self.writer.add_scalar(
                        "train/loss_coloc", loss_color, self.global_step
                    )
                    self.writer.add_scalar(
                        "train/loss_sigma", loss_sigma, self.global_step
                    )
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                if self.scheduler_update_every_step:  # run this
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    if self.global_step < self.args.stage_iters["stage1"]:
                        pbar.set_description(
                            f"loss={total_loss/self.local_step:.5f}, fea_sc={total_loss_fea_sc/self.local_step:.5f}, lr={cur_lr:.5f}"
                        )
                    elif self.global_step < self.args.stage_iters["stage2"]:
                        pbar.set_description(
                            f"loss={total_loss/self.local_step:.5f}, fea_sc={total_loss_fea_sc/self.local_step:.5f}, sigma={total_loss_sigma/self.local_step:.5f}, color={total_loss_color/self.local_step:.5f}, lr={cur_lr:.6f}"
                        )
                    else:
                        pbar.set_description(
                            f"loss={total_loss/self.local_step:.5f}, rgb={total_loss_rgb/self.local_step:.5f},  lr={cur_lr:.5f}"
                        )
                else:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                pbar.update(loader.batch_size)

            if (
                self.opt.model_type == "vm"
                and self.global_step in self.opt.upsample_model_steps
            ):
                # shrink
                if (
                    self.model_stu.cuda_ray
                ):  # and self.global_step == self.opt.upsample_model_steps[0]:
                    self.model_stu.shrink_model()

                # adaptive voxel size from aabb_train
                n_vox = self.upsample_resolutions.pop(0) ** 3  # n_voxels
                aabb = self.model_stu.aabb_train.cpu().numpy()
                vox_size = np.cbrt(np.prod(aabb[3:] - aabb[:3]) / n_vox)
                reso = ((aabb[3:] - aabb[:3]) / vox_size).astype(np.int32).tolist()
                self.log(
                    f"[INFO] upsample model at step {self.global_step} from {self.model_stu.resolution} to {reso}"
                )
                from IPython import embed

                embed()
                self.model_stu.upsample_model(reso)

                # reset optimizer since params changed.
                self.optimizer = self.optimizer_fn(self.model_stu)
                self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        psnr_tool.psnr_list.sort()
        if self.global_step < self.args.stage_iters["stage1"]:
            self.log(
                f"tttttttttt> Train stage1 Epoch:{self.epoch}. loss_fea:{total_loss_fea_sc/self.local_step:.6f}"
            )
        elif self.global_step < self.args.stage_iters["stage2"]:
            self.log(
                f"tttttttttt> Train stage2 Epoch:{self.epoch}. loss_fea_sc:{total_loss_fea_sc/self.local_step:.3f} loss_sigma:{total_loss_sigma/self.local_step:.3f} loss_color:{total_loss_color/self.local_step:.3f}"
            )
        else:
            self.log(
                f"tttttttttt> Train stage3 Epoch:{self.epoch}. loss_rgb:{total_loss_rgb/self.global_step:.3f} loss_fea_sc:{total_loss_fea_sc/self.local_step:.3f} loss_sigma:{total_loss_sigma/self.local_step:.3f} loss_color:{total_loss_color/self.local_step:.3f}"
            )
            self.log(
                f"tttttttttt> Train PSNR Epoch {self.epoch}. psnr_min:{psnr_tool.psnr_list[0]:.3f} psnr_max:{psnr_tool.psnr_list[-1]:.3f} psnr_mean:{np.mean(psnr_tool.psnr_list):.3f}"
            )

    def get_loss(self, pred, gt):
        if self.opt.loss_type == "L2":
            loss = torch.mean((gt - pred) ** 2)
        elif self.opt.loss_type == "normL2":
            loss = torch.norm(pred - gt)
        elif self.opt.loss_type == "normL1":
            loss = torch.norm(pred - gt, p=1)
        elif self.opt.loss_type == "smoothL1":
            loss = torch.nn.functional.smooth_l1_loss(pred, gt, beta=0.05)
        else:
            raise ValueError("error loss_type")
        return loss

    def train_step(self, data):
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]  [1, N=rays_num=4096, 3]

        loss = 0.0

        # if there is no gt image, we train with CLIP loss.
        if "images" not in data:
            assert 1 == 2
            B, N = rays_o.shape[:2]
            H, W = data["H"], data["W"]
            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(
                rays_o,
                rays_d,
                staged=False,
                bg_color=None,
                perturb=True,
                force_all_rays=True,
                **vars(self.opt),
            )
            pred_rgb = (
                outputs["image"].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            )
            loss = self.clip_loss(pred_rgb)
            return pred_rgb, None, loss

        images = data["images"]  # [B, N, 3/4]
        B, N, C = images.shape

        # if self.opt.color_space == 'linear':
        #    images[..., :3] = srgb_to_linear(images[..., :3])

        if (
            C == 3 or self.model_stu.bg_radius > 0
        ):  #  C=4 in synthetic dataset. C=3 for real dataset
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            bg_color = torch.rand(
                [B, rays_o.size(1), 3], dtype=images.dtype, device=images.device
            )

        if self.opt.render_stu_first:
            outputs_stu = self.model_stu.render(
                rays_o,
                rays_d,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                force_all_rays=False,
                **vars(self.opt),
            )
            pred_rgb_stu = outputs_stu["image"]
            with torch.no_grad():
                outputs_tea = self.model_tea.render(
                    rays_o,
                    rays_d,
                    staged=False,
                    bg_color=bg_color,
                    perturb=True,
                    force_all_rays=False,
                    inherited_params=outputs_stu["inherited_params"],
                    **vars(self.opt),
                )
                pred_rgb_tea = outputs_tea["image"]
        else:
            with torch.no_grad():
                outputs_tea = self.model_tea.render(
                    rays_o,
                    rays_d,
                    staged=False,
                    bg_color=bg_color,
                    perturb=True,
                    force_all_rays=False,
                    **vars(self.opt),
                )
                pred_rgb_tea = outputs_tea["image"]
            outputs_stu = self.model_stu.render(
                rays_o,
                rays_d,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                force_all_rays=False,
                inherited_params=outputs_tea["inherited_params"],
                **vars(self.opt),
            )
            pred_rgb_stu = outputs_stu["image"]
        gt_rgb = pred_rgb_tea
        self.opt.loss_rate_fea_sc = update_loss_rate(self.opt.loss_rate_fea_sc, 0.995)

        if (
            "stage1" in outputs_stu
            and self.opt.loss_rate_fea_sc > 0.0
            and self.model_stu.feature_sigma_color is not None
            and self.model_tea.feature_sigma_color is not None
        ):
            assert (
                self.model_stu.feature_sigma_color.shape
                == self.model_tea.feature_sigma_color.shape
            )
            loss_fea_sc = self.get_loss(
                self.model_stu.feature_sigma_color, self.model_tea.feature_sigma_color
            )
            loss = loss + self.opt.loss_rate_fea_sc * loss_fea_sc
            return None, None, loss, 0, loss_fea_sc.detach().item(), 0, 0
        if "stage2" in outputs_stu:
            if self.opt.loss_rate_color > 0.0:
                assert self.model_stu.color_l.shape == self.model_tea.color_l.shape
                loss_color = self.get_loss(
                    self.model_stu.color_l, self.model_tea.color_l
                )
                loss = loss + self.opt.loss_rate_color * loss_color
            else:
                assert self.model_stu.color_l.shape == self.model_tea.color_l.shape
                loss_color = self.get_loss(
                    self.model_stu.color_l, self.model_tea.color_l
                )
            if self.opt.loss_rate_sigma > 0.0:
                assert self.model_stu.sigma_l.shape == self.model_tea.sigma_l.shape
                loss_sigma = self.get_loss(
                    self.model_stu.sigma_l, self.model_tea.sigma_l
                )
                loss = loss + self.opt.loss_rate_sigma * loss_sigma
            else:
                assert self.model_stu.sigma_l.shape == self.model_tea.sigma_l.shape
                loss_sigma = self.get_loss(
                    self.model_stu.sigma_l, self.model_tea.sigma_l
                )
            if (
                self.opt.loss_rate_fea_sc > 0.0
                and self.model_stu.feature_sigma_color is not None
                and self.model_tea.feature_sigma_color is not None
            ):
                assert (
                    self.model_stu.feature_sigma_color.shape
                    == self.model_tea.feature_sigma_color.shape
                )
                loss_fea_sc = self.get_loss(
                    self.model_stu.feature_sigma_color,
                    self.model_tea.feature_sigma_color,
                )
                loss = loss + self.opt.loss_rate_fea_sc * loss_fea_sc
            else:
                loss_fea_sc = torch.tensor(0.0)
            return (
                None,
                None,
                loss,
                0,
                loss_fea_sc.detach().item(),
                loss_color.detach().item(),
                loss_sigma.detach().item(),
            )

        if self.opt.loss_type == "normL2":
            loss_rgb = torch.norm(pred_rgb_tea - pred_rgb_stu)
        elif self.opt.loss_type == "normL1":
            loss_rgb = torch.norm(pred_rgb_tea - pred_rgb_stu, p=1)
        elif self.opt.loss_type == "L2":
            loss_rgb = self.criterion(pred_rgb_tea, pred_rgb_stu).mean(
                -1
            )  # [B, N, 3] --> [B, N]
            if len(loss_rgb.shape) == 3:  # [K, B, N]
                loss_rgb = loss_rgb.mean(0)
            if self.error_map is not None:
                index = data["index"]  # [B]
                inds = data["inds_coarse"]  # [B, N]
                error_map = self.error_map[index]  # [B, H * W]
                error = loss_rgb.detach().to(
                    error_map.device
                )  # [B, N], already in [0, 1]
                ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error  # ema update
                error_map.scatter_(1, inds, ema_error)
                self.error_map[index] = error_map  # put back
            loss_rgb = loss_rgb.mean()
        else:
            raise ValueError("error loss_type")
        loss = loss + loss_rgb * self.opt.loss_rate_rgb

        if self.opt.l1_reg_weight > 0.0 and self.opt.model_type == "vm":
            loss = loss + self.model_stu.density_loss() * self.opt.l1_reg_weight
        if (
            self.opt.loss_rate_fea_sc > 0.0
            and self.model_stu.feature_sigma_color is not None
            and self.model_tea.feature_sigma_color is not None
        ):
            assert (
                self.model_stu.feature_sigma_color.shape
                == self.model_tea.feature_sigma_color.shape
            )
            loss_fea_sc = self.get_loss(
                self.model_stu.feature_sigma_color, self.model_tea.feature_sigma_color
            )
            loss = loss + self.opt.loss_rate_fea_sc * loss_fea_sc
        elif (
            self.model_stu.feature_sigma_color is None
            or self.model_tea.feature_sigma_color is None
        ):
            loss_fea_sc = torch.tensor(0.0)
        else:
            assert (
                self.model_stu.feature_sigma_color.shape
                == self.model_tea.feature_sigma_color.shape
            )
            loss_fea_sc = self.get_loss(
                self.model_stu.feature_sigma_color, self.model_tea.feature_sigma_color
            )
        if self.opt.loss_rate_color > 0.0:
            assert self.model_stu.color_l.shape == self.model_tea.color_l.shape
            loss_color = self.get_loss(self.model_stu.color_l, self.model_tea.color_l)
            loss = loss + self.opt.loss_rate_color * loss_color
        else:
            assert self.model_stu.color_l.shape == self.model_tea.color_l.shape
            loss_color = self.get_loss(self.model_stu.color_l, self.model_tea.color_l)
        if self.opt.loss_rate_sigma > 0.0:
            assert self.model_stu.sigma_l.shape == self.model_tea.sigma_l.shape
            loss_sigma = self.get_loss(self.model_stu.sigma_l, self.model_tea.sigma_l)
            loss = loss + self.opt.loss_rate_sigma * loss_sigma
        else:
            assert self.model_stu.sigma_l.shape == self.model_tea.sigma_l.shape
            loss_sigma = self.get_loss(self.model_stu.sigma_l, self.model_tea.sigma_l)

        loss_rgb_show = self.criterion(
            pred_rgb_tea.detach(), pred_rgb_stu.detach()
        ).mean()  # [B, N, 3] --> [B, N]
        return (
            pred_rgb_stu,
            gt_rgb,
            loss,
            loss_rgb_show.detach().item(),
            loss_fea_sc.detach().item(),
            loss_color.detach().item(),
            loss_sigma.detach().item(),
        )

    ### ------------------------------

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def evaluate_one_epoch(self, loader, name=None):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()
        if self.opt.test_teacher:
            self.model_stu = self.model_tea
        self.model_stu.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        with torch.no_grad():
            self.local_step = 0
            self.ssim = 0.0
            self.lpips_vgg = 0.0
            self.lpips_alex = 0.0

            # update grid
            if self.model_stu.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.opt.update_stu_extra:
                        self.model_stu.update_extra_state()
                    else:
                        pass

            frames = []
            frames_depth = []
            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    preds_list = [
                        torch.zeros_like(preds).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [
                        torch.zeros_like(preds_depth).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [
                        torch.zeros_like(truths).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                loss_val = loss.item()
                total_loss += loss_val

                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)
                    self.lpips_alex += rgb_lpips(truths, preds, "alex")
                    self.lpips_vgg += rgb_lpips(truths, preds, "vgg")
                    self.ssim += compute_ssim(
                        preds,
                        truths,
                        max_val=max(preds.max().item(), truths.max().item()),
                    ).item()

                    # save image
                    save_path = os.path.join(
                        self.workspace,
                        loader._data.type,
                        f"{name}_{self.local_step:04d}.png",
                    )
                    save_path_depth = os.path.join(
                        self.workspace,
                        loader._data.type,
                        f"{name}_{self.local_step:04d}_depth.png",
                    )
                    # save_path_gt = os.path.join(self.workspace, loader._data.type, f'{name}_{self.local_step:04d}_gt.png')

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == "linear":
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    truth = truths[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    cv2.imwrite(
                        save_path,
                        cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                    frames.append((pred * 255).astype(np.uint8))
                    frames_depth.append((pred_depth * 255).astype(np.uint8))

                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                    pbar.update(loader.batch_size)

            print(
                f"\n----video num(frames): {len(frames)} depth video num:{len(frames_depth)}  ----\n"
            )
            imageio.mimwrite(
                os.path.join(os.path.dirname(save_path), "video.mp4"),
                frames,
                fps=int(30 * 0.7),
                macro_block_size=8,
            )
            imageio.mimwrite(
                os.path.join(os.path.dirname(save_path), "video_depth.mp4"),
                frames_depth,
                fps=int(30 * 0.7),
                macro_block_size=8,
            )

        psnr_tool = self.metrics[0]

        psnr_tool.psnr_list.sort()
        self.log(
            f"\neeeeeeeee> {loader._data.type} PSRN Report: Epoch{self.epoch}.  psnr_mean:{np.mean(psnr_tool.psnr_list):.2f}"
        )

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            for metric in self.metrics:
                # self.log(metric.report(), style="blue")
                psnr = metric.report().split("=")[-1].strip()[:5]
                self.psnr = float(psnr)
                if self.use_tensorboardX and loader._data.type == 'val':
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.ssim /= self.local_step
        self.lpips_alex /= self.local_step
        self.lpips_vgg /= self.local_step
        if self.ema is not None:
            self.ema.restore()
        self.log(
            f"eeeeeeeeee> {loader._data.type} Metric Report: Epoch{self.epoch}. psnr:{psnr} ssim:{self.ssim:.2f} alex:{self.lpips_alex:.2f} vgg:{self.lpips_vgg:.2f}"
        )

    def eval_step(self, data):

        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        images = data["images"]  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == "linear":
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:]
            )
        else:
            gt_rgb = images

        outputs = self.model_stu.render(
            rays_o,
            rays_d,
            staged=True,
            bg_color=bg_color,
            perturb=False,
            **vars(self.opt),
        )

        pred_rgb = outputs["image"].reshape(B, H, W, 3)
        pred_depth = outputs["depth"].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        full = False
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"
        if self.opt.model_type == "vm":
            state = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "stats": self.stats,
                "resolution": self.model_stu.resolution,
            }
        else:
            state = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "stats": self.stats,
            }

        if self.model_stu.cuda_ray:
            state["mean_count"] = self.model_stu.mean_count
            state["mean_density"] = self.model_stu.mean_density

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model_stu.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model_stu.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_teacher_checkpoint(self):
        checkpoint_dict = torch.load(self.opt.ckpt_teacher, map_location=self.device)

        missing_keys, unexpected_keys = self.model_tea.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded teacher model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")
        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if self.model_tea.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model_tea.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model_tea.mean_density = checkpoint_dict["mean_density"]
        """
        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


        if self.model_tea.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model_tea.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model_tea.mean_density = checkpoint_dict['mean_density']
        """

    def load_student_checkpoint(self):
        if self.opt.ckpt_student:
            checkpoint_dict = torch.load(
                self.opt.ckpt_student, map_location=self.device
            )
        else:
            checkpoint_dict = torch.load(
                self.opt.ckpt_teacher, map_location=self.device
            )

        if self.opt.model_type == "vm" and "resolution" in checkpoint_dict:
            self.model_stu.upsample_model(checkpoint_dict["resolution"])
        missing_keys, unexpected_keys = self.model_stu.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded student model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.model_stu.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model_stu.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model_stu.mean_density = checkpoint_dict["mean_density"]

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        """
        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
        """

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if self.model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if model_only:
            return

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def test(self, loader, save_path=None, name=None):
        assert 1 == 2
        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model_stu.eval()
        with torch.no_grad():

            # update grid
            if self.model_stu.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model_stu.update_extra_state()

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                path = os.path.join(save_path, f"{name}_{i:04d}.png")
                path_depth = os.path.join(save_path, f"{name}_{i:04d}_depth.png")

                # self.log(f"[INFO] saving test image to {path}")

                if self.opt.color_space == "linear":
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred_depth = preds_depth[0].detach().cpu().numpy()

                cv2.imwrite(
                    path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(path_depth, (pred_depth * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        H, W = data["H"], data["W"]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model_stu.render(
            rays_o,
            rays_d,
            staged=True,
            bg_color=bg_color,
            perturb=perturb,
            **vars(self.opt),
        )

        pred_rgb = outputs["image"].reshape(-1, H, W, 3)
        pred_depth = outputs["depth"].reshape(-1, H, W)

        return pred_rgb, pred_depth
