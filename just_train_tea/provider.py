import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, srgb_to_linear


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
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


def rand_poses(
    size,
    device,
    radius=1,
    theta_range=[np.pi / 3, 2 * np.pi / 3],
    phi_range=[0, 2 * np.pi],
):
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    """

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = (
        torch.rand(size, device=device) * (theta_range[1] - theta_range[0])
        + theta_range[0]
    )
    phis = (
        torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    )

    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = (
        torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    )  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = (
        torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    )
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    interval_nums = torch.tensor(
        [i * 1 / (size - 1) for i in range(size)], dtype=torch.float32, device=device
    )
    thetas = interval_nums * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = interval_nums * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = (
        torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    )  # confused at the coordinate system...
    right_vector = normalize(
        torch.cross(forward_vector, up_vector, dim=-1)
    )  # cross product
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = (
        torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    )
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type="train", downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.args = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode  # only support blender
        self.preload = opt.preload  # preload data into GPU
        self.scale = (
            opt.scale
        )  # camera radius scale to make sure camera are inside the bounding box.
        self.bound = (
            opt.bound
        )  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.training = self.type in ["train", "all", "trainval"]
        self.num_rays = self.opt.num_rays if self.training else -1

        if self.mode == "blender":
            if type == "all":
                transform_paths = glob.glob(os.path.join(self.root_path, "*.json"))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, "r") as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform["frames"].extend(tmp_transform["frames"])
            # load train and val split
            elif type == "trainval":
                with open(
                    os.path.join(self.root_path, f"transforms_train.json"), "r"
                ) as f:
                    transform = json.load(f)
                with open(
                    os.path.join(self.root_path, f"transforms_val.json"), "r"
                ) as f:
                    transform_val = json.load(f)
                transform["frames"].extend(transform_val["frames"])
            # only load one specified split
            else:
                with open(
                    os.path.join(self.root_path, f"transforms_{type}.json"), "r"
                ) as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f"unknown dataset mode: {self.mode}")

        # load image size
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"]) // downscale
            self.W = int(transform["w"]) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        # read images
        frames = transform["frames"]
        if True:
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f"Loading {type} data:"):
                f_path = os.path.join(self.root_path, f["file_path"])
                if (
                    self.mode == "blender"
                    and f_path[-4:].lower() != ".png"
                    and f_path[-4:].lower() != ".jpg"
                ):
                    f_path += ".png"  # so silly...
                if not os.path.exists(f_path):
                    continue
                pose = np.array(f["transform_matrix"], dtype=np.float32)  # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                image = cv2.imread(
                    f_path, cv2.IMREAD_UNCHANGED
                )  # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(
                        image, (self.W, self.H), interpolation=cv2.INTER_AREA
                    )

                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(
                np.stack(self.images, axis=0)
            )  # [N, H, W, C]
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        if self.training and self.opt.error_map:
            self.error_map = torch.ones(
                [self.images.shape[0], 128 * 128], dtype=torch.float
            )  # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                if self.fp16 and self.opt.color_space != "linear":
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if "fl_x" in transform or "fl_y" in transform:
            fl_x = (
                transform["fl_x"] if "fl_x" in transform else transform["fl_y"]
            ) / downscale
            fl_y = (
                transform["fl_y"] if "fl_y" in transform else transform["fl_x"]
            ) / downscale
        elif "camera_angle_x" in transform or "camera_angle_y" in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = (
                self.W / (2 * np.tan(transform["camera_angle_x"] / 2))
                if "camera_angle_x" in transform
                else None
            )
            fl_y = (
                self.H / (2 * np.tan(transform["camera_angle_y"] / 2))
                if "camera_angle_y" in transform
                else None
            )
            if fl_x is None:
                fl_x = fl_y
            if fl_y is None:
                fl_y = fl_x
        else:
            raise RuntimeError(
                "Failed to load focal length, please check the transforms.json!"
            )

        cx = (transform["cx"] / downscale) if "cx" in transform else (self.H / 2)
        cy = (transform["cy"] / downscale) if "cy" in transform else (self.W / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):

        B = len(index)  # a list of length 1
        poses = self.poses[index].to(self.device)  # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(
            poses, self.intrinsics, self.H, self.W, self.num_rays, error_map
        )
        results = {
            "H": self.H,
            "W": self.W,
            "rays_o": rays["rays_o"],
            "rays_d": rays["rays_d"],
        }

        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(
                    images.view(B, -1, C), 1, torch.stack(C * [rays["inds"]], -1)
                )  # [B, N, 3/4]
            results["images"] = images

        # need inds to update error_map
        if error_map is not None:
            results["index"] = index
            results["inds_coarse"] = rays["inds_coarse"]

        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(
            list(range(size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )
        loader._data = self
        return loader
