import torch
import os
import argparse

from distill_mutual.network import NeRFNetwork
from functools import partial
from time import time
from distill_mutual.provider import NeRFDataset
from distill_mutual.utils import *
from IPython import embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_codes_env(workspace):
    path = os.path.join(workspace, "codes_env")
    os.makedirs(path, exist_ok=True)
    os.system(f"cp *.py {path}")
    os.system(f"cp -r raymarching {path}")
    os.system(f"cp -r distill_mutual {path}")
    os.system(f"cp -r nerf {path}")


def load_from_txt(opt, except_space=""):
    # except_space = {'workspace', 'teacher_type', 'model_type', 'test', 'test_teacher', 'use_spiral_pose', 'ckpt_teacher'}
    except_space = {"workspace"}
    with open(
        os.path.join(opt.ckpt_teacher.split("checkpoints")[0], "args.txt"), "r"
    ) as f:  # change this path to your own params settings
        load_args = f.readlines()
    for i in range(1, len(load_args)):
        if "(" in load_args[i]:
            k, v = eval(load_args[i])
        else:
            continue
        if k in opt and k not in except_space and v != opt.__dict__[k]:
            print(k, v, opt.__dict__[k])
            opt.__dict__[k] = v


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "-O", action="store_true", help="equals --fp16 --cuda_ray --preload"
    )
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--seed", type=int, default=0)

    # training options
    parser.add_argument("--iters", type=int, default=30000, help="training iters")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument(
        "--num_rays",
        type=int,
        default=4096,
        help="num rays sampled per image for each training step",
    )
    parser.add_argument(
        "--cuda_ray",
        action="store_true",
        help="use CUDA raymarching instead of pytorch",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1024,
        help="max num steps sampled per ray (only valid when using --cuda_ray)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=512,
        help="num steps sampled per ray (only valid when NOT using --cuda_ray)",
    )
    parser.add_argument(
        "--upsample_steps",
        type=int,
        default=0,
        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)",
    )
    parser.add_argument(
        "--update_extra_interval",
        type=int,
        default=16,
        help="iter interval to update extra status (only valid when using --cuda_ray)",
    )
    parser.add_argument(
        "--max_ray_batch",
        type=int,
        default=4096,
        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)",
    )

    parser.add_argument(
        "--fp16", action="store_true", help="use amp mixed precision training"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="blender",
        help="dataset mode, supports (colmap, blender)",
    )
    parser.add_argument(
        "--color_space",
        type=str,
        default="srgb",
        help="Color space, supports (linear, srgb)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="preload all data into GPU, accelerate training but use more GPU memory",
    )
    parser.add_argument(
        "--bound",
        type=float,
        default=1,
        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.8,
        help="scale camera location into box[-bound, bound]^3",
    )
    parser.add_argument(
        "--dt_gamma",
        type=float,
        default=0,
        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)",
    )
    parser.add_argument(
        "--min_near", type=float, default=0.2, help="minimum near distance for camera"
    )
    parser.add_argument(
        "--density_thresh",
        type=float,
        default=10,
        help="threshold for density grid to be occupied",
    )
    parser.add_argument(
        "--bg_radius",
        type=float,
        default=-1,
        help="if positive, use a background model at sphere(bg_radius)",
    )

    # experimental
    parser.add_argument(
        "--error_map", action="store_true", help="use error map to sample rays"
    )
    parser.add_argument(
        "--clip_text", type=str, default="", help="text input for CLIP guidance"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="normL2",
        choices=["normL2", "L2", "normL1", "L1"],
    )
    parser.add_argument(
        "--distill_mode",
        type=str,
        default="no_fix_mlp",
        choices=["fix_mlp", "no_fix_mlp"],
        help="fix mlp for hash",
    )
    parser.add_argument("--loss_rate_rgb", type=float, default=1.0)
    parser.add_argument("--loss_rate_fea_sc", type=float, default=0.002)
    parser.add_argument("--loss_rate_color", type=float, default=0.002)
    parser.add_argument("--loss_rate_sigma", type=float, default=0.002)
    parser.add_argument("--l1_reg_weight", type=float, default=1e-4)

    parser.add_argument("--ckpt_teacher", type=str, default="")
    parser.add_argument("--ckpt_student", type=str, default="")
    parser.add_argument("--sigma_clip_min", type=float, default=-2)
    parser.add_argument("--sigma_clip_max", type=float, default=7)
    parser.add_argument("--render_stu_first", action="store_true", default=False)
    parser.add_argument("--use_diagonal_matrix", action="store_true", default=False)

    parser.add_argument("--test_teacher", action="store_true", default=False)
    parser.add_argument("--test_metric", action="store_true", default=False)
    parser.add_argument(
        "--test_type_trainval", action="store_true", default=False
    )  # XXX

    parser.add_argument("--PE", type=int, default=10)
    parser.add_argument("--nerf_layer_num", type=int, default=8)
    parser.add_argument("--nerf_layer_wide", type=int, default=256)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--residual", type=int, default=3)

    parser.add_argument("--resolution0", type=int, default=300)
    parser.add_argument("--resolution1", type=int, default=300)
    parser.add_argument(
        "--upsample_model_steps", type=int, action="append", default=[1e10]
    )

    parser.add_argument("--teacher_type", default="hash", type=str)
    parser.add_argument("--model_type", default="hash", type=str)
    parser.add_argument(
        "--data_type",
        default="synthetic",
        type=str,
        choices=["synthetic", "llff", "tank"],
    )

    parser.add_argument("--update_stu_extra", action="store_true", default=False)
    parser.add_argument("--ema_decay", type=float, default=-1.0)
    parser.add_argument("--grid_size", type=int, default=128)

    parser.add_argument("--plenoxel_degree", type=int, default=3)
    parser.add_argument("--plenoxel_res", type=str, default="[128,128,128]")

    parser.add_argument("--load_args", action="store_true", default=False)

    parser.add_argument("--eval_interval_epoch", default=1e5, type=int, help="")

    parser.add_argument(
        "--use_real_data_for_train",
        action="store_true",
        default=False,
    )

    parser.add_argument("--enable_embed", action="store_true")
    parser.add_argument("--enable_edit_plenoxel", action="store_true")
    parser.add_argument(
        "--stage_iters", type=str, default="{'stage1':2000, 'stage2':5000}"
    )

    opt = parser.parse_args()
    opt.stage_iters = eval(opt.stage_iters)
    opt.O = True  # always use -O
    opt.render_stu_first = True
    if opt.model_type == "mlp":
        opt.lr *= 0.1
    if (
        "tensors" == opt.model_type or "tensors" == opt.teacher_type
    ):  # plenoxel have no features
        opt.stage_iters["stage1"] = -1
    save_codes_env(opt.workspace)

    if opt.load_args:
        load_from_txt(opt)
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    assert opt.model_type in ["hash", "mlp", "vm", "tensors"]
    assert opt.teacher_type in ["hash", "mlp", "vm", "tensors"]
    print(opt)
    seed_everything(opt.seed)

    model_tea = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        model_type=opt.teacher_type,
        args=opt,
        grid_size=opt.grid_size,
        is_teacher=True,
    )

    model_stu = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        model_type=opt.model_type,
        args=opt,
        grid_size=opt.grid_size,
    )

    print("\nteacher:", model_tea)
    print(f"\n{opt.model_type}", model_stu)

    criterion = torch.nn.MSELoss(reduction="none")

    # ------------------------------------ test-test-test-test-test  ----------------------------------------------
    if opt.test or opt.test_teacher or opt.test_type_trainval:
        trainer = Trainer(
            f"{opt.teacher_type}2{opt.model_type}",
            opt,
            model_tea,
            model_stu,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            metrics=[PSNRMeter()],
            use_checkpoint=opt.ckpt,
            ema_decay=opt.ema_decay,
        )

        if opt.test_type_trainval:
            test_loader = NeRFDataset(opt, device=device, type="trainval").dataloader()
        else:
            test_loader = NeRFDataset(opt, device=device, type="test").dataloader()
        if opt.mode == "blender":
            trainer.evaluate(test_loader)
        else:
            trainer.test(test_loader)

    # ------------------------------------ train-train-train-train  ----------------------------------------------
    else:
        for p in model_tea.parameters():
            p.requires_grad = False
        if opt.distill_mode == "fix_mlp":
            for n, p in model_stu.named_parameters():
                if "sigma_net" in n or "color_net" in n:
                    p.requires_grad = False
            idx = 1 if opt.model_type == "vm" else 3
            optimizer = lambda model_stu: torch.optim.AdamW(
                model_stu.get_params(opt.lr)[idx:],
                betas=(0.9, 0.99),
                eps=1e-15,
                amsgrad=False,
            )
        else:
            optimizer = lambda model_stu: torch.optim.AdamW(
                model_stu.get_params(opt.lr),
                betas=(0.9, 0.99),
                eps=1e-15,
                amsgrad=False,
            )
        # fake train loader. The real random data for distillating will be generated in utils.py
        train_loader = NeRFDataset(opt, device=device, type="train").dataloader()
        train_loader = NeRFDataset(opt, device=device, type="train").dataloader()
        opt.iters = opt.iters + opt.iters % len(
            train_loader
        )  # will be updated in utils according to the number of random data
        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        scheduler = lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.iters * 1, eta_min=5e-5
        )

        trainer = Trainer(
            f"{opt.teacher_type}2{opt.model_type}",
            opt,
            model_tea,
            model_stu,
            device=device,
            workspace=opt.workspace,
            optimizer=optimizer,
            criterion=criterion,
            ema_decay=opt.ema_decay,
            fp16=opt.fp16,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            metrics=[PSNRMeter()],
            use_checkpoint=opt.ckpt,
            eval_interval=opt.eval_interval_epoch,
        )
        upsample_resolutions = (
            (
                np.round(
                    np.exp(
                        np.linspace(
                            np.log(opt.resolution0),
                            np.log(opt.resolution1),
                            len(opt.upsample_model_steps) + 1,
                        )
                    )
                )
            )
            .astype(np.int32)
            .tolist()[1:]
        )
        trainer.upsample_resolutions = upsample_resolutions
        argstxt = sorted(opt.__dict__.items())
        with open(os.path.join(opt.workspace, "args.txt"), "w") as f:
            for t in argstxt:
                f.write(str(t) + "\n")
        start_time = time.time()
        valid_loader = NeRFDataset(
            opt, device=device, type="val", downscale=1
        ).dataloader()
        test_loader = NeRFDataset(opt, device=device, type="test").dataloader()
        trainer.train(train_loader, valid_loader, max_epoch)

        end_time = time.time()
        train_time = end_time - start_time
        print(f"\nusing_time : {train_time:.2f}s\n")

        # run test data
        test_loader = NeRFDataset(opt, device=device, type="test").dataloader()
        print(opt.workspace)

        trainer.evaluate(test_loader)

        with open(os.path.join(trainer.workspace, "args.txt"), "a+") as f:
            txt = f"\npsnr: {trainer.psnr:.2f} \nssim: {trainer.ssim:.3f} \nalex: {trainer.lpips_alex:.3f}\nvgg:{trainer.lpips_vgg:.3f}"
            f.write(txt)
        cmd = f"mv {trainer.workspace} {trainer.workspace}-pnsr{trainer.psnr}"
        os.system(cmd)
