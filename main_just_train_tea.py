import torch
import os
import argparse

from just_train_tea.network import NeRFNetwork

from functools import partial
from just_train_tea.provider import NeRFDataset
from just_train_tea.utils import *
from time import time


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "-O", action="store_true", help="equals --fp16 --cuda_ray --preload"
    )
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--seed", type=int, default=0)

    ### training options
    parser.add_argument("--iters", type=int, default=40000, help="training iters")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument(
        "--num_rays",
        type=int,
        default=8192,
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
    parser.add_argument("--ff", action="store_true", help="use fully-fused MLP")
    parser.add_argument("--tcnn", action="store_true", help="use TCNN backend")

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
    # (the default value is for the fox dataset)
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

    ### GUI options
    parser.add_argument("--gui", action="store_true", help="start a GUI")
    parser.add_argument("--W", type=int, default=1920, help="GUI width")
    parser.add_argument("--H", type=int, default=1080, help="GUI height")
    parser.add_argument(
        "--radius", type=float, default=5, help="default GUI camera radius from center"
    )
    parser.add_argument(
        "--fovy", type=float, default=50, help="default GUI camera fovy"
    )
    parser.add_argument(
        "--max_spp", type=int, default=64, help="GUI rendering max sample per pixel"
    )

    ### experimental
    parser.add_argument(
        "--error_map", action="store_true", help="use error map to sample rays"
    )
    parser.add_argument(
        "--clip_text", type=str, default="", help="text input for CLIP guidance"
    )
    parser.add_argument(
        "--rand_pose",
        type=int,
        default=-1,
        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses",
    )

    parser.add_argument(
        "--distill_mode",
        type=str,
        default="no_fix_mlp",
        choices=["fix_mlp", "no_fix_mlp"],
    )
    parser.add_argument("--loss_rate_rgb", type=float, default=1.0)
    parser.add_argument("--loss_rate_fea", type=float, default=0.1)
    parser.add_argument("--loss_rate_fea_sc", type=float, default=0.1)
    parser.add_argument("--loss_rate_color", type=float, default=0.0)
    parser.add_argument("--loss_rate_sigma", type=float, default=0)
    parser.add_argument(
        "--L1_tensorAB_reg", type=float, default=1e-3, help="reg for tensor_ab"
    )
    parser.add_argument("--l1_reg_weight", type=float, default=1e-4)

    parser.add_argument("--ckpt_teacher", type=str, default="")
    parser.add_argument("--ckpt_student", type=str, default="")
    parser.add_argument("--sigma_clip_min", type=float, default=-2)
    parser.add_argument("--sigma_clip_max", type=float, default=7)
    parser.add_argument("--use_sigma_clip", action="store_true")
    parser.add_argument("--render_stu_first", action="store_true", default=False)
    parser.add_argument("--nerf_pe", action="store_true", default=False)
    parser.add_argument("--use_real_gt", action="store_true", default=False)
    parser.add_argument("--use_diagonal_matrix", action="store_true", default=False)
    parser.add_argument(
        "--loss_rate_real_gt", type=float, default=0, help="range in [0, 1]"
    )
    parser.add_argument("--test_teacher", action="store_true", default=False)
    parser.add_argument("--test_metric", action="store_true", default=False)

    parser.add_argument("--resolution0", type=int, default=300)
    parser.add_argument("--resolution1", type=int, default=300)
    parser.add_argument(
        "--upsample_model_steps", type=int, action="append", default=[1e10]
    )

    parser.add_argument(
        "--loss_type", type=str, default="L2", choices=["normL2", "L2", "normL1", "L1"]
    )

    parser.add_argument("--PE", type=int, default=10)
    parser.add_argument("--nerf_layer_num", type=int, default=8)
    parser.add_argument("--nerf_layer_wide", type=int, default=256)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--residual", type=int, default=3)

    parser.add_argument("--model_type", default="hash", type=str)
    parser.add_argument("--teacher_type", default="hash", type=str)

    parser.add_argument("--use_upsample_vm", action="store_true", default=False)
    parser.add_argument("--update_stu_extra", action="store_true", default=False)
    parser.add_argument("--ema_decay", type=float, default=-1)
    parser.add_argument("--grid_size", type=int, default=128)

    parser.add_argument("--plenoxel_degree", type=int, default=3)
    parser.add_argument("--plenoxel_res", type=str, default="[128,128,128]")
    parser.add_argument("--just_train_a_model", action="store_true", default=False)
    parser.add_argument("--data_type", type=str, default="")

    opt = parser.parse_args()
    opt.just_train_a_model = True
    opt.update_stu_extra = True
    opt.render_stu_first = True
    opt.O = True
    if opt.model_type == "mlp":
        opt.lr *= 0.1

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    assert opt.model_type in ["hash", "mlp", "vm", "tensors"]
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

    criterion = torch.nn.MSELoss(reduction="none")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.test or opt.test_teacher or opt.test_metric:
        trainer = Trainer(
            opt.model_type,
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
        test_loader = NeRFDataset(opt, device=device, type="test").dataloader()
        trainer.evaluate(test_loader)
    else:
        for p in model_tea.parameters():
            p.requires_grad = False
        optimizer = lambda model_stu: torch.optim.AdamW(
            model_stu.get_params(opt.lr, opt.lr * 0.1),
            betas=(0.9, 0.99),
            eps=1e-15,
            amsgrad=False,
        )
        train_loader = NeRFDataset(opt, device=device, type="train").dataloader()
        valid_loader = NeRFDataset(opt, device=device, type="val").dataloader()
        test_loader = NeRFDataset(opt, device=device, type="test").dataloader()

        if opt.just_train_a_model:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
            )
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=opt.iters * 1
            )
        print(scheduler)

        trainer = Trainer(
            opt.model_type,
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
            eval_interval=500000000,
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

        start_time = time()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)
        print(opt.workspace)

        trainer.evaluate(test_loader)

        with open(os.path.join(trainer.workspace, "args.txt"), "a+") as f:
            txt = f"\npsnr: {trainer.psnr:.2f} \nssim: {trainer.ssim:.3f} \nalex: {trainer.lpips_alex:.3f}\nvgg:{trainer.lpips_vgg:.3f}"
            f.write(txt)
        cmd = f"mv {trainer.workspace} {trainer.workspace}-pnsr{trainer.psnr}"
        print(f"\n{cmd}\n")
        os.system(cmd)
