import os
from argparse import ArgumentParser



tea_ingp = [
            'log/teacher/1lego/lego-lr0.02-4w-bs8192-feascclip-2to7-b1s0.8gamma0-srgb-L2-NoEma-pnsr35.70/checkpoints/ngp_ep0200.pth',
            'log/teacher/1chair/chair-lr0.02-5w-bs8192-feascclip-2to7-b1s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.05-pnsr34.33/checkpoints/ngp_ep0250.pth',
            'log/teacher/1drums/drums-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr25.91/checkpoints/ngp_ep0250.pth',
            'log/teacher/1ficus/ficus2-lr0.007-50.0k-bs8192-feascclip-2to7-b1.0s0.75gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr32.87/checkpoints/ngp_ep0250.pth',
            'log/teacher/1hotdog/hotdog-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr36.60/checkpoints/ngp_ep0250.pth',
            'log/teacher/1materials/materials3-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.72gamma0-srgb-L2-NoEma-errorMap-sc0.07-pnsr29.74/checkpoints/ngp_ep0250.pth',
            'log/teacher/1mic/mic-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr35.36/checkpoints/ngp_ep0250.pth',
            'log/teacher/1ship/ship-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr30.27/checkpoints/ngp_ep0250.pth']

tea_vm = [
        'log/teacher/tensorf/0lego/lego-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res300-pnsr34.72-1546.23s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0chair/chair-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res300-pnsr33.21-1419.67s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0drums/drums-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res300-pnsr25.61-1205.05s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0ficus/ficus-30.0k-lr0.005-feascClip-2to7-gamma0-L2-bound1-scale0.75-res300-pnsr30.74-727.18s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0hotdog/hotdog-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.67-res300-pnsr35.85-2206.01s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0materials/materials-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.72-res300-pnsr29.65-1375.94s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0mic/mic-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res300-pnsr32.94-768.93s/checkpoints/ngp_ep0150.pth',
        'log/teacher/tensorf/0ship/ship-30.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.67-res300-pnsr29.47-1771.65s/checkpoints/ngp_ep0150.pth',
        ]

tea_nerf = [
        'log/teacher/nerf/0lego/lego-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.8-10PE256MLP8num3skip-pnsr33.64-3298.29s/checkpoints/ngp_ep0250.pth',  # 32.54
        'log/teacher/nerf/0chair/chair-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.8-10PE256MLP8num3skip-pnsr31.88-2302.67s/checkpoints/ngp_ep0250.pth',  # 33
        'log/teacher/nerf/0drums/drums-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.8-10PE256MLP8num3skip-pnsr24.95-1764.35s/checkpoints/ngp_ep0250.pth',  # 25.01
        'log/teacher/nerf/0ficus/ficus-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.75-10PE256MLP8num3skip-pnsr30.21-1616.88s/checkpoints/ngp_ep0250.pth',  #30.13
        'log/teacher/nerf/0hotdog/hotdog-50.0k-lr0.001-feascClip-2to7-gamma0-L2-bound1-scale0.67-10PE256MLP8num3skip-pnsr35.35-3238.17s/checkpoints/ngp_ep0250.pth',  # 36.18
        'log/teacher/nerf/0materials/materials-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.72-10PE256MLP8num3skip-pnsr29.05-2841.05s/checkpoints/ngp_ep0250.pth',  # 29.62
        'log/teacher/nerf/0mic/mic-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.8-10PE256MLP8num3skip-pnsr32.4-1165.16s/checkpoints/ngp_ep0250.pth',  # 32.91
        'log/teacher/nerf/0ship/ship-50.0k-lr0.002-feascClip-2to7-gamma0-L2-bound1-scale0.67-10PE256MLP8num3skip-pnsr28.8-4132.97s/checkpoints/ngp_ep0250.pth',  # 28.65
        ]

tea_plenoxel = [
        'log/teacher/plenoxel/0lego/lego-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res128-pnsr27.9-404.24s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0chair/chair-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res128-pnsr29.07-387.19s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0drums/drums-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res128-pnsr23.34-754.54s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0ficus/ficus-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.75-res128-pnsr26.18-530.18s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0hotdog/hotdog-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.67-res128-pnsr32.23-508.68s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0materials/materials-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.72-res128-pnsr26.08-516.29s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0mic/mic-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.8-res128-pnsr29.29-450.94s/checkpoints/ngp_ep0075.pth',
        'log/teacher/plenoxel/0ship/ship-15.0k-lr0.02-feascClip-2to7-gamma0-L2-bound1-scale0.67-res128-pnsr25.87-507.31s/checkpoints/ngp_ep0075.pth',
        ]


def get_params_ngp(tea):
    scene = tea.split('/')[2].strip('1')
    tea_psnr = tea.split('/')[3][-5:]
    scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
    bound, gamma = 1, 0
    return scene, tea_psnr, bound, scale, gamma


def get_params_vm(tea):
    scene = tea.split('/')[3].strip('0')
    tea_psnr = tea.split('pnsr')[-1][:5]
    scale = tea.split('scale')[-1].split('-')[0]
    bound, gamma = 1, 0
    return scene, tea_psnr, bound, scale, gamma


def get_params_nerf(tea):
    scene = tea.split('/')[3].strip('0')
    tea_psnr = tea.split('pnsr')[-1].split('-')[0]
    scale = tea.split('scale')[-1].split('-')[0]
    bound, gamma = 1, 0
    return scene, tea_psnr, bound, scale, gamma


def get_params_plenoxel(tea):
    scene = tea.split('/')[3].strip('0')
    tea_psnr = tea.split('pnsr')[-1].split('-')[0]
    scale = tea.split('scale')[-1].split('-')[0]
    bound, gamma = 1, 0
    return scene, tea_psnr, bound, scale, gamma


for tea in tea_ingp:
    print(tea)
    print(get_params_ngp(tea))

for tea in tea_nerf:
    print(tea)
    print(get_params_nerf(tea))

for tea in tea_vm:
    print(tea)
    print(get_params_vm(tea))


for tea in tea_plenoxel:
    print(tea)
    print(get_params_plenoxel(tea))

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--search_mode', type=str, default="")
    args = parser.parse_args()

    if args.search_mode == "ingp2others":
        it, loss_type = 10000, 'normL2'
        # --bound {bound} \
        # --scale {scale} \
        # --dt_gamma {gamma} \
        for stu in ['vm', 'mlp', 'tensors']:
            fea, fea_sc, sigma, color, rgb, lr = 0, 0.002, 0.002, 0.002, 1, 0.01
            scene, tea_psnr, bound, scale, gamma = 'chair', 35, 1, 0.8, 0
            tea1 = './log/train_teacher/hash_chair-pnsr33.57/checkpoints/ngp_ep0400.pth'
            cmd = f"python3 main_distill_mutual.py  /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                    --teacher_type hash \
                    --ckpt_teacher {tea1} \
                    --model_type {stu} \
                    --data_type synthetic \
                    --workspace ./log/distill_student/hash2{stu}/chair"
                    # --workspace ./log/conversion/ingp2{stu}/{scene}"
                    # --workspace ./log/test"
            print('\n\n', cmd)
            os.system(cmd)
            assert 1 == 2

    if args.search_mode == "vm2others":
        it, loss_type = 30000, 'normL2'
        for stu in ['ingp', 'vm', 'nerf', 'plenoxel']:
            if stu == 'nerf':
                fea, fea_sc, sigma, color, rgb, lr = 0, 0.002, 0.05, 0.01, 1, 0.001
            else:
                fea, fea_sc, sigma, color, rgb, lr = 0, 0.002, 0.05, 0.01, 1, 0.01
            for tea in tea_vm:
                scene, tea_psnr, bound, scale, gamma = get_params_vm(tea)
                tea1 = '/data/nerf/torch-ngp/' + tea
                cmd = f"python3 main_distill_mutual.py  /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --teacher_type vm \
                        --test_teacher \
                        --ckpt_teacher {tea1} \
                        --model_type {stu} \
                        --data_type synthetic \
                        --mode blender \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --loss_rate_fea {fea} \
                        --loss_rate_fea_sc {fea_sc} \
                        --loss_rate_sigma {sigma} \
                        --loss_rate_color {color} \
                        --loss_rate_rgb {rgb} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --distill_mode no_fix_mlp \
                        --render_stu_first \
                        --loss_rate_real_gt 0 \
                        --workspace ./log/searchpaper/vm2{stu}/0{scene}/tea{tea_psnr}_fea{fea}_feasc{fea_sc}_sigma{sigma}_color{color}_rgb{rgb}_NoFixMLP_iter{it/1000}k_lr{lr}-feascClip-2to7-noEma-firstStu-Loss{loss_type}-cosLrSche-b{bound}s{scale}g{gamma}"
                print('\n\n', cmd)
                os.system(cmd)
                assert 1==2
        assert 1 == 2

    elif args.search_mode == "nerf2others":
        it, loss_type = 30000, 'normL2'
        for stu in ['ingp', 'vm', 'nerf', 'plenoxel']:
            if stu == 'nerf':
                fea, fea_sc, sigma, color, rgb, lr = 0, 0.002, 0.05, 0.01, 1, 0.001
            else:
                fea, fea_sc, sigma, color, rgb, lr = 0, 0.002, 0.05, 0.01, 1, 0.01
            for tea in tea_nerf:
                scene, tea_psnr, bound, scale, gamma = get_params_nerf(tea)
                tea1 = '/data/nerf/torch-ngp/' + tea
                cmd = f"python3 main_distill_mutual.py  /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --teacher_type nerf \
                        --ckpt_teacher {tea1} \
                        --model_type {stu} \
                        --data_type synthetic \
                        --mode blender \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --loss_rate_fea {fea} \
                        --loss_rate_fea_sc {fea_sc} \
                        --loss_rate_sigma {sigma} \
                        --loss_rate_color {color} \
                        --loss_rate_rgb {rgb} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --distill_mode no_fix_mlp \
                        --render_stu_first \
                        --loss_rate_real_gt 0 \
                        --workspace ./log/searchpaper/nerf2{stu}/0{scene}/tea{tea_psnr}_fea{fea}_feasc{fea_sc}_sigma{sigma}_color{color}_rgb{rgb}_NoFixMLP_iter{it/1000}k_lr{lr}-feascClip-2to7-noEma-firstStu-Loss{loss_type}-cosLrSche-b{bound}s{scale}g{gamma}"
                print('\n\n', cmd)
                os.system(cmd)
        assert 1 == 2

    elif args.search_mode == "plenoxel2others":
        it, loss_type = 30000, 'normL2'
        for stu in ['ingp', 'vm', 'nerf', 'plenoxel']:
            if stu == 'nerf':
                fea, fea_sc, sigma, color, rgb, lr = 0, 0.00, 0.05, 0.01, 1, 0.001
            else:
                fea, fea_sc, sigma, color, rgb, lr = 0, 0.00, 0.05, 0.01, 1, 0.01
            for tea in tea_plenoxel:
                scene, tea_psnr, bound, scale, gamma = get_params_plenoxel(tea)
                tea1 = '/data/nerf/torch-ngp/' + tea
                cmd = f"python3 main_distill_mutual.py  /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --teacher_type plenoxel \
                        --ckpt_teacher {tea1} \
                        --model_type {stu} \
                        --data_type synthetic \
                        --mode blender \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --loss_rate_fea {fea} \
                        --loss_rate_fea_sc {fea_sc} \
                        --loss_rate_sigma {sigma} \
                        --loss_rate_color {color} \
                        --loss_rate_rgb {rgb} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --distill_mode no_fix_mlp \
                        --render_stu_first \
                        --loss_rate_real_gt 0 \
                        --workspace ./log/searchpaper/plenoxel2{stu}/0{scene}/tea{tea_psnr}_fea{fea}_feasc{fea_sc}_sigma{sigma}_color{color}_rgb{rgb}_NoFixMLP_iter{it/1000}k_lr{lr}-feascClip-2to7-noEma-firstStu-Loss{loss_type}-cosLrSche-b{bound}s{scale}g{gamma}"
                print('\n\n', cmd)
                assert not os.system(cmd)
        assert 1 == 2


    if args.search_mode == 'ingp2plenoxel-tank':
        tea_ingp_tank = [
                    # 'log/teacher/2Truck/Truck1-lr0.02-50.0k-bs8192-feascclip-2to7-b2s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr28.54/checkpoints/ngp_ep0200.pth',
                    'log/teacher/2Ignatius/Ignatius1-lr0.01-30.0k-bs8192-feascclip-2to7-b3s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr28.26/checkpoints/ngp_ep0115.pth',
                    'log/teacher/2Barn/Barn1-lr0.02-50.0k-bs8192-feascclip-2to7-b2.5s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr28.54/checkpoints/ngp_ep0131.pth',
                    'log/teacher/2Caterpillar/Caterpillar1-lr0.02-50.0k-bs8192-feascclip-2to7-b1.5s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.47/checkpoints/ngp_ep0136.pth',
                    'log/teacher/2Family/Family1-lr0.02-50.0k-bs8192-feascclip-2to7-b1s0.5gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr34.48/checkpoints/ngp_ep0329.pth',
        ]
        '''
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for rgb in [0.1, 1, 5, 10]:
            for fea_sc in [0.001, 0.01, 0.1, 0.5]:
                for color in [0.001, 0.01, 0.1, 0.5]:
                    for fea in [0.0005, 0.001, 0.1, 0.5]:
                        for sigma in [0.001, 0.01, 0.5]:
        '''
        for fea, fea_sc, sigma, color, rgb in [(0., 0.002, 0.002, 0.002, 1)]:
            lr, it = 0.1, 20000
            loss_type = "normL2"
            data_type = 'tank'
            for tea in tea_ingp_tank:
                scene = tea.split('/')[2].strip('2')
                tea_psnr = tea.split('/')[3][-5:]
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound = tea.split('-b')[-1].split('s')[0].split('s')[0]
                if 'gridsize' in tea:
                    grid_size = tea.split('gridsize')[-1].split('-')[0]
                else:
                    grid_size = 128
                gamma = 0
                stu = "plenoxel"
                loss_type = "normL2"
                tea1 = '/data/nerf/torch-ngp/' + tea
                cmd = f"python3 main_distill_mutual.py  /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} \
                        --ckpt_teacher {tea1} \
                        --mode blender \
                        --model_type {stu} \
                        --data_type {data_type} \
                        --teacher_type ingp \
                        --plenoxel_res [128,128,128] \
                        --grid_size {grid_size} \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --loss_rate_fea {fea} \
                        --loss_rate_fea_sc {fea_sc} \
                        --loss_rate_sigma {sigma} \
                        --loss_rate_color {color} \
                        --loss_rate_rgb {rgb} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --distill_mode no_fix_mlp \
                        --render_stu_first \
                        --workspace ./log/conversion/ingp2{stu}/{scene}"
                print('\n\n', cmd)
                os.system(cmd)
            assert 1 == 2

    if args.search_mode == "llff-curri-ingp2vm":
        for fea, fea_sc, sigma, color, rgb in [(0, 0.002, 0.002, 0.002, 1)]:
            tea = 'log/teacher/0fern/tea_fern-3w-bound2-lr0.02-lv14-rs4096-img4-bc4096-feascClip-2to7-scale0.33-gamma0.003-L2-pnsr26.19/checkpoints/ngp_ep1667.pth'
            scene = tea.split('/')[2].strip('0')
            tea_psnr = tea.split('/')[3][-5:]
            scale = tea.split('-scale')[-1].split('-')[0]
            bound = tea.split('-bound')[-1].split('-')[0]
            gamma = tea.split('-gamma')[-1].split('-')[0]
            if 'gridsize' in tea:
                grid_size = tea.split('gridsize')[-1].split('-')[0]
            else:
                grid_size = 128
            res0, res1 = 1024, 1024
            lr, it, randpose = 0.01, 25000, -1
            tea1 = '/data/nerf/torch-ngp/' + tea
            loss_type = "normL2"
            stu = 'vm'
            cmd = f"python3 main_distill_mutual.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/{scene} \
                    --ckpt_teacher {tea1} \
                    --model_type {stu} \
                    --teacher_type ingp \
                    --mode blender \
                    --data_type llff \
                    --grid_size {grid_size} \
                    --loss_type {loss_type} \
                    --bound {bound} \
                    --scale {scale} \
                    --dt_gamma {gamma} \
                    --color_space srgb \
                    --iters {it} \
                    --resolution0 {res0} \
                    --resolution1 {res1} \
                    -O \
                    --lr {lr} \
                    --loss_rate_fea {fea} \
                    --loss_rate_fea_sc {fea_sc} \
                    --loss_rate_sigma {sigma} \
                    --loss_rate_color {color} \
                    --loss_rate_rgb {rgb} \
                    --sigma_clip_min -2 \
                    --sigma_clip_max 7 \
                    --distill_mode no_fix_mlp \
                    --workspace ./log/conversion/ingp2{stu}/{scene}-realRansomPosesPlus1e-6_Only"
            os.system(cmd)
            assert 1 == 2

    else:
        raise ValueError(opt.search_mode)
