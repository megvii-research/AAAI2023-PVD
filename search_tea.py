import os
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--search_mode', type=str, default="room")
    args = parser.parse_args()

    llff_set = [
                'log/teacher/0fern/tea_fern-3w-bound2-lr0.02-lv14-rs4096-img4-bc4096-feascClip-2to7-scale0.33-gamma0.003-L2-pnsr26.19/checkpoints/ngp_ep1667.pth',
                'log/teacher/0orchids/tea_orchids-6k-bound10-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-scale0.33-gamma0.003-L2-pnsr21.73/checkpoints/ngp_ep0273.pth',
                'log/teacher/0flower/flower2-30.0k-bound10-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.005-L2-scale0.03-schelambdalr-adamw-pnsr27.17/checkpoints/ngp_ep1000.pth',
                'log/teacher/0fortress/tea_fortress-3w-bound8-lr0.04-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.004-L2-scale0.33-sc0.005-pnsr30.48/checkpoints/ngp_ep0811.pth',
                'log/teacher/0horns/horns-5w-bound9-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.003-L2-scale0.3-sc0.001-pnsr27.86/checkpoints/ngp_ep0910.pth',
                'log/teacher/0leaves/tea_leaves-5k-bound30-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.001-L2-scale0.005-pnsr21.38/checkpoints/ngp_ep0218.pth',
                'log/teacher/0room/room1-5w-bound2.5-lr0.06-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.002-L2-scale0.6-sc0.2-pnsr31.92/checkpoints/ngp_ep1389.pth',
                'log/teacher/0trex/trex-5k-bound5-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0-L2-scale0.2-pnsr26.84/checkpoints/ngp_ep0103.pth',]

    syn_set = [
                'log/teacher/1lego/lego-lr0.02-4w-bs8192-feascclip-2to7-b1s0.8gamma0-srgb-L2-NoEma-pnsr35.70/checkpoints/ngp_ep0200.pth',
                'log/teacher/1hotdog/hotdog-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr36.60/checkpoints/ngp_ep0250.pth',
                'log/teacher/1chair/chair-lr0.02-5w-bs8192-feascclip-2to7-b1s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.05-pnsr34.33/checkpoints/ngp_ep0250.pth',
                'log/teacher/1drums/drums-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr25.91/checkpoints/ngp_ep0250.pth',
                'log/teacher/1ficus/ficus2-lr0.007-50.0k-bs8192-feascclip-2to7-b1.0s0.75gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr32.87/checkpoints/ngp_ep0250.pth',
                'log/teacher/1hotdog/hotdog-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr36.60/checkpoints/ngp_ep0250.pth',
                'log/teacher/1materials/materials3-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.72gamma0-srgb-L2-NoEma-errorMap-sc0.07-pnsr29.74/checkpoints/ngp_ep0250.pth',
                'log/teacher/1mic/mic-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr35.36/checkpoints/ngp_ep0250.pth',
                'log/teacher/1ship/ship-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr30.27/checkpoints/ngp_ep0250.pth']

    tank_set = [
                'log/teacher/2Ignatius/Ignatius1-lr0.01-30.0k-bs8192-feascclip-2to7-b3s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr28.26/checkpoints/ngp_ep0115.pth',
                'log/teacher/2Truck/Truck1-lr0.02-50.0k-bs8192-feascclip-2to7-b2s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr28.54/checkpoints/ngp_ep0200.pth',
                'log/teacher/2Barn/Barn1-lr0.02-50.0k-bs8192-feascclip-2to7-b2.5s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr28.54/checkpoints/ngp_ep0131.pth',
                'log/teacher/2Caterpillar/Caterpillar1-lr0.02-50.0k-bs8192-feascclip-2to7-b1.5s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.47/checkpoints/ngp_ep0136.pth',
                'log/teacher/2Family/Family1-lr0.02-50.0k-bs8192-feascclip-2to7-b1s0.5gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr34.48/checkpoints/ngp_ep0329.pth',

            ]

# --------------------------------------------------------- synthetic -----------------------------------------------
    if args.search_mode == "synthetic-tea-ingp":
        for lr in [0.01]:
            for tea in syn_set[:1]:
                scene = tea.split('/')[2].strip('1')
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound, gamma = 1, 0
                it = 40000
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --model_type hash \
                        --data_type synthetic \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --iters {it} \
                        --lr {lr} \
                        --workspace ./log/train_teacher/hash/{scene}/L2_lr{lr}_noema_train"
                print('\n\n', cmd)
                os.system(cmd)
                assert 1 == 2

    if args.search_mode == "synthetic-tea-nerf":
        for lr in [0.001]:
            for tea in syn_set:
                scene = tea.split('/')[2].strip('1')
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound, gamma = 1, 0
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --model_type mlp \
                        --data_type synthetic \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --lr {lr} \
                        --workspace ./log/train_teacher/mlp/{scene}-noema-lr{lr}"
                print('\n\n', cmd)
                os.system(cmd)
                assert 1 == 2

    if args.search_mode == "synthetic-tea-vm":
        for lr in [0.01]:
            for tea in syn_set:
                scene = tea.split('/')[2].strip('1')
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound, gamma = 1, 0
                loss_type = "L2"
                tea1 = '/data/nerf/torch-ngp/' + tea
                cmd = f"python3 main_just_train_tea.py /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --model_type vm \
                        --data_type synthetic \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --lr {lr} \
                        --workspace ./log/train_teacher/vm/{scene}-lr{lr}-noema"
                print('\n\n', cmd)
                os.system(cmd)
                assert 1 == 2

    if args.search_mode == "synthetic-tea-plenoxel":
        for lr in [0.01]:
            for tea in syn_set:
                scene = tea.split('/')[2].strip('1')
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound, gamma = 1, 0
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                        --model_type tensors \
                        --data_type synthetic \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --workspace ./log/train_teacher/tensors/{scene}-test"
                print('\n\n', cmd)
                os.system(cmd)
                assert 1 == 2

# --------------------------------------------------------- llff -----------------------------------------------

    if args.search_mode == "llff-tea-nerf":
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for lr in [0.001]:
            for tea in llff_set:
                gamma1 = tea.split('-gamma')[-1].split('-')[0]
                for gamma in [gamma1, 0.]:
                    scene = tea.split('/')[2].strip('0')
                    tea_psnr = tea.split('/')[3][-5:]
                    scale = tea.split('-scale')[-1].split('-')[0]
                    bound = tea.split('-bound')[-1].split('-')[0]
                    if 'gridsize' in tea:
                        grid_size = tea.split('gridsize')[-1].split('-')[0]
                    else:
                        grid_size = 128
                    pe, layer_num, layer_wide, skip = 10, 8, 256, 3
                    it, randpose = 30000, -1
                    loss_type = "L2"
                    tea1 = '/data/nerf/torch-ngp/' + tea
                    cmd = f"python3 main_just_train_tea.py /data/nerf/torch-ngp/datasets/nerf_llff_data/{scene} \
                            --ckpt_teacher {tea1} \
                            --model_type nerf \
                            --data_type llff \
                            --teacher_type nerf \
                            --plenoxel_degree 3 \
                            --just_train_a_model \
                            --mode blender \
                            --loss_type {loss_type} \
                            --bound {bound} \
                            --scale {scale} \
                            --dt_gamma {gamma} \
                            --color_space srgb \
                            --iters {it} \
                            -O \
                            --lr {lr} \
                            --nerf_pe \
                            --PE {pe} \
                            --nerf_layer_num {layer_num} \
                            --nerf_layer_wide {layer_wide} \
                            --skip {skip} \
                            --sigma_clip_min -2 \
                            --sigma_clip_max 7 \
                            --render_stu_first \
                            --distill_mode no_fix_mlp \
                            --workspace ./log/teacher/nerf/1{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-{pe}pe{layer_num}MLP{layer_wide}hidden{skip}skip"
                    print('\n\n', cmd)
                    try:
                        os.system(cmd)
                    except:
                        with open('log_fail.txt', 'a+') as f:
                            txt = './log/teacher/nerf/1{scene}/{scene}-{it/1000}k-lr{lr}'
                            f.write(txt)
        assert 1 == 2

    if args.search_mode == "llff-tea-vm":
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for lr in [0.05, 0.08]:
            for tea in llff_set:
                scene = tea.split('/')[2].strip('0')
                tea_psnr = tea.split('/')[3][-5:]
                scale = tea.split('-scale')[-1].split('-')[0]
                bound = tea.split('-bound')[-1].split('-')[0]
                gamma = tea.split('-gamma')[-1].split('-')[0]
                res0, res1 = 640, 640
                # gamma = 0
                it, randpose = 30000, -1
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/torch-ngp/datasets/nerf_llff_data/{scene} \
                        --update_stu_extra \
                        --model_type vm \
                        --data_type llff \
                        --teacher_type ingp \
                        --just_train_a_model \
                        --mode blender \
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
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --render_stu_first \
                        --distill_mode no_fix_mlp \
                        --workspace ./log/train_teacher/vm/{scene}"
                print('\n\n', cmd)
                try:
                    os.system(cmd)
                except:
                    with open('log_fail.txt', 'a+') as f:
                        f.write(f'/log/teacher/tensorf/1{scene}/{scene}-{it//1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-res{res0}\n')
                assert 1 == 2

    if args.search_mode == "llff-tea-plenoxel":
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for lr in [0.02]:
            for tea in llff_set:
                gamma1 = tea.split('-gamma')[-1].split('-')[0]
                for gamma in [gamma1]:
                    scene = tea.split('/')[2].strip('0')
                    tea_psnr = tea.split('/')[3][-5:]
                    scale = tea.split('-scale')[-1].split('-')[0]
                    bound = tea.split('-bound')[-1].split('-')[0]
                    if 'gridsize' in tea:
                        grid_size = tea.split('gridsize')[-1].split('-')[0]
                    else:
                        grid_size = 128
                    it, randpose = 30000, -1
                    loss_type = "L2"
                    tea1 = '/data/nerf/torch-ngp/' + tea
                    cmd = f"python3 main_just_train_tea.py /data/nerf/torch-ngp/datasets/nerf_llff_data/{scene} \
                            --ckpt_teacher {tea1} \
                            --model_type plenoxel \
                            --data_type llff \
                            --teacher_type nerf \
                            --plenoxel_degree 3 \
                            --just_train_a_model \
                            --mode blender \
                            --loss_type {loss_type} \
                            --plenoxel_res [512,512,128] \
                            --plenoxel_degree 3 \
                            --bound {bound} \
                            --scale {scale} \
                            --dt_gamma {gamma} \
                            --color_space srgb \
                            --iters {it} \
                            -O \
                            --lr {lr} \
                            --sigma_clip_min -2 \
                            --sigma_clip_max 7 \
                            --render_stu_first \
                            --distill_mode no_fix_mlp \
                            --workspace ./log/teacher/plenoxel/1{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-plenoxel512512128"
                    print('\n\n', cmd)
                    try:
                        os.system(cmd)
                    except:
                        with open('log_fail.txt', 'a+') as f:
                            txt = './log/teacher/plenoxel/1{scene}/{scene}-{it/1000}k-lr{lr}'
                            f.write(txt)
        assert 1 == 2


# --------------------------------------------------------- tanksandtemples-----------------------------------------------
    if args.search_mode == "tank-tea-nerf":
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for lr in [0.001]:
            for tea in tank_set:
                scene = tea.split('/')[2].strip('2')
                tea_psnr = tea.split('/')[3][-5:]
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound = tea.split('-b')[-1].split('s')[0].split('s')[0]
                if 'gridsize' in tea:
                    grid_size = tea.split('gridsize')[-1].split('-')[0]
                else:
                    grid_size = 128
                gamma = 0
                pe, layer_num, layer_wide, skip = 10, 8, 256, 3
                # res = [128, 128, 128]
                it, randpose = 50000, -1
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} \
                        --ckpt_teacher {tea} \
                        --model_type nerf \
                        --data_type tank \
                        --teacher_type nerf \
                        --plenoxel_degree 3 \
                        --just_train_a_model \
                        --mode blender \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --nerf_pe \
                        --PE {pe} \
                        --nerf_layer_num {layer_num} \
                        --nerf_layer_wide {layer_wide} \
                        --skip {skip} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --render_stu_first \
                        --distill_mode no_fix_mlp \
                        --workspace ./log/teacher/nerf/2{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-{pe}pe{layer_num}MLP{layer_wide}hidden{skip}skip"
                print('\n\n', cmd)
                try:
                    os.system(cmd)
                except:
                    with open('log_fail.txt', 'a+') as f:
                        txt = './log/teacher/nerf/2{scene}/{scene}-{it/1000}k-lr{lr}'
                        f.write(txt)
        assert 1 == 2

    if args.search_mode == "tank-tea-vm":
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for lr in [0.02]:
            for tea in tank_set:
                scene = tea.split('/')[2].strip('2')
                tea_psnr = tea.split('/')[3][-5:]
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound = tea.split('-b')[-1].split('s')[0].split('s')[0]
                if 'gridsize' in tea:
                    grid_size = tea.split('gridsize')[-1].split('-')[0]
                else:
                    grid_size = 128
                gamma = 0

                # res = [128, 128, 128]
                it, randpose = 15000, -1
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} \
                        --ckpt_teacher {tea} \
                        --model_type plenoxel \
                        --data_type synthetic \
                        --teacher_type nerf \
                        --plenoxel_res [128,128,128] \
                        --plenoxel_degree 3 \
                        --just_train_a_model \
                        --mode blender \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --render_stu_first \
                        --distill_mode no_fix_mlp \
                        --workspace ./log/teacher/plenoxel/2{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-res128"
                print('\n\n', cmd)
                try:
                    os.system(cmd)
                except:
                    with open('log_fail.txt', 'a+') as f:
                        txt = './log/teacher/plenoxel/2{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-res128'
                        f.write(txt)
        assert 1 == 2

    if args.search_mode == "tank-tea-plenoxel":
        fea, fea_sc, sigma, color, rgb = 0.01, 0.05, 0.0, 0.5, 1
        for lr in [0.02]:
            for tea in tank_set:
                scene = tea.split('/')[2].strip('2')
                tea_psnr = tea.split('/')[3][-5:]
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound = tea.split('-b')[-1].split('s')[0].split('s')[0]
                if 'gridsize' in tea:
                    grid_size = tea.split('gridsize')[-1].split('-')[0]
                else:
                    grid_size = 128
                gamma = 0

                # res = [128, 128, 128]
                it, randpose = 15000, -1
                loss_type = "L2"
                cmd = f"python3 main_just_train_tea.py /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} \
                        --ckpt_teacher {tea} \
                        --model_type plenoxel \
                        --data_type synthetic \
                        --teacher_type nerf \
                        --plenoxel_res [128,128,128] \
                        --plenoxel_degree 3 \
                        --just_train_a_model \
                        --mode blender \
                        --loss_type {loss_type} \
                        --bound {bound} \
                        --scale {scale} \
                        --dt_gamma {gamma} \
                        --color_space srgb \
                        --iters {it} \
                        -O \
                        --lr {lr} \
                        --sigma_clip_min -2 \
                        --sigma_clip_max 7 \
                        --render_stu_first \
                        --distill_mode no_fix_mlp \
                        --workspace ./log/teacher/plenoxel/2{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-res128"
                print('\n\n', cmd)
                try:
                    os.system(cmd)
                except:
                    with open('log_fail.txt', 'a+') as f:
                        txt = './log/teacher/plenoxel/2{scene}/{scene}-{it/1000}k-lr{lr}-feascClip-2to7-gamma{gamma}-{loss_type}-bound{bound}-scale{scale}-res128'
                        f.write(txt)
        assert 1 == 2




# ------------------------------- just evalation --------------------------
    if args.search_mode == "synthetic-ngp2nerf-eval":
        for tea in [
                # 'log/searchpaper/ngp2nerf/0lego/nerf_tea35.7_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter6k_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE12-256MLP8num3skip-LossnormL2-cosLrSche-pnsr32.58-1840.70s/checkpoints/ngp_ep0150.pth',
                'log/searchpaper/ngp2nerf/0chair/nerf_tea34.33_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter30.0k_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-512MLP8num3skip-LossnormL2-cosLrSche-b1s0.8g0-randpose0-fixData-pnsr31.58-2453.73s/checkpoints/ngp_ep0150.pth',
                'log/searchpaper/ngp2nerf/0drums/nerf_tea25.91_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter3w_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-256MLP8num3skip-LossnormL2-cosLrSche-b1s0.8g0-pnsr25.29-1041.67s/checkpoints/ngp_ep0150.pth',
                'log/searchpaper/ngp2nerf/0ficus/nerf_tea32.87_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter3w_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-256MLP8num3skip-LossnormL2-cosLrSche-b1s0.75g0-randpose1-pnsr31.63-724.75s/checkpoints/ngp_ep0075.pth',
                'log/searchpaper/ngp2nerf/0hotdog/nerf_tea36.60_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter3w_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-256MLP8num3skip-LossnormL2-cosLrSche-b1s0.67g0-pnsr34.92-1556.74s/checkpoints/ngp_ep0150.pth',
                'log/searchpaper/ngp2nerf/0materials/nerf_tea29.74_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter3w_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-256MLP8num3skip-LossnormL2-cosLrSche-b1s0.72g0-pnsr29.14-1553.22s/checkpoints/ngp_ep0150.pth',
                'log/searchpaper/ngp2nerf/0mic/nerf_tea35.36_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter3w_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-256MLP8num3skip-LossnormL2-cosLrSche-b1s0.8g0-pnsr32.68-729.44s/checkpoints/ngp_ep0150.pth',
                'log/searchpaper/ngp2nerf/0ship/nerf_tea30.27_fea0.0005_feasc0.001_sigma0.001_color0.001_rgb0.1_NoFixMLP_iter3w_lr0.001_adamw-ray8192-feascClip-2to7-noEma-firstStu-PE10-256MLP8num3skip-LossnormL2-cosLrSche-b1s0.67g0-pnsr28.17-2060.90s/checkpoints/ngp_ep0150.pth',
                ]:
            scene = tea.split('/')[3].strip('0')
            tea_psnr = tea.split('pnsr')[-1][:5]
            scale = tea.split('-b')[-1].split('s')[1].split('g0')[0]
            print(scene, tea_psnr, scale)

            if 'gridsize' in tea:
                grid_size = tea.split('gridsize')[-1].split('-')[0]
            else:
                grid_size = 128
            bound, gamma = 1, 0
            lr, it, randpose = 0.0012, 50000, 0
            loss_type = "normL2"
            pe, layer_num, layer_wide, skip = 10, 8, 256, 3
            if 'chair' in tea: layer_wide = 512
            cmd = f"python3 main_distill_mutual.py  /data/nerf/IBRNet/data/nerf_synthetic/{scene} \
                    --ckpt_teacher {tea} \
                    --model_type nerf \
                    --teacher_type nerf \
                    --test_teacher \
                    --mode blender \
                    --grid_size {grid_size} \
                    --loss_type {loss_type} \
                    --bound {bound} \
                    --scale {scale} \
                    --dt_gamma {gamma} \
                    --color_space srgb \
                    --iters {it} \
                    -O \
                    --lr {lr} \
                    --sigma_clip_min -2 \
                    --sigma_clip_max 7 \
                    --distill_mode no_fix_mlp \
                    --render_stu_first \
                    --nerf_pe \
                    --PE {pe} \
                    --nerf_layer_num {layer_num} \
                    --nerf_layer_wide {layer_wide} \
                    --skip {skip} \
                    --loss_rate_real_gt 0 \
                    --workspace ./log/test "
            print('\n\n', cmd)
            os.system(cmd)
        assert 1 == 2


    if args.search_mode == "leaves":
        for lr in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
            for gamma in [0, 0.001, 0.002, 0.005, 0.007, 0.01]:
                cmd = f"python3 main_nerf.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/leaves --mode blender --iters 5000 --workspace ./log/teacher/0leaves/tea_leaves-5k-bound30-lr{lr}-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma{gamma}-L2-scale0.005  --bound 30 --lr {lr} -O  --dt_gamma {gamma}  --loss_type L2 --scale 0.005"
                print('\n\n', cmd)
                os.system(cmd)

    if args.search_mode == "flower":
        # flower2-30.0k-bound10-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.005-L2-scale0.03-schelambdalr-adamw-pnsr27.17/
        # flower-5.0k-bound10-lr0.04-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.005-L2-scale0.05-sc0.05-pnsr26.82/
        for lr in [0.04, 0.02]:
            gamma = 0.005
            for scale in [0.03, 0.04, 0.05]:
                for sc in [0.1, 0.05]:
                    it = 30000
                    cmd = f"python3 main_nerf.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/flower --mode blender --iters {it} --workspace ./log/teacher/0flower/flower3-boundClip-{it/1000}k-bound10-lr{lr}-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma{gamma}-L2-scale{scale}-sc{sc}-adamw  --bound 10 --lr {lr} -O  --dt_gamma {gamma}  --loss_type L2 --scale {scale} --sc_lr {sc}"
                    print('\n\n', cmd)
                    os.system(cmd)
                    # assert 1==2

    if args.search_mode == "fortress":
        # tea_fortress-5w-bound8-lr0.04-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.004-L2-scale0.33-sc0.005-pnsr30.46
        # fortress-lowbound-lr0.01-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.004-L2-b2.5s0.15-adamw-pnsr28.77
        for bound in [2.5]:
            for scale in [0.12, 0.13, 0.14, 0.16, 0.17, 0.18]:
                # bound, scale = 8, 0.33
                lr, gamma = 0.01, 0.004
                cmd = f"python3 main_nerf.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/fortress --mode blender --iters 5000 --workspace ./log/teacher/0fortress/fortress-lowbound-lr{lr}-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma{gamma}-L2-b{bound}s{scale}-iter5k-adamw  --bound {bound} --lr {lr} -O  --dt_gamma {gamma}  --loss_type L2 --scale {scale}"
                print('\n\n', cmd)
                os.system(cmd)
        assert "search" == "over"

    if args.search_mode == "horns":
        # horns-5k-bound9-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.003-L2-scale0.3-pnsr27.02/
        bound, scale = 9, 0.3
        lr, gamma = 0.02, 0.003
        for sc in [0.05, 0.01, 0.005, 0.001, 0.0005]:
            cmd = f"python3 main_nerf.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/horns --mode blender --iters 50000 --workspace ./log/teacher/0horns/horns-5w-bound{bound}-lr{lr}-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma{gamma}-L2-scale{scale}-sc{sc}  --bound {bound} --lr {lr} -O --dt_gamma {gamma}  --loss_type L2 --scale {scale} --sc_lr {sc}"
            print('\n\n', cmd)
            os.system(cmd)

    if args.search_mode == "trex":
        bound, scale = 9, 0.3
        for lr in [0.01, 0.02, 0.03, 0.04, 0.05]:
            for gamma in [0, 0.001, 0.002, 0.003, 0.004, 0.005]:
                bound, scale = 5, 0.2
                cmd = f"python3 main_nerf.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/trex --mode blender --iters 5000 --workspace ./log/teacher/0trex/trex-5k-bound{bound}-lr{lr}-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma{gamma}-L2-scale{scale}  --bound {bound} --lr {lr} -O --dt_gamma {gamma}  --loss_type L2 --scale {scale}"
                print('\n\n', cmd)
                os.system(cmd)

    if args.search_mode == "room":
        # tea_room-5k-bound2.5-lr0.02-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma0.002-L2-scale0.6-pnsr31.90/
        bound, scale, gamma = 2.5, 0.6, 0.002
        for lr in [0.005, 0.01, 0.02, 0.03, 0.04, 0.06]:
            for sc in [0.01, 0.05, 0.1, 0.2]:
                cmd = f"python3 main_nerf.py  /data/nerf/torch-ngp/datasets/nerf_llff_data/room --mode blender --iters 50000 --workspace ./log/teacher/0room/room1-5w-bound{bound}-lr{lr}-lv14-rs4096-img4-bc8192-feascClip-2to7-gamma{gamma}-L2-scale{scale}-sc{sc}  --bound {bound} --lr {lr} -O --dt_gamma {gamma}  --loss_type L2 --scale {scale} --sc_lr {sc}"
                print('\n\n', cmd)
                os.system(cmd)

    if args.search_mode == "chair":
        for bound in [0.8, 1.0, 1.5, 2.0, 3.0]:
            for scale in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
                for sc in [0.1, 0.07, 0.05, 0.03]:
                    bound, scale = 0.8, 0.6
                    cmd = f'python3 main_nerf.py  /data/nerf/IBRNet/data/nerf_synthetic/chair --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters 50000 -O --workspace ./log/teacher/1chair/chair-lr0.02-5w-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc} --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                    os.system(cmd)
                assert 'search' == "over"

    if args.search_mode == "drums":
        for bound in [0.8, 1.0, 1.5, 2.0, 3.0]:
            for scale in [0.4, 0.6, 0.8, 0.9]:
                # for sc in [0.1, 0.05]:
                sc, it = 0.1, 50000
                cmd = f'python3 main_nerf.py  /data/nerf/IBRNet/data/nerf_synthetic/drums --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/1drums/drums-lr0.02-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc} --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)
                # assert 'search' == "over"

    if args.search_mode == "materials":
        for lr in [0.002, 0.005, 0.01, 0.015, 0.025, 0.03, 0.04]:
            for bound in [1.0]:
                for scale in [0.72]:
                    sc = 0.07
                    gamma, it = 0, 50000
                    cmd = f'python3 main_nerf.py  /data/nerf/IBRNet/data/nerf_synthetic/materials --mode blender --bound {bound} --scale {scale} --color_space srgb --iters {it} -O --workspace ./log/teacher/1materials/materials4-lr{lr}-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma{gamma}-srgb-L2-NoEma-errorMap-sc{sc} --lr {lr} --loss_type L2 --error_map --sc_lr {sc} --dt_gamma {gamma}'
                    os.system(cmd)
                    # assert 'search' == "over"

    if args.search_mode == "ficus":
        for lr in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]:
            for scale in [0.75]:
                # for sc in [0.1, 0.05]:
                sc, it, bound = 0.1, 50000, 1.0
                cmd = f'python3 main_nerf.py  /data/nerf/IBRNet/data/nerf_synthetic/ficus --mode blender --bound {bound} --scale {scale} --color_space srgb --iters {it} -O --workspace ./log/teacher/1ficus/ficus2-lr{lr}-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-adamw --lr {lr} --loss_type L2 --error_map --sc_lr {sc} --dt_gamma 0'
                os.system(cmd)
                # assert 'search' == "over"

    if args.search_mode == "ship":
        for lr in [0.01, 0.02, 0.04]:
            for sc in [0.05, 0.1, 0.2]:
                bound, scale = 2.0, 0.8
                # for sc in [0.1, 0.05]:
                it = 30000
                cmd = f'python3 main_nerf.py  /data/nerf/IBRNet/data/nerf_synthetic/ship --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/1ship/ship-lr{lr}-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc} --lr {lr} --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)
                # assert 'search' == "over"

    if args.search_mode == "synthetic":
        bound, scale = 1.0, 0.67
        # mic-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr35.36
        for sc in ['0.3', '0.1', '0.2', '0.05', '0.01']:
            # for scene in ['mic', 'ficus', 'hotdog', 'materials', 'mic', 'ship', 'chair']:
            for tea in ['log/teacher/1drums/drums-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr25.91/checkpoints/ngp_ep0250.pth',
                        'log/teacher/1ficus/ficus2-lr0.007-50.0k-bs8192-feascclip-2to7-b1.0s0.75gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr32.87/checkpoints/ngp_ep0250.pth',
                        'log/teacher/1hotdog/hotdog-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr36.60/checkpoints/ngp_ep0250.pth',
                        'log/teacher/1materials/materials3-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.72gamma0-srgb-L2-NoEma-errorMap-sc0.07-pnsr29.74/checkpoints/ngp_ep0250.pth',
                        'log/teacher/1mic/mic-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.8gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr35.36/checkpoints/ngp_ep0250.pth',
                        'log/teacher/1ship/ship-lr0.02-50.0k-bs8192-feascclip-2to7-b1.0s0.67gamma0-srgb-L2-NoEma-errorMap-sc0.1-pnsr30.27/checkpoints/ngp_ep0250.pth'
                        ]:
                scene = tea.split('/')[2].strip('1')
                scale = tea.split('-b')[-1].split('-s')[0].split('s')[-1].split('gamma')[0]
                bound, gamma = 1, 0
                it = 50000
                lr, sc = 0.02, 0.1
                cmd = f'python3 main_nerf.py  /data/nerf/IBRNet/data/nerf_synthetic/{scene} --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/1{scene}/{scene}-lr0.02-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-gridsize64 --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)
            assert 'search' == "over"

    # ------------------------------------------ TanksAndTemple ---------------------------------------

    elif args.search_mode == "tankandtemple":
        # Ignatius-lr0.02-5.0k-bs8192-feascclip-2to7-b2s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr27.47
        # Ignatius-lr0.02-5.0k-bs8192-feascclip-2to7-b3s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr27.62
        for bound, scale in [(2, 0.33), (3, 0.33), (3, 0.2)]:
            for lr in [0.02, 0.01, 0.04]:
                for scene in ['Ignatius']:  # ['Barn', 'Caterpillar', 'Family', 'Truck']:
                    sc, it = 0.1, 30000
                    cmd = f'python3 main_nerf.py  /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/2{scene}/{scene}1-lr{lr}-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-adamw --lr {lr} --loss_type L2 --error_map --sc_lr {sc}'
                    os.system(cmd)

    elif args.search_mode == "Truck":
        # Truck-lr0.02-5.0k-bs8192-feascclip-2to7-b1.5s0.38gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.90
        # Truck-lr0.02-5.0k-bs8192-feascclip-2to7-b2s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.97
        # Truck-lr0.02-5.0k-bs8192-feascclip-2to7-b3s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.91
        for bound, scale in [(1.5, 0.38), (2, 0.6), (3, 0.6)]:
            for scene in ['Truck']:
                sc, it = 0.1, 50000
                cmd = f'python3 main_nerf.py  /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/2{scene}/{scene}1-lr0.02-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-adamw --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)

    elif args.search_mode == "Barn":
        # Barn-lr0.02-5.0k-bs8192-feascclip-2to7-b2.5s0.5gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.51
        # Barn-lr0.02-5.0k-bs8192-feascclip-2to7-b1.5s0.35gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.35
        # Barn-lr0.02-5.0k-bs8192-feascclip-2to7-b2.5s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr26.56
        for bound, scale in [(2.5, 0.61), (2.5, 0.5), (1.5, 0.35), (2.5, 0.6)]:
            for scene in ['Barn']:
                sc, it = 0.1, 50000
                cmd = f'python3 main_nerf.py  /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/2{scene}/{scene}1-lr0.02-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-adamw --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)
                assert 1 == 2

    elif args.search_mode == "Caterpillar":
        # Caterpillar-lr0.02-5.0k-bs8192-feascclip-2to7-b1.5s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr24.60
        # Caterpillar-lr0.02-5.0k-bs8192-feascclip-2to7-b1.5s0.5gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr24.59
        # Caterpillar-lr0.02-5.0k-bs8192-feascclip-2to7-b2s0.5gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr24.58
        for bound, scale in [(1.5, 0.33), (1.5, 0.5), (2, 0.5)]:
            for scene in ['Caterpillar']:
                sc, it = 0.1, 50000
                cmd = f'python3 main_nerf.py  /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/2{scene}/{scene}1-lr0.02-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-adamw --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)

    elif args.search_mode == "Family":
        # Family-lr0.02-5.0k-bs8192-feascclip-2to7-b1s0.5gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr32.38
        # Family-lr0.02-5.0k-bs8192-feascclip-2to7-b1s0.33gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr32.23
        # Family-lr0.02-5.0k-bs8192-feascclip-2to7-b1.5s0.6gamma0-srgb-L2-NoEma-errorMap-sc0.1-adamw-pnsr32.06
        for bound, scale in [(1, 0.5), (1, 0.33), (1.5, 0.6)]:
            for scene in ['Family']:
                sc, it = 0.1, 50000
                cmd = f'python3 main_nerf.py  /data/nerf/torch-ngp/datasets/TanksAndTemple/{scene} --mode blender --bound {bound} --scale {scale} --dt_gamma 0 --color_space srgb --iters {it} -O --workspace ./log/teacher/2{scene}/{scene}1-lr0.02-{it/1000}k-bs8192-feascclip-2to7-b{bound}s{scale}gamma0-srgb-L2-NoEma-errorMap-sc{sc}-adamw --lr 0.02 --loss_type L2 --error_map --sc_lr {sc}'
                os.system(cmd)

