import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F

from tools.encoding import get_encoder
from tools.activation import trunc_exp
from .renderer import NeRFRenderer
import raymarching


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        encoding="hashgrid",
        encoding_dir="sphere_harmonics",
        encoding_bg="hashgrid",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        num_layers_bg=2,
        hidden_dim_bg=64,
        bound=1,
        model_type="hash",
        args=None,
        is_teacher=False,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)
        # sigma network
        assert model_type in ["hash", "mlp", "vm", "tensors"]
        self.is_teacher = is_teacher
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.args = args
        self.opt = args
        self.model_type = model_type

        self.plenoxel_degree = args.plenoxel_degree
        self.plenoxel_res = eval(args.plenoxel_res)

        assert len(self.plenoxel_res) == 3

        self.encoder, self.in_dim = get_encoder(
            encoding, desired_resolution=2048 * bound, num_levels=14,
        )

        if 'hash' != self.model_type:
            self.encoder = None

        if self.model_type == "mlp":
            self.encoder_nerf_pe, self.in_dim_nerf = get_encoder(encoding="frequency", multires=self.args.PE)
            self.skips = self.args.skip
            self.nerf_layer_num = self.args.nerf_layer_num
            W = self.args.nerf_layer_wide
            self.nerf_mlp = [nn.Linear(self.in_dim_nerf, W)]
            for i in range(self.nerf_layer_num - 2):
                if i != self.skips:
                    self.nerf_mlp.append(nn.Linear(W, W))
                else:
                    self.nerf_mlp.append(nn.Linear(W + self.in_dim_nerf, W))
            self.nerf_mlp.append(nn.Linear(W, self.in_dim))
            self.nerf_mlp = nn.ModuleList(self.nerf_mlp)

        elif self.model_type == "vm":
            self.sigma_rank = [16] * 3
            self.color_rank = [48] * 3
            self.color_feat_dim = 15  # geo_feat_dim
            self.mat_ids = [[0, 1], [0, 2], [1, 2]]
            self.vec_ids = [2, 1, 0]
            self.resolution = [self.opt.resolution0] * 3
            # mat: paralist[1,16,res0,res0] repeat 3   vec: paralist[1,16,res0,1] repeat 3; repeat3 because decompose 3D grid [H, W, D] to three 2D mat [H, W], [H,D], [W, D] or decompose to three 1D vec [H], [W], [D]
            self.sigma_mat, self.sigma_vec = self.init_one_vm(self.sigma_rank, self.resolution)
            # mat: paralist[1,48,res0,res0] repeat 3   vec: paralist[1,48,res0,1] repeat 3
            self.color_mat, self.color_vec = self.init_one_vm(self.color_rank, self.resolution)
            # Linear(in_features=144, out_features=27)
            self.basis_mat = nn.Linear(sum(self.color_rank), self.color_feat_dim, bias=False)
        elif self.model_type == "tensors":
            self.init_plenoxel_volume(s=0.02, fea_dim= self.plenoxel_degree**2*3+1, volume=self.plenoxel_res)

        elif self.model_type == "hash":
            pass
        else:
            raise ValueError(f"error model_type:{self.model_type}")

        if self.model_type != "vm" and self.model_type != "tensors":
            sigma_net = []
            for l in range(num_layers):
                if l == 0:
                    in_dim = self.in_dim
                else:
                    in_dim = hidden_dim

                if l == num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim

                sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        # self.encoder_dir, self.in_dim_dir = get_encoder(encoding=encoding_dir)
        if self.model_type == "tensors":
            self.encoder_dir, self.in_dim_dir = get_encoder(
                encoding="sphere_harmonics", degree=self.plenoxel_degree,
            )

        else:
            self.encoder_dir, self.in_dim_dir = get_encoder(
                encoding=encoding_dir, input_dim=3, multires=2
            )

        if self.model_type != "tensors":
            color_net = []
            for l in range(num_layers_color):
                if l == 0:
                    in_dim = self.in_dim_dir + self.geo_feat_dim
                else:
                    in_dim = hidden_dim

                if l == num_layers_color - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim

                color_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(
                encoding_bg,
                input_dim=2,
                num_levels=4,
                log2_hashmap_size=19,
                desired_resolution=2048,
            )  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def init_plenoxel_volume(self, s=0.1, fea_dim=27+1, volume=[128, 128, 128]):
        tensor = []
        tensor.append(
            torch.nn.Parameter(
                s * torch.randn((1, fea_dim, volume[0], volume[1], volume[2]))
            )
        )
        self.tensor_volume = torch.nn.ParameterList(tensor).cuda()

    def init_one_vm(self, n_component, resolution, scale=0.1):
        # self.mat_ids = [[0, 1], [0, 2], [1, 2]]  self.vec_ids = [2, 1, 0]
        mat, vec = [], []

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]
            mat.append(nn.Parameter(scale * torch.randn((1, n_component[i], resolution[mat_id_1], resolution[mat_id_0]))))  # [1, R, H, W]
            vec.append(nn.Parameter(scale * torch.randn((1, n_component[i], resolution[vec_id], 1))))  # [1, R, D, 1] (fake 2d to use grid_sample)

        return nn.ParameterList(mat), nn.ParameterList(vec)

    def get_sigma_feat(self, x):
        # x: [N, 3], in [-1, 1] (outliers will be treated as zero due to grid_sample padding mode)
        # self.mat_ids = [[0, 1], [0, 2], [1, 2]]  self.vec_ids = [2, 1, 0]
        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).detach().view(3, -1, 1, 2)  # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2)  # [3, N, 1, 2], fake 2d coord

        sigma_feat = torch.zeros([N,], device=x.device)

        for i in range(len(self.sigma_mat)):
            mat_feat = F.grid_sample(self.sigma_mat[i], mat_coord[[i]], align_corners=True).view(-1, N) # [1, R, N, 1] --> [R, N]
            vec_feat = F.grid_sample(self.sigma_vec[i], vec_coord[[i]], align_corners=True).view(-1, N) # [R, N]
            sigma_feat = sigma_feat + torch.sum(mat_feat * vec_feat, dim=0)

        return sigma_feat

    def get_color_feat(self, x):
        # x: [N, 3], in [-1, 1]
        N = x.shape[0]

        # plane + line basis
        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        mat_feat, vec_feat = [], []
        for i in range(len(self.color_mat)):
            mat_feat.append(F.grid_sample(self.color_mat[i], mat_coord[[i]], align_corners=True).view(-1, N)) # [1, R, N, 1] --> [R, N]
            vec_feat.append(F.grid_sample(self.color_vec[i], vec_coord[[i]], align_corners=True).view(-1, N)) # [R, N]
        
        mat_feat = torch.cat(mat_feat, dim=0) # [3 * R, N]
        vec_feat = torch.cat(vec_feat, dim=0) # [3 * R, N]

        color_feat = self.basis_mat((mat_feat * vec_feat).T) # [N, 3R] --> [N, color_feat_dim]

        return color_feat

    def compute_plenoxel_fea(self, x):
        composed = self.tensor_volume[0]
        if self.args.enable_edit_plenoxel and self.is_teacher:
            composed[:, 0, :, 160:, :128] = -100  # This will erase the bucket in the lego scene for resolution 256
        composed = (
            F.grid_sample(composed, x.view(1, 1, -1, 1, 3), align_corners=True)
            .view(-1, x.shape[0])
            .permute(1, 0)
        )
        return composed  # [N, fea_dim]

    def forward_nerf_mlp(self, x):
        x = self.encoder_nerf_pe(x)
        in_pts = x
        for i in range(len(self.nerf_mlp)):
            x = self.nerf_mlp[i](x)
            if i != len(self.nerf_mlp) - 1:
                x = F.relu(x, inplace=True)
            if i == self.skips:
                x = torch.cat([in_pts, x], -1)
        return x

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]  d: [N, 3], nomalized in [-1, 1]
        # sigma
        if self.model_type == "hash":
            x = self.encoder(x, bound=self.bound)  # out_x[N, 28=num_levels * fea_per_level]
        elif self.model_type == "mlp":
            x = self.forward_nerf_mlp(x)  # 28
        elif self.model_type == "vm":
            x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1  # x:[N, 3]
            sigma_feat = self.get_sigma_feat(x)  # sigma_feat:[N]
            color_feat = self.get_color_feat(x)  # color_feat:[N, 15]
            if self.opt.enable_edit_plenoxel:
                sigma_feat = torch.clamp(sigma_feat, -100, self.args.sigma_clip_max)
            else:
                sigma_feat = torch.clamp(sigma_feat, self.args.sigma_clip_min, self.args.sigma_clip_max)
            color_feat = torch.clamp(color_feat, self.args.sigma_clip_min, self.args.sigma_clip_max)
            self.feature_sigma_color = torch.cat([sigma_feat.unsqueeze(-1), color_feat], dim=-1)
            if self.training and self.args.global_step < self.args.stage_iters['stage1']:
                return None, None
            self.sigma_l = sigma_feat
            sigma = trunc_exp(sigma_feat)  # sigma:[N]
            enc_d = self.encoder_dir(d)  # enc_d:[N, 16]
            h = torch.cat([enc_d, color_feat], dim=-1)  # h:[N, 16+15]
            for l in range(self.num_layers_color):
                h = self.color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)

            color = torch.sigmoid(h)
            self.color_l = color

            return sigma, color
        elif self.model_type == "tensors":
            x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1  # x:[N, 3]
            x = self.compute_plenoxel_fea(x)
            h = x
            if self.opt.enable_edit_plenoxel:
                sigma = torch.clamp(h[..., 0], -100, self.args.sigma_clip_max)
            else:
                sigma = torch.clamp(h[..., 0], self.args.sigma_clip_min, self.args.sigma_clip_max)
            self.sigma_l = sigma
            sigma = trunc_exp(sigma)
            self.sigma = sigma
            sh = h[..., 1:].view(-1, 3, self.plenoxel_degree**2)  # [N, 3, 9]   ## .permute(1, 0, 2)  # [B, 27]-->[9, B, 3]
            enc_d = self.encoder_dir(d).unsqueeze(1)  # [N, 9]-->[N,1,9]
            color = (sh * enc_d).sum(-1)  # [N, 3]
            color = torch.sigmoid(color)
            self.feature_sigma_color = None
            self.color_l = color
            return sigma, color
        else:
            raise ValueError(f"not illegal model_type:{self.model_type}")

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        h[..., 0] = torch.clamp(h[..., 0].clone(), self.args.sigma_clip_min, self.args.sigma_clip_max)
        self.feature_sigma_color = h
        if self.training and self.args.global_step < self.args.stage_iters['stage1']:
            return None, None
        self.sigma_l = h[..., 0]
        sigma = trunc_exp(h[..., 0])  # sigma: [n]
        geo_feat = h[..., 1:]  # geo_feat: [n, 15]

        d = self.encoder_dir(d)  # d: [n, 16]
        h = torch.cat([d, geo_feat], dim=-1)  # h: [n, 15+16]
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        color = torch.sigmoid(h)
        self.color_l = color
        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        if self.model_type == "hash":
            x = self.encoder(x, bound=self.bound)  # out_x[N, 32=num_levels * fea_per_level]
        elif self.model_type == "mlp":
            x = self.forward_nerf_mlp(x)
        elif self.model_type == "vm":
            x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1
            sigma_feat = self.get_sigma_feat(x)
            sigma_feat = torch.clamp(sigma_feat, self.args.sigma_clip_min, self.args.sigma_clip_max)
            sigma = trunc_exp(sigma_feat)
            return {'sigma': sigma}
        elif self.model_type == "tensors":
            x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1  # x:[N, 3]
            x = self.compute_plenoxel_fea(x)
            h = x
            # h = torch.clamp(h, self.args.sigma_clip_min, self.args.sigma_clip_max)
            sigma = trunc_exp(torch.clamp(h[..., 0], self.args.sigma_clip_min, self.args.sigma_clip_max))
            sigma = trunc_exp(h[..., 0])
            return {'sigma': sigma}

        else:
            raise ValueError(f"not illegal model_type:{self.model_type}")

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        h = torch.clamp(h, self.args.sigma_clip_min, self.args.sigma_clip_max)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }

    def background(self, x, d):
        assert 1 == 2
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        assert 1==2
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], 3, dtype=x.dtype, device=x.device
            )  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # L1 penalty for loss
    def density_loss(self):
        loss = 0
        for i in range(len(self.sigma_mat)):
            loss = loss + torch.mean(torch.abs(self.sigma_mat[i])) + torch.mean(torch.abs(self.sigma_vec[i]))
        return loss
    
    # upsample utils
    @torch.no_grad()
    def upsample_params(self, mat, vec, resolution):

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]
            mat[i] = nn.Parameter(F.interpolate(mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=True))
            vec[i] = nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))


    @torch.no_grad()
    def upsample_model(self, resolution):
        self.upsample_params(self.sigma_mat, self.sigma_vec, resolution)
        self.upsample_params(self.color_mat, self.color_vec, resolution)
        self.resolution = resolution

    @torch.no_grad()
    def shrink_model(self):
        # shrink aabb_train and the model so it only represents the space inside aabb_train.

        half_grid_size = self.bound / self.grid_size
        thresh = min(self.density_thresh, self.mean_density)

        # get new aabb from the coarsest density grid (TODO: from the finest that covers current aabb?)
        valid_grid = self.density_grid[self.cascade - 1] > thresh # [N]
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid)) # [Nz] --> [Nz, 3], in [0, H - 1]
        #plot_pointcloud(valid_pos.detach().cpu().numpy()) # lots of noisy outliers in hashnerf...
        valid_pos = (2 * valid_pos / (self.grid_size - 1) - 1) * (self.bound - half_grid_size) # [Nz, 3], in [-b+hgs, b-hgs]
        min_pos = valid_pos.amin(0) - half_grid_size # [3]
        max_pos = valid_pos.amax(0) + half_grid_size # [3]

        # shrink model
        reso = torch.LongTensor(self.resolution).to(self.aabb_train.device)
        units = (self.aabb_train[3:] - self.aabb_train[:3]) / reso
        tl = (min_pos - self.aabb_train[:3]) / units
        br = (max_pos - self.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            mat_id_0, mat_id_1 = self.mat_ids[i]

            self.sigma_vec[i] = nn.Parameter(self.sigma_vec[i].data[..., tl[vec_id]:br[vec_id], :])
            self.color_vec[i] = nn.Parameter(self.color_vec[i].data[..., tl[vec_id]:br[vec_id], :])

            self.sigma_mat[i] = nn.Parameter(self.sigma_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
            self.color_mat[i] = nn.Parameter(self.color_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {self.aabb_train.cpu().numpy().tolist()}')

    # optimizer utils
    def get_params(self, lr, lr2=1e-3):
        if self.model_type == "hash":
            params = [
                {'params': self.encoder.parameters(), 'lr': lr},
                {'params': self.sigma_net.parameters(), 'lr': lr},
                {'params': self.encoder_dir.parameters(), 'lr': lr},
                {'params': self.color_net.parameters(), 'lr': lr},
            ]
        elif self.model_type == "mlp":
            params = [
                {"params": self.sigma_net.parameters(), "lr": lr},
                {"params": self.encoder_dir.parameters(), "lr": lr},
                {"params": self.color_net.parameters(), "lr": lr},
                {"params": self.nerf_mlp.parameters(), "lr": lr},
            ]
        elif self.model_type == "vm":
            params = [
                {'params': self.color_net.parameters(), 'lr': lr2},
                {'params': self.sigma_mat, 'lr': lr},
                {'params': self.sigma_vec, 'lr': lr},
                {'params': self.color_mat, 'lr': lr},
                {'params': self.color_vec, 'lr': lr},
                {'params': self.basis_mat.parameters(), 'lr': lr2},
            ]
        elif self.model_type == "tensors":
            params = [
                {"params": self.tensor_volume.parameters(), "lr": lr},
                {"params": self.encoder_dir.parameters(), "lr": lr},
            ]

        else:
            raise ValueError(f"not illegal model_type:{self.model_type}")

        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})

        return params
