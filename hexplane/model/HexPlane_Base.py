import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
from torch.nn import functional as F

from hexplane.model.mlp import General_MLP
from hexplane.model.sh import eval_sh_bases


def raw2alpha(sigma: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    rgb = features
    return rgb


def DensityRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    density = features
    return density


class EmptyGridMask(torch.nn.Module):
    def __init__(
        self, device: torch.device, aabb: torch.Tensor, empty_volume: torch.Tensor
    ):
        super().__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.empty_volume = empty_volume.view(1, 1, *empty_volume.shape[-3:])
        self.gridSize = torch.LongTensor(
            [empty_volume.shape[-1], empty_volume.shape[-2], empty_volume.shape[-3]]
        ).to(self.device)

    def sample_empty(self, xyz_sampled):
        empty_vals = F.grid_sample(
            self.empty_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)
        return empty_vals


class HexPlane_Base(torch.nn.Module):
    """
    HexPlane Base Class.
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        gridSize: List[int],
        device: torch.device,
        time_grid: int,
        near_far: List[float],
        density_n_comp: Union[int, List[int]] = 8,
        app_n_comp: Union[int, List[int]] = 24,
        density_dim: int = 1,
        app_dim: int = 27,
        DensityMode: str = "plain",
        AppMode: str = "general_MLP",
        emptyMask: Optional[EmptyGridMask] = None,
        fusion_one: str = "multiply",
        fusion_two: str = "concat",
        fea2denseAct: str = "softplus",
        init_scale: float = 0.1,
        init_shift: float = 0.0,
        normalize_type: str = "normal",
        **kwargs,
    ):
        super().__init__()

        self.aabb = aabb
        self.device = device
        self.time_grid = time_grid
        self.near_far = near_far
        self.near_far_org = near_far
        self.step_ratio = kwargs.get("step_ratio", 2.0)
        self.update_stepSize(gridSize)

        # Density and Appearance HexPlane components numbers and value regression mode.
        self.density_n_comp = density_n_comp
        self.app_n_comp = app_n_comp
        self.density_dim = density_dim
        self.app_dim = app_dim
        self.align_corners = kwargs.get(
            "align_corners", True
        )  # align_corners for grid_sample

        # HexPlane weights initialization: scale and shift for uniform distribution.
        self.init_scale = init_scale
        self.init_shift = init_shift

        # HexPlane fusion mode.
        self.fusion_one = fusion_one
        self.fusion_two = fusion_two

        # Plane Index
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

        # Coordinate normalization type.
        self.normalize_type = normalize_type

        # Plane initialization.
        self.init_planes(gridSize[0], device)

        # Density calculation settings.
        self.fea2denseAct = fea2denseAct  # feature to density activation function
        self.density_shift = kwargs.get(
            "density_shift", -10.0
        )  # density shift for density activation function.
        self.distance_scale = kwargs.get(
            "distance_scale", 25.0
        )  # distance scale for density activation function.
        self.DensityMode = DensityMode
        self.density_t_pe = kwargs.get("density_t_pe", -1)
        self.density_pos_pe = kwargs.get("density_pos_pe", -1)
        self.density_view_pe = kwargs.get("density_view_pe", -1)
        self.density_fea_pe = kwargs.get("density_fea_pe", 6)
        self.density_featureC = kwargs.get("density_featureC", 128)
        self.density_n_layers = kwargs.get("density_n_layers", 3)
        self.init_density_func(
            self.DensityMode,
            self.density_t_pe,
            self.density_pos_pe,
            self.density_view_pe,
            self.density_fea_pe,
            self.density_featureC,
            self.density_n_layers,
            self.device,
        )

        # Appearance calculation settings.
        self.AppMode = AppMode
        self.app_t_pe = kwargs.get("app_t_pe", -1)
        self.app_pos_pe = kwargs.get("app_pos_pe", -1)
        self.app_view_pe = kwargs.get("app_view_pe", 6)
        self.app_fea_pe = kwargs.get("app_fea_pe", 6)
        self.app_featureC = kwargs.get("app_featureC", 128)
        self.app_n_layers = kwargs.get("app_n_layers", 3)
        self.init_app_func(
            AppMode,
            self.app_t_pe,
            self.app_pos_pe,
            self.app_view_pe,
            self.app_fea_pe,
            self.app_featureC,
            self.app_n_layers,
            device,
        )

        # Density HexPlane mask and other acceleration tricks.
        self.emptyMask = emptyMask
        self.emptyMask_thres = kwargs.get(
            "emptyMask_thres", 0.001
        )  # density threshold for emptiness mask
        self.rayMarch_weight_thres = kwargs.get(
            "rayMarch_weight_thres", 0.0001
        )  # density threshold for rendering colors.

        # Regulartization settings.
        self.random_background = kwargs.get("random_background", False)
        self.depth_loss = kwargs.get("depth_loss", False)

    def init_density_func(
        self, DensityMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device
    ):
        """
        Initialize density regression function.
        """
        if (
            DensityMode == "plain"
        ):  # Use extracted features directly from HexPlane as density.
            assert self.density_dim == 1  # Assert the extracted features are scalers.
            self.density_regressor = DensityRender
        elif DensityMode == "general_MLP":  # Use general MLP to regress density.
            assert (
                view_pe < 0
            )  # Assert no view position encoding. Density should not depend on view direction.
            self.density_regressor = General_MLP(
                self.density_dim,
                1,
                t_pe,
                fea_pe,
                pos_pe,
                view_pe,
                featureC,
                n_layers,
                use_sigmoid=False,
                zero_init=False,
            ).to(device)
        else:
            raise NotImplementedError("No such Density Regression Mode")
        print("DENSITY REGRESSOR:")
        print(self.density_regressor)

    def init_app_func(
        self, AppMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device
    ):
        """
        Initialize appearance regression function.
        """
        if AppMode == "SH":  # Use Spherical Harmonics SH to render appearance.
            self.app_regressor = SHRender
        elif AppMode == "RGB":  # Use RGB to render appearance.
            assert self.app_dim == 3
            self.app_regressor = RGBRender
        elif AppMode == "general_MLP":  # Use general MLP to render appearance.
            self.app_regressor = General_MLP(
                self.app_dim,
                3,
                t_pe,
                fea_pe,
                pos_pe,
                view_pe,
                featureC,
                n_layers,
                use_sigmoid=True,
                zero_init=True,
            ).to(device)
        else:
            raise NotImplementedError("No such App Regression Mode")
        print("APP REGRESSOR:")
        print(self.app_regressor)
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_planes(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled, frame_time):
        pass

    def compute_appfeature(self, xyz_sampled, frame_time):
        pass

    def normalize_coord(self, xyz_sampled):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """
        if self.normalize_type == "normal":
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def feature2density(self, density_features: torch.Tensor) -> torch.Tensor:
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        else:
            raise NotImplementedError("No such activation function for density feature")

    def sample_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        is_train: bool = True,
        N_samples: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points along rays based on the given ray origin and direction.

        Args:
            rays_o: (B, 3) tensor, ray origin.
            rays_d: (B, 3) tensor, ray direction.
            is_train: bool, whether in training mode.
            N_samples: int, number of samples along each ray.

        Returns:
            rays_pts: (B, N_samples, 3) tensor, sampled points along each ray.
            interpx: (B, N_samples) tensor, sampled points' distance to ray origin.
            ~mask_outbbox: (B, N_samples) tensor, mask for points within bounding box.
        """
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def filtering_rays(
        self,
        all_rays: torch.Tensor,
        all_rgbs: torch.Tensor,
        all_times: torch.Tensor,
        all_depths: Optional[torch.Tensor] = None,
        N_samples: int = 256,
        chunk: int = 10240 * 5,
        bbox_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter out rays that are not within the bounding box.

        Args:
            all_rays: (N_rays, N_samples, 6) tensor, rays [rays_o, rays_d].
            all_rgbs: (N_rays, N_samples, 3) tensor, rgb values.
            all_times: (N_rays, N_samples) tensor, time values.
            all_depths: (N_rays, N_samples) tensor, depth values.
            N_samples: int, number of samples along each ray.

        Returns:
            all_rays: (N_rays, N_samples, 6) tensor, filtered rays [rays_o, rays_d].
            all_rgbs: (N_rays, N_samples, 3) tensor, filtered rgb values.
            all_times: (N_rays, N_samples) tensor, filtered time values.
            all_depths: Optional, (N_rays, N_samples) tensor, filtered depth values.
        """
        print("========> filtering rays ...")
        tt = time.time()
        N = torch.tensor(all_rays.shape[:-1]).prod()
        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            # Filter based on bounding box.
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            # Filter based on emptiness mask.
            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, N_samples=N_samples, is_train=False
                )
                xyz_sampled = self.normalize_coord(xyz_sampled)
                mask_inbbox = (
                    self.emptyMask.sample_empty(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        if all_depths is not None:
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_times[mask_filtered],
                all_depths[mask_filtered],
            )
        else:
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_times[mask_filtered],
                None,
            )

    def forward(
        self,
        rays_chunk: torch.Tensor,
        frame_time: torch.Tensor,
        white_bg: bool = True,
        is_train: bool = False,
        ndc_ray: bool = False,
        N_samples: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HexPlane.

        Args:
            rays_chunk: (B, 6) tensor, rays [rays_o, rays_d].
            frame_time: (B, 1) tensor, time values.
            white_bg: bool, whether to use white background.
            is_train: bool, whether in training mode.
            ndc_ray: bool, whether to use normalized device coordinates.
            N_samples: int, number of samples along each ray.

        Returns:
            rgb: (B, 3) tensor, rgb values.
            depth: (B, 1) tensor, depth values.
            alpha: (B, 1) tensor, accumulated weights.
            z_vals: (B, N_samples) tensor, z values.
        """
        # Prepare rays.
        viewdirs = rays_chunk[:, 3:6]
        xyz_sampled, z_vals, ray_valid = self.sample_rays(
            rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
        )
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1
        )
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        if ndc_ray:
            dists = dists * rays_norm
        viewdirs = viewdirs / rays_norm

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        frame_time = frame_time.view(-1, 1, 1).expand(
            xyz_sampled.shape[0], xyz_sampled.shape[1], 1
        )

        # Normalize coordinates.
        xyz_sampled = self.normalize_coord(xyz_sampled)

        # If emptiness mask is availabe, we first filter out rays with low opacities.
        if self.emptyMask is not None:
            emptiness = self.emptyMask.sample_empty(xyz_sampled[ray_valid])
            empty_mask = emptiness > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~empty_mask
            ray_valid = ~ray_invalid

        # Initialize sigma and rgb values.
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        # Compute density feature and density if there are valid rays.
        if ray_valid.any():
            density_feature = self.compute_densityfeature(
                xyz_sampled[ray_valid], frame_time[ray_valid]
            )
            density_feature = self.density_regressor(
                xyz_sampled[ray_valid],
                viewdirs[ray_valid],
                density_feature,
                frame_time[ray_valid],
            )
            validsigma = self.feature2density(density_feature)
            sigma[ray_valid] = validsigma.view(-1)
        alpha, weight, bg_weight = raw2alpha(
            sigma, dists * self.distance_scale
        )  # alpha is the opacity, weight is the accumulated weight. bg_weight is the accumulated weight for last sampling point.

        # Compute appearance feature and rgb if there are valid rays (whose weight are above a threshold).
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            app_features = self.compute_appfeature(
                xyz_sampled[app_mask], frame_time[app_mask]
            )
            valid_rgbs = self.app_regressor(
                xyz_sampled[app_mask],
                viewdirs[app_mask],
                app_features,
                frame_time[app_mask],
            )
            rgb[app_mask] = valid_rgbs
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        # If white_bg or (is_train and torch.rand((1,))<0.5):
        if white_bg or not is_train:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        else:
            rgb_map = rgb_map + (1.0 - acc_map[..., None]) * torch.rand(
                size=(1, 3), device=rgb_map.device
            )
        rgb_map = rgb_map.clamp(0, 1)

        # Calculate depth.
        if self.depth_loss:
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        else:
            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)
                depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        return rgb_map, depth_map, alpha, z_vals

    @torch.no_grad()
    def updateEmptyMask(self, gridSize=(200, 200, 200), time_grid=64):
        """
        Like TensoRF, we compute the emptiness voxel to store the opacities of the scene and skip computing the opacities of rays with low opacities.
        For HexPlane, the emptiness voxel is the union of the density volumes of all the frames.

        This is the same idea as AlphaMask in TensoRF, while we rename it for better understanding.

        Note that we always assume the voxel is a cube [-1, 1], and we sample for normalized coordinate.
        TODO: add voxel shrink function and inverse normalization functions.
        """
        emptiness, dense_xyz = self.getDenseEmpty(gridSize, time_grid)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        emptiness = emptiness.clamp(0, 1).transpose(0, 2).contiguous()[None, None]

        ks = 3
        emptiness = F.max_pool3d(
            emptiness, kernel_size=ks, padding=ks // 2, stride=1
        ).view(gridSize[::-1])
        emptiness[emptiness >= self.emptyMask_thres] = 1
        emptiness[emptiness < self.emptyMask_thres] = 0

        self.emptyMask = EmptyGridMask(self.device, self.aabb, emptiness)

        return None

    @torch.no_grad()
    def getDenseEmpty(self, gridSize=None, time_grid=None):
        """
        For a 4D volume, we sample the opacity values of discrete spacetime points and store them in a 3D volume.
        Note that we always assume the 4D volume is in the range of [-1, 1] for each axis.
        """
        gridSize = self.gridSize if gridSize is None else gridSize
        time_grid = self.time_grid if time_grid is None else time_grid

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = samples * 2.0 - 1.0
        emptiness = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            emptiness[i] = self.compute_emptiness(
                dense_xyz[i].view(-1, 3).contiguous(), time_grid, self.stepSize
            ).view((gridSize[1], gridSize[2]))
        return emptiness, dense_xyz

    def compute_emptiness(self, xyz_locs, time_grid=64, length=1):
        """
        Compute the emptiness of spacetime points. Emptiness is the density.
        For each sapce point, we calcualte its densitis for various time steps and calculate its maximum density.
        """
        if self.emptyMask is not None:
            emptiness = self.emptyMask.sample_empty(xyz_locs)
            empty_mask = emptiness > 0
        else:
            empty_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if empty_mask.any():
            xyz_sampled = xyz_locs[empty_mask]
            time_samples = torch.linspace(-1, 1, time_grid).to(xyz_sampled.device)
            N, T = xyz_sampled.shape[0], time_samples.shape[0]
            xyz_sampled = (
                xyz_sampled.unsqueeze(1).expand(-1, T, -1).contiguous().view(-1, 3)
            )
            time_samples = (
                time_samples.unsqueeze(0).expand(N, -1).contiguous().view(-1, 1)
            )

            density_feature = self.compute_densityfeature(xyz_sampled, time_samples)
            sigma_feature = self.density_regressor(
                xyz_sampled,
                xyz_sampled,
                density_feature,
                time_samples,
            ).view(N, T)

            sigma_feature = torch.amax(sigma_feature, -1)
            validsigma = self.feature2density(sigma_feature)
            sigma[empty_mask] = validsigma

        emptiness = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return emptiness
