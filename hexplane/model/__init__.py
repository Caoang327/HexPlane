import torch

from hexplane.model.HexPlane import HexPlane
from hexplane.model.HexPlane_Slim import HexPlane_Slim
from hexplane.render.util.util import N_to_reso


def init_model(cfg, aabb, near_far, device):
    reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)

    if cfg.systems.ckpt is not None:
        model = torch.load(cfg.systems.ckpt, map_location=device)
    else:
        # There are two types of upsampling: aligned and unaligned.
        # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
        # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
        if cfg.model.upsampling_type == "aligned":
            reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
        model = eval(cfg.model.model_name)(
            aabb, reso_cur, device, cfg.model.time_grid_init, near_far, **cfg.model
        )
    return model, reso_cur
