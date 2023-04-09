import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight_dim1=1.0, TVLoss_weight_dim2=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight_dim1 = TVLoss_weight_dim1
        self.TVLoss_weight_dim2 = TVLoss_weight_dim2

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = (
            torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
            * self.TVLoss_weight_dim1
        )
        w_tv = (
            torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
            * self.TVLoss_weight_dim2
        )
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


@torch.jit.script
def compute_dist_loss(weights, svals):
    """Compute the distortion loss of each ray.
    Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields.
        Barron et al., CVPR 2022.
        https://arxiv.org/abs/2111.12077
    As per Equation (15) in the paper. Note that we slightly modify the loss to
    account for "sampling at infinity" when rendering NeRF.
    Args:
        pred_weights (jnp.ndarray): (..., S) predicted weights of each
            sample along the ray.
        svals (jnp.ndarray): (..., S + 1) normalized marching step of each
            sample along the ray.
    """

    smids = 0.5 * (svals[..., 1:] + svals[..., :-1])
    sdeltas = svals[..., 1:] - svals[..., :-1]

    loss_uni = (1 / 3) * (sdeltas * weights.pow(2)).sum(dim=-1).mean()
    wm = weights * smids
    w_cumsum = weights.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = weights[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni
