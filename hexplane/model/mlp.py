from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Union

import torch


def positional_encoding(positions, freqs):
    """
    Return positional_encoding results with frequency freqs.
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class General_MLP(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): t_pe, feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe.

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
        self,
        inChanel: int,
        outChanel: int,
        t_pe: int = 6,
        fea_pe: int = 6,
        pos_pe: int = 6,
        view_pe: int = 6,
        featureC: int = 128,
        n_layers: int = 3,
        use_sigmoid: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()

        self.in_mlpC = inChanel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid

        # Whether use these features as inputs
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        if self.use_fea:
            self.in_mlpC += 2 * fea_pe * inChanel
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3

        assert n_layers >= 2  # Assert at least two layers of MLP
        layers = [torch.nn.Linear(self.in_mlpC, featureC), torch.nn.ReLU(inplace=True)]

        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(featureC, outChanel)]
        self.mlp = torch.nn.Sequential(*layers)

        if zero_init:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        features: torch.Tensor,
        frame_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward.
        """
        # Collect input data
        indata = [features]
        if self.use_t:
            indata += [frame_time]
            if self.t_pe > 0:
                indata += [positional_encoding(frame_time, self.t_pe)]
        if self.use_fea:
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        if self.use_pos:
            indata += [pts]
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view:
            indata += [viewdirs]
            if self.view_pe > 0:
                indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)

        rgb = self.mlp(mlp_in)
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        return rgb
