from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class System_Config:
    seed: int = 20220401
    basedir: str = "./log"
    ckpt: Optional[str] = None
    progress_refresh_rate: int = 10
    vis_every: int = 10000
    add_timestamp: bool = True


@dataclass
class Model_Config:
    model_name: str = "HexPlane_Slim"  # choose from "HexPlane", "HexPlane_Slim"
    N_voxel_init: int = 64 * 64 * 64  # initial voxel number
    N_voxel_final: int = 200 * 200 * 200  # final voxel number
    step_ratio: float = 0.5
    nonsquare_voxel: bool = True  # if yes, voxel numbers along each axis depend on scene length along each axis
    time_grid_init: int = 16
    time_grid_final: int = 128
    normalize_type: str = "normal"
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    update_emptymask_list: List[int] = field(
        default_factory=lambda: [4000, 8000, 10000]
    )

    # Plane Initialization
    density_n_comp: List[int] = field(default_factory=lambda: [24, 24, 24])
    app_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])
    density_dim: int = 1
    app_dim: int = 27
    DensityMode: str = "plain"  # choose from "plain", "general_MLP"
    AppMode: str = "general_MLP"
    init_scale: float = 0.1
    init_shift: float = 0.0

    # Fusion Methods
    fusion_one: str = "multiply"
    fusion_two: str = "concat"

    # Density Feature Settings
    fea2denseAct: str = "softplus"
    density_shift: float = -10.0
    distance_scale: float = 25.0

    # Density Regressor MLP settings
    density_t_pe: int = -1
    density_pos_pe: int = -1
    density_view_pe: int = -1
    density_fea_pe: int = 2
    density_featureC: int = 128
    density_n_layers: int = 3

    # Appearance Regressor MLP settings
    app_t_pe: int = -1
    app_pos_pe: int = -1
    app_view_pe: int = 2
    app_fea_pe: int = 2
    app_featureC: int = 128
    app_n_layers: int = 3

    # Empty mask settings
    emptyMask_thes: float = 0.001
    rayMarch_weight_thres: float = 0.0001

    # Reg
    random_background: bool = False
    depth_loss: bool = False
    depth_loss_weight: float = 1.0
    dist_loss: bool = False
    dist_loss_weight: float = 0.01

    TV_t_s_ratio: float = 2.0  # ratio of TV loss along temporal and spatial dimensions
    TV_weight_density: float = 0.0001
    TV_weight_app: float = 0.0001
    L1_weight_density: float = 0.0
    L1_weight_app: float = 0.0

    # Sampling
    align_corners: bool = True
    # There are two types of upsampling: aligned and unaligned.
    # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    # using "unaligned" upsampling will essentially double the grid sizes at each time, ignoring N_voxel_final.
    upsampling_type: str = "aligned"  # choose from "aligned", "unaligned".
    nSamples: int = 1000000


@dataclass
class Data_Config:
    datadir: str = "./data"
    dataset_name: str = "dnerf"  # choose from "dnerf", "neural3D_NDC"
    downsample: float = 1.0
    cal_fine_bbox: bool = False
    N_vis: int = -1
    time_scale: float = 1.0
    scene_bbox_min: List[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    scene_bbox_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    N_random_pose: int = 1000
    # for dnerf

    # for neural3D_NDC
    nv3d_ndc_bd_factor: float = 0.75
    nv3d_ndc_eval_step: int = 1
    nv3d_ndc_eval_index: int = 0
    nv3d_ndc_sphere_scale: float = 1.0

    # Hierachical Sampling for Neural3D_NDC
    stage_1_iteration: int = 300000
    stage_2_iteration: int = 250000
    stage_3_iteration: int = 100000
    key_f_num: int = 10
    stage_1_gamma: float = 0.001
    stage_2_gamma: float = 0.02
    stage_3_alpha: float = 0.1

    datasampler_type: str = "rays"  # choose from "rays", "images", "hierach"


@dataclass
class Optim_Config:
    # Learning Rate
    lr_density_grid: float = 0.02
    lr_app_grid: float = 0.02
    lr_density_nn: float = 0.001
    lr_app_nn: float = 0.001

    # Optimizer, Adam deault
    beta1: float = 0.9
    beta2: float = 0.99
    lr_decay_type: str = "exp"  # choose from "exp" or "cosine" or "linear"
    lr_decay_target_ratio: float = 0.1
    lr_decay_step: int = -1
    lr_upsample_reset: bool = True

    batch_size: int = 4096
    n_iters: int = 25000


@dataclass
class Config:
    config: Optional[str] = None
    expname: str = "default"

    render_only: bool = False
    render_train: bool = False
    render_test: bool = True
    render_path: bool = False

    systems: System_Config = System_Config()
    model: Model_Config = Model_Config()
    data: Data_Config = Data_Config()
    optim: Optim_Config = Optim_Config()
