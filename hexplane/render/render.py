import os
import wandb
import imageio
import numpy as np
import torch
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm
import wandb

from hexplane.render.util.metric import rgb_lpips, rgb_ssim
from hexplane.render.util.util import visualize_depth_numpy


def OctreeRender_trilinear_fast(
    rays,
    time,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    device="cuda",
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals = [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map, alpha_map, z_val_map = model(
            rays_chunk,
            time_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
        )
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        None,
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps = [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    for idx in tqdm(idxs):
        data = test_dataset[idx]
        samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
        depth = None

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])
        rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
            rays,
            times,
            model,
            chunk=4096,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depth" in data.keys():
            depth = data["depth"]
            gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if len(test_dataset):
            gt_rgb = gt_rgb.view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)
                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        if depth is not None:
            gt_depth_maps.append(gt_depth)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_gt.png", gt_rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

            #log all images to wandb
            wandb.log({ f"{prefix}{idx:03d}.png": wandb.Image(rgb_map) })
            wandb.log({ f"{prefix}{idx:03d}_gt.png": wandb.Image(gt_rgb_map) })
            wandb.log({ f"rgbd/{prefix}{idx:03d}.png": wandb.Image(rgb_map) })

            if depth is not None:
                rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}_gt.png", rgb_map)
                wandb.log({ f"rgbd/{prefix}{idx:03d}_gt.png": wandb.Image(rgb_map) })

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        format="FFMPEG",
        quality=10,
    )

    wandb.log({ f"{prefix}video.mp4": wandb.Video(f"{savePath}/{prefix}video.mp4", fps=30, format="mp4") })
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps),
        format="FFMPEG",
        fps=30,
        quality=10,
    )

    wandb.log({ f"{prefix}depthvideo.mp4": wandb.Video(f"{savePath}/{prefix}depthvideo.mp4", fps=30, format="mp4") })

    if depth is not None:
        imageio.mimwrite(
            f"{savePath}/{prefix}_gt_depthvideo.mp4",
            np.stack(gt_depth_maps),
            format="FFMPEG",
            fps=30,
            quality=10,
        )

        wandb.log({ f"{prefix}_gt_depthvideo.mp4": wandb.Video(f"{savePath}/{prefix}_gt_depthvideo.mp4", fps=30, format="mp4") })

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )

                # log to wandb
                wandb.log({
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "MS-SSIM": msssim,
                    "LPIPS_a": l_a,
                    "LPIPS_v": l_v
                })

                for i in range(len(PSNRs)):
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n"
                    )
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs


@torch.no_grad()
def evaluation_path(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times = test_dataset.get_val_rays()

    for idx in tqdm(range(val_times.shape[0])):
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        time = time.expand(rays.shape[0], 1)
        rgb_map, _, depth_map, _, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            wandb.log({ f"{prefix}{idx:03d}.png": wandb.Image(rgb_map) })
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
            wandb.log({ f"rgbd/{prefix}{idx:03d}.png": wandb.Image(rgb_map) })

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
    )
    wandb.log({ f"{prefix}video.mp4": wandb.Video(f"{savePath}/{prefix}video.mp4", fps=30, format="mp4") })
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )
    wandb.log({ f"{prefix}depthvideo.mp4": wandb.Video(f"{savePath}/{prefix}depthvideo.mp4", fps=30, format="mp4") })
    
    return 0
