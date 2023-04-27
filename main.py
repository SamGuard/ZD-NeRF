"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time
import random

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.dnerf_synthetic import SubjectLoader
from mlp import ZD_NeRFRadianceField
from utils import (
    render_image,
    set_random_seed,
    enforce_structure,
    sample_specular,
    flow_loss_func,
)
from flow_trainer import train_flow_field

from nerfacc import ContractionType, OccupancyGrid

# For testing
from torch.profiler import profile, record_function, ProfilerActivity


def new_model():
    radiance_field = ZD_NeRFRadianceField().to(device)
    optim = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=[
            max_steps // 2,
            max_steps * 3 // 4,
            max_steps * 5 // 6,
            max_steps * 9 // 10,
        ],
        gamma=0.33,
    )
    flow_opt = torch.optim.Adam(radiance_field.parameters(), lr=1e-5)
    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    return radiance_field, optim, scheduler, occupancy_grid


if __name__ == "__main__":
    device = "cuda:0"
    # set_random_seed(27)
    set_random_seed(int(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # dnerf
            "bouncingballs",
            "hellwarrior",
            "hook",
            "jumpingjacks",
            "lego",
            "mutant",
            "standup",
            "trex",
            "basic_sphere",
            "basic_sphere_2",
            "world_deform",
            "world_deform_v2",
            "brick",
            "brick_v2",
            "bouncy",
            "balls",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-5,-5,-5,5,5,5",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--just_render",
        default="false",
        help="Bool, whether to train or not, just render",
        type=lambda x: x.lower() == "true",
    )
    parser.add_argument(
        "--num_renders",
        default=11,
        help="Int, number of images to render. Only has effect when just_render is True",
        type=int,
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--train_in_order", type=bool, default=False)
    parser.add_argument("--ray_batch_size", type=int, default=1 << 16)
    parser.add_argument("--flow_step", type=int, default=10000)
    args = parser.parse_args()

    render_n_samples = args.samples
    train_in_order = args.train_in_order

    # setup the dataset
    data_root_fp = "/home/ruilongli/data/dnerf/"
    target_sample_batch_size = args.ray_batch_size
    grid_resolution = 128

    # create output folders
    try:
        os.mkdir("network_out")
    except:
        pass

    try:
        os.stat("/mnt/io/")
        RENDER_PATH = "/mnt/io/render_out"
    except:
        RENDER_PATH = "./render_out"

    try:
        os.mkdir(RENDER_PATH)
    except:
        pass

    # setup the scene bounding box.
    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
    ).item()

    # setup the radiance field we want to train.
    max_steps = args.max_steps
    grad_scaler = torch.cuda.amp.GradScaler(1)
    radiance_field, optimizer, scheduler, occupancy_grid = new_model()

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        batch_over_images=False,
    )
    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)
    train_dataset.timestamps = train_dataset.timestamps.to(device)
    has_keypoints = train_dataset.has_points
    if has_keypoints:
        print("Keypoints found")
        train_dataset.points_time = train_dataset.points_time.to(device)
        train_dataset.points_data = train_dataset.points_data.to(device)
    else:
        print("Keypoints NOT found, continuing")

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)
    test_dataset.timestamps = test_dataset.timestamps.to(device)

    # training
    step = 0
    attempts = 0
    tic = time.time()
    flow_field_start_step = args.flow_step
    flow_field_n_steps = 1
    num_data = len(train_dataset)
    if not args.just_render:
        for epoch in range(10000000):
            for i in range(len(train_dataset)):
                radiance_field.train()

                data = train_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                timestamps = (
                    torch.zeros(size=(pixels.shape[0], 1), device="cuda:0")
                    + data["timestamps"]
                )

                if step == flow_field_start_step and has_keypoints:
                    # Decreased epochs for testing, revert once done
                    train_flow_field(
                        radiance_field.warp.odefunc,
                        train_dataset.points_time,
                        train_dataset.points_data,
                        epochs=1000,
                        steps_ahead=5,
                    )

                # update occupancy grid
                occupancy_grid.every_n_step(
                    step=step,
                    occ_eval_fn=lambda x: radiance_field.query_opacity(
                        x, timestamps, render_step_size
                    ),
                )

                # render
                rgb, acc, depth, n_rendering_samples = render_image(
                    radiance_field,
                    occupancy_grid,
                    rays,
                    scene_aabb,
                    # rendering options
                    near_plane=near_plane,
                    far_plane=far_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=args.cone_angle,
                    alpha_thre=0.01 if step > 1000 else 0.00,
                    # dnerf options
                    timestamps=timestamps,
                )
                if (
                    step >= flow_field_start_step
                    and 0 == (step - flow_field_start_step) % flow_field_n_steps
                ):
                    (
                        start_keypoints_rgb,
                        end_keypoints_rgb,
                        start_keypoints_dense,
                        end_keypoints_dense,
                    ) = enforce_structure(
                        radiance_field=radiance_field,
                        scene_aabb=scene_aabb,
                        n_samples=2**15,
                        max_time_diff=0.25,
                    )

                    spec_samples = sample_specular(
                        radiance_field=radiance_field,
                        scene_aabb=scene_aabb,
                        rays_d=rays.viewdirs,
                        n_samples=2**14,
                    )

                    loss_nerf_flow = flow_loss_func(
                        start_keypoints_rgb, end_keypoints_rgb, 1
                    ) + flow_loss_func(start_keypoints_dense, end_keypoints_dense, 1)

                    n_flow_samples = len(start_keypoints_rgb) + len(
                        start_keypoints_dense
                    )
                    loss_spec = F.mse_loss(spec_samples, torch.zeros_like(spec_samples))
                else:
                    loss_nerf_flow = 0
                    loss_spec = 0
                    n_flow_samples = 0

                if n_rendering_samples == 0:
                    continue

                alive_ray_mask = acc.squeeze(-1) > 0
                n_alive_rays = alive_ray_mask.long().sum()
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = int(
                    train_dataset.num_rays
                    * (target_sample_batch_size / float(n_rendering_samples))
                )

                # TEMPORARY FIX, CHANGE min/max rays TO arg
                num_rays = min(8192, num_rays)
                if step < 100:
                    num_rays = max(num_rays, 2048)
                    num_rays = min(4096, num_rays)
                train_dataset.update_num_rays(num_rays)

                if n_alive_rays == 0:
                    if attempts < 10000:
                        del radiance_field
                        del optimizer
                        del scheduler
                        del occupancy_grid
                        train_dataset.update_num_rays(
                            target_sample_batch_size // render_n_samples
                        )
                        set_random_seed(int(time.time()))
                        (
                            radiance_field,
                            optimizer,
                            scheduler,
                            occupancy_grid,
                        ) = new_model()
                        attempts += 1
                        step = 0
                        print(
                            "Model to failed to not keep enough rays alive, reseting. Attempt number:",
                            attempts,
                        )
                        continue
                    else:
                        print("No rays hit target, exiting")
                        exit(-1)

                if n_alive_rays > 0:
                    # compute loss
                    loss_nerf = F.smooth_l1_loss(
                        rgb[alive_ray_mask], pixels[alive_ray_mask], beta=0.05
                    )

                    loss = loss_nerf + loss_nerf_flow + loss_spec
                    optimizer.zero_grad()
                    # do not unscale it because we are using Adam.
                    grad_scaler.scale(loss).backward()
                    optimizer.step()
                    scheduler.step()

                if step % 1 == 0 and n_alive_rays > 0:
                    elapsed_time = time.time() - tic
                    loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                    print(
                        f"time={elapsed_time:.2f}s | step={step} | "
                        f"loss_nerf={loss_nerf:.5f} | ",
                        f"loss_flow={loss_nerf_flow:.5f} |",
                        f"loss_spec={loss_spec:.5f} |"
                        f"alive={alive_ray_mask.long().sum():d} | "
                        f"n_samples={n_rendering_samples:d} |",
                        f"n_flow={n_flow_samples} |",
                    )

                if step % 5000 == 0:
                    torch.save(
                        radiance_field.state_dict(),
                        os.path.join(
                            "/",
                            "mnt",
                            "io",
                            "train_out",  # "train_out",
                            "zdnerf_nerf_step" + str(step) + ".pt",
                        ),
                    )

                if step >= 0 and step % max_steps == 0 and step > 0:
                    # evaluation
                    radiance_field.eval()

                    psnrs = []
                    with torch.no_grad():
                        for i in tqdm.tqdm(range(len(test_dataset))):
                            data = test_dataset[i]
                            render_bkgd = data["color_bkgd"]
                            rays = data["rays"]
                            pixels = data["pixels"]
                            timestamps = data["timestamps"]

                            # rendering
                            rgb, acc, depth, _ = render_image(
                                radiance_field,
                                occupancy_grid,
                                rays,
                                scene_aabb,
                                # rendering options
                                near_plane=None,
                                far_plane=None,
                                render_step_size=render_step_size,
                                render_bkgd=render_bkgd,
                                cone_angle=args.cone_angle,
                                alpha_thre=0.01,
                                # test options
                                test_chunk_size=args.test_chunk_size,
                                # dnerf options
                                timestamps=timestamps,
                            )
                            mse = F.mse_loss(rgb, pixels)
                            psnr = -10.0 * torch.log(mse) / np.log(10.0)
                            psnrs.append(psnr.item())
                            # imageio.imwrite(
                            #     "acc_binary_test.png",
                            #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                            # )
                            # imageio.imwrite(
                            #     "rgb_test.png",
                            #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                            # )
                            # break
                    psnr_avg = sum(psnrs) / len(psnrs)
                    print(f"evaluation: psnr_avg={psnr_avg}")
                    train_dataset.training = True

                if step == max_steps:
                    print("training stops")
                    exit()

                step += 1
    else:
        radiance_field = ZD_NeRFRadianceField().to(device)
        radiance_field.load_state_dict(
            torch.load(os.path.join("/", "mnt", "io", "train_out", args.model), device)
        )
        train_dataset.training = False

        radiance_field.eval()
        step = 0
        num_time = args.num_renders
        timestamps = torch.tensor([[0.0]], dtype=torch.float32).to(device)
        with torch.no_grad():
            for i in range(10):
                occupancy_grid._update(
                    step=step,
                    occ_eval_fn=lambda x: radiance_field.query_opacity(
                        x, timestamps, render_step_size
                    ),
                )

            for t in map(lambda x: x / (num_time - 1), range(num_time)):
                for i in [
                    0
                ]:  # range(len(test_dataset)): #[int(t * num_time) % len(test_dataset)]:#
                    # data = test_dataset[i]
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]
                    timestamps[0][0] = t

                    occupancy_grid._update(
                        step=step,
                        occ_eval_fn=lambda x: radiance_field.query_opacity(
                            x, timestamps, render_step_size
                        ),
                    )
                    """
                    occupancy_grid.every_n_step(
                        step=step,
                        occ_eval_fn=lambda x: radiance_field.query_opacity(
                            x, timestamps, render_step_size
                        ),
                    )
                    """

                    # rendering
                    rgb, acc, depth, _ = render_image(
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb,
                        # rendering options
                        near_plane=None,
                        far_plane=None,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=args.cone_angle,
                        alpha_thre=0.01,
                        # test options
                        test_chunk_size=args.test_chunk_size,
                        # dnerf options
                        timestamps=timestamps,
                    )

                    imageio.imwrite(
                        os.path.join(
                            RENDER_PATH, "rgb_time_{:.3f}_img_{}.png".format(t, i)
                        ),
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    print(f"Image at time={t}, render={i}")

                    step += 1

        print("All done!")
