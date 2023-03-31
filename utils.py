"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import *

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import OccupancyGrid, ray_marching, rendering


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        packed_info, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if t_starts.shape[0] > 0:
            rgb, opacity, depth = rendering(
                t_starts=t_starts,
                t_ends=t_ends,
                rgb_sigma_fn=rgb_sigma_fn,
                n_rays=chunk_rays.origins.shape[0],
                ray_indices=packed_info,
                render_bkgd=render_bkgd,
            )
            # print("SHAPES", rgb.shape, opacity.shape, depth.shape, len(t_starts))
            chunk_results = [rgb, opacity, depth, len(t_starts)]
        else:
            s = len(chunk_rays.origins)
            chunk_results = [
                torch.zeros(size=(s, 3), device="cuda:0") + 1.0,
                torch.zeros(size=(s, 1), device="cuda:0") + 1.0,
                torch.zeros(size=(s, 1), device="cuda:0") + 1.0,
                0,
            ]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )


def enforce_structure(
    radiance_field: torch.nn.Module, scene_aabb: torch.Tensor, num_samples: int, max_time_diff=0.01, device="cuda:0"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses the flow field to enforce structure by using the flowfield 
    to predict where points will move to and sample those points and 
    compare to what the TNerf says
    """
    sizes = torch.abs((scene_aabb[3:] - scene_aabb[:3]))
    x = torch.rand(size=(num_samples, 3), device=device) * sizes + scene_aabb[:3]
    dirs = torch.rand(size=(num_samples, 3), device=device) * 2.0 - 1.0
    mags = torch.sqrt(torch.sum(dirs ** 2, dim=1))
    dirs /= torch.stack((mags, mags, mags), dim=1)

    return radiance_field.enforce(x, dirs, t_diff=max_time_diff)