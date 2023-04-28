"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import *

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc.estimators.occ_grid import OccGridEstimator as OccupancyGrid
from nerfacc import rendering

from mlp import ZD_NeRFRadianceField


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
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if t_starts.shape[0] > 0:
            rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
            # print("SHAPES", rgb.shape, opacity.shape, depth.shape, len(t_starts))
            chunk_results = [rgb, opacity, depth, len(t_starts)]
        else:
            print(0/0)
            s = len(chunk_rays.origins)
            chunk_results = [
                torch.zeros(size=(s, 3), device="cuda:0") + 1.0,
                torch.zeros(size=(s, 1), device="cuda:0") + 0.0,
                torch.zeros(size=(s, 1), device="cuda:0") + 0.0,
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
    radiance_field: ZD_NeRFRadianceField,
    scene_aabb: torch.Tensor,
    n_samples: int,
    max_time_diff=0.01,
    device="cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses the flow field to enforce structure by using the flowfield
    to predict where points will move to and sample those points and
    compare to what the TNerf says
    """
    aabb_size = torch.abs((scene_aabb[3:] - scene_aabb[:3]))
    sample_points = (
        torch.rand(size=(n_samples, 3), device=device) * aabb_size + scene_aabb[:3]
    )
    return radiance_field.flow_field_pred(
        sample_points, t_diff=max_time_diff
    )


def sample_specular(
    radiance_field: ZD_NeRFRadianceField,
    scene_aabb: torch.Tensor,
    rays_d: torch.Tensor,
    n_samples: int,
    device="cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample specular nerf so that its magnitude can be reduced
    """
    aabb_size = torch.abs((scene_aabb[3:] - scene_aabb[:3]))
    sample_points = (
        torch.rand(size=(n_samples, 3), device=device) * aabb_size + scene_aabb[:3]
    )
    # Get random view dirs to use when rendering the points
    idx = torch.randint(0, len(rays_d), (n_samples,), device=device)

    sample_times = torch.rand(size=(n_samples, 1), device=device)
    
    return radiance_field.sample_spec(sample_points, sample_times, rays_d[idx])

def flow_loss_func(target: torch.Tensor, pred:torch.Tensor, alpha=10)-> torch.Tensor:
    return ((alpha*(target - pred))**2).mean()


"""
Small Expirement, not actually used
def keypoints_loss(
    radiance_field: ZD_NeRFRadianceField,
    points: torch.Tensor,
    rays_d: torch.Tensor,
    n_samples=512,
    alpha=0.01,
):
    n_times = int(points.shape[0])
    n_points = int(points.shape[1])
    # If alpha is greater than 0, you can sample the same set of keypoints more than once
    if alpha > 0.0:
        id_t = torch.randint(0, n_times - 1, (n_samples,), device=points.device)
        id_p = torch.randint(0, n_points, (n_samples,), device=points.device)
    else:
        raise NotImplemented("Need to implement for when alpha is 0")
    
    reverse = random.random() > 0.5
    # Samples from the data set at time t
    samples_t_0 = points[id_t, id_p]
    noise = (torch.rand_like(samples_t_0) - 0.5) * alpha
    samples_t_0 += noise

    # Samples at t plus one 
    id_next_t = id_t + (-1 if reverse else 1)
    samples_t_1 = points[id_next_t, id_p] + noise

    rays_ids = torch.randint(0, len(rays_d), (n_samples,))
    dirs = rays_d.reshape(-1, 3)[rays_ids]
    return radiance_field(samples_t_0, id_t.unsqueeze(1), dirs)[0],radiance_field(samples_t_1, id_next_t.unsqueeze(1), dirs)[0]
"""