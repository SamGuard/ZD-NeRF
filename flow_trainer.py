import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def train_flow_field(warp: nn.Module, timestamps: torch.Tensor, points: torch.Tensor, epochs=50):
    points.requires_grad_()
    timestamps.requires_grad_()

    optimizer = torch.optim.Adam(warp.parameters(), 0.01)

    step = 0
    p_target = points[0].clone()
    while step < epochs:
        for t, p in zip(timestamps, points):
            if t > 0:
                p0 = warp(
                    torch.full(
                        size=(p.shape[0],), fill_value=float(t), device=p.device
                    ),
                    p.clone(),
                )

                loss = F.smooth_l1_loss(p_target, p0)
                optimizer.zero_grad()
                # grad_scaler.scale(loss).backward()
                loss.backward()
                optimizer.step()
                print(f"flow field: step {step} loss {loss}")
                step += 1

"""
Code for testing
import matplotlib.pyplot as plt


def plot_points(points, pause=None):
    fig = plt.figure(0)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    if(pause == None):
        plt.show()
    else:
        plt.pause(pause)

def test(warp, timestamps, points):
    p = points[1]
    t = timestamps[1]

    for t in torch.linspace(0, 1, 22):
        p0 = (
            warp(
                torch.full(size=(p.shape[0],), fill_value=float(t), device=p.device),
                p.clone(),
            )
            .detach()
            .numpy()
        )
        plot_points(p0, pause=0.5)


from datasets.dnerf_synthetic import load_json_file, load_verts
from mlp import ZD_NeRFRadianceField

timestamps, points = load_verts(load_json_file(".", "val"))
radiance_field = ZD_NeRFRadianceField()
train_flow_field(radiance_field.warp, timestamps, points)
test(radiance_field.warp, timestamps, points)
"""