import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def train_flow_field(
    warp: nn.Module, timestamps: torch.Tensor, points: torch.Tensor, epochs=50
):
    points.requires_grad_()
    timestamps.requires_grad_()

    optimizer = torch.optim.Adam(warp.parameters(), 0.01)

    step = 0
    while step < epochs:
        for t1, t2, p1, p2 in zip(
            timestamps[:-1], timestamps[1:], points[:-1], points[1:]
        ):
            pred_p2 = warp(t1, t2, p1)
            loss = F.smooth_l1_loss(pred_p2, p2)
            optimizer.zero_grad()
            # grad_scaler.scale(loss).backward()
            loss.backward()
            optimizer.step()
            print(f"flow field: step {step} loss {loss}")
            step += 1


def train_flow_field_mshooting(
    warp: nn.Module, timestamps: torch.Tensor, points: torch.Tensor, epochs=50
):
    points.requires_grad_()
    timestamps.requires_grad_()

    optimizer = torch.optim.Adam(warp.parameters(), 0.01)

    step = 0
    p_target = points[0].clone()
    while step < epochs:
        pred_points = warp(
            timestamps,
            points[-1].clone(),
        )

        print(pred_points)
        print(pred_points.shape)

        loss = F.l1_loss(p_target, 0)
        optimizer.zero_grad()
        # grad_scaler.scale(loss).backward()
        loss.backward()
        optimizer.step()
        print(f"flow field: step {step} loss {loss}")
        step += 1


"""
def plot_points(points, ax, pause=None):
    ax.axes.set_xlim3d(left=-3, right=3)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-3, top=3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    if pause == None:
        plt.show()
    else:
        plt.pause(pause)


def test(warp, timestamps, points):
    fig = plt.figure(0)
    ax = fig.add_subplot(projection="3d")
    p = points[0]
    plot_points(p.detach().numpy(), ax, pause=0.5)

    for t in torch.linspace(0, 1, 22):
        p = warp(
            t,
            t + 0.05,
            p.clone(),
        )
        plot_points(p.detach().numpy(), ax, pause=0.5)


from datasets.dnerf_synthetic import load_json_file, load_verts
from mlp import ZD_NeRFRadianceField, DivergenceFreeNeuralField
import matplotlib.pyplot as plt

timestamps, points = load_verts(load_json_file(".", "train"))

radiance_field = ZD_NeRFRadianceField()
warp = radiance_field.warp
# warp = ODEBlock_MS(DivergenceFreeNeuralField(width=16, depth=5))

fig = plt.figure(0)
ax = fig.add_subplot(projection="3d")
for p in points:
    plot_points(p.detach().numpy(), ax, pause=1.0)

train_flow_field(
    warp,
    timestamps,
    points,
    epochs=1000,
)
for i in range(100):
    test(warp, timestamps, points)"""
