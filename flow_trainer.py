import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from torchdiffeq import odeint as odeint


def train_flow_field(
    odefunc: nn.Module,
    timestamps_base: torch.Tensor,
    points_base: torch.Tensor,
    epochs=50,
    steps_ahead=2,
):
    """points_base.requires_grad_()
    timestamps.requires_grad_()"""

    optimizer = torch.optim.Adam(odefunc.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            epochs // 2,
            epochs * 3 // 4,
            epochs * 5 // 6,
            epochs * 9 // 10,
        ],
        gamma=0.1,
    )

    step = 0
    batch_size = 4
    loss = 0
    go = True
    while go:
        # Add random noise
        # points = points_base + torch.rand_like(points_base, device=points_base.device) * alpha
        reversed = 1 if torch.rand(size=(1,)) > 0.5 else -1
        start_t = int(torch.rand(size=(1,)) * (len(timestamps_base) - steps_ahead))
        start_t += steps_ahead if reversed == -1 else 0
        end_t = start_t + reversed * steps_ahead

        points = (
            torch.flip(points_base[end_t : start_t + 1], dims=(0,))
            if reversed == -1
            else points_base[start_t : end_t + 1]
        )

        timestamps = (
            torch.flip(timestamps_base[end_t : start_t + 1], dims=(0,))
            if reversed == -1
            else timestamps_base[start_t : end_t + 1]
        )

        pred = odeint(
            odefunc,
            points[0],
            timestamps,
            rtol=1e-7,
            atol=1e-9,
        )[1:]
        loss = loss + F.mse_loss(pred, points[1:])
        if step % batch_size == 0 and step > 0:
            loss = loss / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()
            print(f"flow field: step {step} loss {loss}")
            loss = 0
        if step >= epochs:
            go = False
            break

        # grad_scaler.scale(loss).backward()
        step += 1


def train_flow_field_old(
    odefunc: nn.Module,
    timestamps: torch.Tensor,
    points_base: torch.Tensor,
    epochs=50,
    alpha=0.05,
    ms_shooting=True,
):
    """points_base.requires_grad_()
    timestamps.requires_grad_()"""

    optimizer = torch.optim.Adam(odefunc.parameters(), 5e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            epochs // 2,
            epochs * 3 // 4,
            epochs * 5 // 6,
            epochs * 9 // 10,
        ],
        gamma=0.33,
    )

    loss_scaler = 0.95 ** torch.linspace(0, len(timestamps) - 2, len(timestamps) - 1)
    loss_scaler = loss_scaler.unsqueeze(1).unsqueeze(1)
    step = 0
    batch_size = 1
    loss = 0
    go = True
    while go:
        # Add random noise
        # points = points_base + torch.rand_like(points_base, device=points_base.device) * alpha
        points = points_base
        if ms_shooting:
            """depracated need to remove
            pred = odeint_mshooting(
                odefunc,
                points[0],
                timestamps,
                solver="mszero",
                B0=points.clone(),
                fine_steps=len(points),
            )[1][1:]"""
        else:
            pred = odeint(odefunc, points[0], timestamps, rtol=1e-7, atol=1e-9)[1:]
        loss += torch.mean(((pred - points[1:]) * loss_scaler) ** 2)
        if step % batch_size == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()
            print(f"flow field: step {step} loss {loss}")
            loss = 0
        if step >= epochs:
            go = False
            break

        # grad_scaler.scale(loss).backward()
        step += 1
