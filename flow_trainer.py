import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchdiffeq import odeint as odeint



def train_flow_field(
    warp: nn.Module, timestamps: torch.Tensor, points: torch.Tensor, epochs=50
):
    points.requires_grad_()
    timestamps.requires_grad_()

    optimizer = torch.optim.Adam(warp.parameters(), 5e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-2, verbose=True)

    step = 0
    batch_size = 1
    loss = 0
    go = True
    while go:
        pred = odeint(warp.odefunc, points[0], timestamps, rtol=1e-4, atol=1e-5)[1:]
        loss += F.smooth_l1_loss(pred, points[1:])
        if(step % batch_size == 0):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()
            print(f"flow field: step {step} loss {loss}")
            loss = 0
        if(step >= epochs):
            go = False
            break

        # grad_scaler.scale(loss).backward()
        step += 1

def train_flow_field_old(
    warp: nn.Module, timestamps: torch.Tensor, points: torch.Tensor, epochs=50
):
    points.requires_grad_()
    timestamps.requires_grad_()

    optimizer = torch.optim.Adam(warp.parameters(), 5e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-2, verbose=True)

    step = 0
    batch_size = 1
    dist_loss = 0
    vel_loss = 0
    go = True
    while go:
        for t1, t2, p1, p2 in zip(
            timestamps[:-1], timestamps[1:], points[:-1], points[1:]
        ):
            pred_p2 = warp(t1, t2, p1)
            dist_loss += F.smooth_l1_loss(pred_p2, p2, beta=0.01)
            vel_loss += F.l1_loss(warp.odefunc(t1, p1), torch.zeros_like(p1))
            if(step % batch_size == 0):
                optimizer.zero_grad()
                total_loss = dist_loss + 1e-2 * vel_loss 
                total_loss.backward()
                optimizer.step()
                sched.step()
                print(f"flow field: step {step} loss {total_loss} {dist_loss} {vel_loss}")
                dist_loss = 0
                vel_loss = 0    
            if(step >= epochs):
                go = False
                break

            # grad_scaler.scale(loss).backward()
            step += 1