import torch
from datasets.dnerf_synthetic import load_json_file, load_verts,SubjectLoader
from mlp import (
    ZD_NeRFRadianceField,
    DivergenceFreeNeuralField,
    ODEBlock_Forward,
    NeuralField,
    CurlField,
)
from flow_trainer import *
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

from libs.nerflow.run_nerf_helpers import NeRF

import time


class DivergenceFreeNeuralFieldAnnealing(nn.Module):
    def __init__(self, spatial_dims=3, other_inputs=1, width=32, depth=8):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.other_inputs = other_inputs

        # One full network for each output dim
        networks = nn.ModuleList()
        for i in range(spatial_dims):
            layers = nn.Sequential(nn.Linear(spatial_dims + other_inputs, width))
            layers.append(nn.Tanh())
            for i in range(depth - 2):
                layers.append(nn.Linear(width, width))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(width, 1))
            networks.append(layers)

        self.networks = networks

        self.trace_params = torch.nn.ParameterList()
        for i in range(spatial_dims - 1):
            self.trace_params.append(torch.randn((1,)))

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        t = torch.zeros((x.shape[0], 1), device=t.device) + t
        output = torch.zeros(x.shape[0], self.spatial_dims, device=x.device)
        trace_param_residual = torch.sum(
            torch.tensor(self.trace_params, device=x.device)
        )
        x = torch.cat((x, t), dim=1)
        for i in range(self.spatial_dims):
            mask = torch.zeros(size=(x.shape[0], self.spatial_dims + self.other_inputs))

            output[:, i] = self.networks[i](x).squeeze()
            if i < self.spatial_dims - 1:
                output[:, i] += self.trace_params[i] * x[:, i]
            else:
                # The last dimension is the negative trace of the
                # previous matrix if it was one dimension smaller
                # The ensures the trace == 0
                output[:, i] += -trace_param_residual * x[:, i]
        return output


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


def plot_points(points, ax, pause=None):
    ax.axes.set_xlim3d(left=-3, right=3)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-3, top=3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.1)
    if pause == None:
        plt.show()
    else:
        plt.pause(pause)


def test(func: DivergenceFreeNeuralField, timestamps, points):
    timestamps = torch.linspace(0, 1, 21)
    out = odeint_adjoint(func, points[0], timestamps).detach().numpy()
    name = "zdnerf"
    for i, t in enumerate(timestamps[:]):
        fig = plt.figure(0)
        ax = fig.add_subplot(projection="3d")
        if(i%5 == 0):
            plot_points(out[i], ax, pause=0.5)
            fig.savefig(f"testing/figure_out/{name}_{i}")


def eval_flow(flow: callable, res=8):
    #scene_aabb = torch.tensor([-5, -5, -5, 5, 5, 5], dtype=torch.float32)
    scene_aabb = torch.tensor([0,0,0,10,10,10], dtype=torch.float32)
    aabb_size = float(torch.abs((scene_aabb[3:] - scene_aabb[:3]))[0])
    lin = torch.linspace(0, aabb_size, res, dtype=torch.float32)
    x, y, z = torch.meshgrid((lin, lin, lin), indexing="xy")
    sample_points = torch.stack((x, y, z), dim=3).reshape(-1, 3)
    p = sample_points.detach().numpy()
    for t in torch.linspace(0, 1, 10):
        vals = flow(t.unsqueeze(0), sample_points).detach().numpy()
        print(vals)
        ax = plt.figure().add_subplot(projection="3d")
        ax.quiver(
            p[:, 0], p[:, 1], p[:, 2], vals[:, 0], vals[:, 1], vals[:, 2], length=1.0
        )
        plt.show()


def training_loop(points, timestamps_base):
    radiance_field = ZD_NeRFRadianceField(allow_div=True)
    train = False
    #radiance_field.load_state_dict(torch.load("testing/test_out.pt", map_location="cpu"))
    radiance_field.train(train)
    
    # warp = ODEBlock_Forward(DivergenceFreeNeuralField(3, 1, 16, 8, torch.nn.ReLU)) #radiance_field.warp
    # warp = ODEBlock_Forward(CurlField(NeuralField(4, 3, 64, 5)))
    # warp = ODEBlock_Forward(NeuralField(4, 3, 32, 8))
    # warp = ODEBlock_MS(DivergenceFreeNeuralField(width=16, depth=5))
    # warp = ODEBlock_Forward(NeRF(sin_init=True).velocity_module)
    # radiance_field.warp = warp

    """indexs = torch.linspace(0, len(timestamps_base) - 1, 25, dtype=torch.long)
    timestamps_base = timestamps_base[indexs]
    points = points[indexs]"""

    fig = plt.figure(0)
    ax = fig.add_subplot(projection="3d")

    """for p in points:
        plot_points(p.detach().numpy(), ax, pause=.1)
    plt.show()"""
    if train:
        t_start = time.time()
        train_flow_field(
            radiance_field.warp.odefunc,
            timestamps_base,
            points,
            epochs=200, steps_ahead=5
        )
        print("Time to train", time.time() - t_start)
        torch.save(radiance_field.state_dict(), "testing/test_out.pt")
    test(radiance_field.warp.odefunc, timestamps_base, points)
    plt.show()

    

def test_keypoints(points, timestamps):
    train_data = SubjectLoader("brick_v2", "/home/ruilongli/data/dnerf/", split="train")
    radiance_field = ZD_NeRFRadianceField()
    #print(keypoints_loss(radiance_field, points, train_data[0]["rays"].viewdirs, n_samples=8))

timestamps, points = load_verts(load_json_file("testing", "train"))

training_loop(points, timestamps)
#test_keypoints(points, timestamps)