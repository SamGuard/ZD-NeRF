import torch
from torch import nn
from torchdiffeq import odeint
from mlp import NeuralField, ZD_NeRFRadianceField
from utils import enforce_structure
from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

steps = 50
num_data = 2**15
x = torch.rand(size=(num_data, 3)).to(DEVICE)
y = torch.rand(size=(steps - 1, num_data, 3)).to(DEVICE)
t = torch.linspace(0, 1, steps).to(DEVICE)
odefunc = NeuralField(4, 3, 32, 8).to(DEVICE)


def test_odeint():
    optim = torch.optim.Adam(odefunc.parameters())
    for i in range(10):
        pred = odeint(
            func=odefunc,
            y0=x,
            t=t,
            rtol=1e-4,
            atol=1e-5,
        )[1:]
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optim.step()
        print(f"Step {i}, loss {loss}")


def test_enforce():
    radiance_field = ZD_NeRFRadianceField().to(DEVICE)
    optim = torch.optim.Adam(radiance_field.parameters())
    for i in range(1):
        start_keypoints, end_keypoints = enforce_structure(
            radiance_field,
            torch.tensor([-3, -3, -3, 3, 3, 3], dtype=torch.float32, device=DEVICE),
            n_samples=4096,
            max_time_diff=0.25,
            device=DEVICE
        )

        loss = torch.nn.functional.smooth_l1_loss(
            start_keypoints, end_keypoints, beta=0.05
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"{i} {len(start_keypoints)} {len(end_keypoints)} {loss}")

def prof(func):

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            # test_odeint()
            func()

    open("profiler.out", "w").write(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=100,
        )
    )


test_enforce()