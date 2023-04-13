import torch
from torchdiffeq import odeint
from mlp import NeuralField
from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test():
    steps = 50
    odefunc = NeuralField(4, 3, 32, 8).to(DEVICE)
    num_data = 2**15
    t = torch.linspace(0, 1, steps).to(DEVICE)

    optim = torch.optim.Adam(odefunc.parameters())
    for i in range(2):
        x = torch.rand(size=(num_data, 3)).to(DEVICE)
        y = torch.rand(size=(steps - 1, num_data, 3)).to(DEVICE)
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

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
      test()
  
open("profiler.out", "w").write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time", row_limit=100))