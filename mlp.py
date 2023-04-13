"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
import copy

from typing import *
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.functional import jacobian
from functorch import vmap, jacrev, make_functional

from torchdiffeq import odeint_adjoint as torchdiffeq_odeint

from libs.torchdyn.torchdyn.numerics import odeint_mshooting


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (int(self.use_identity) + (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class IdentityEncoder(nn.Module):
    """Identity encoder, returns what is put in"""

    def __init__(self, dim):
        super().__init__()
        self.dim =  dim
    
    @property
    def latent_dim(self,) -> int:
        return self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)


class TimeNeRFRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ) -> None:
        super().__init__()
        #self.posi_encoder = SinusoidalEncoder(4, 0, 10, True)
        #self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.posi_encoder = IdentityEncoder(4)
        self.view_encoder = IdentityEncoder(3)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def join_inputs(self, x, t):
        return torch.cat((x, t), dim=1)

    def query_opacity(self, x, t, step_size):
        density = self.query_density(x, t)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        x = self.join_inputs(x, t)
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, t, condition=None):
        x = self.join_inputs(x, t)
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition)
        return torch.sigmoid(rgb), F.relu(sigma)


class CurlField(nn.Module):
    def __init__(self, neural_field):
        super().__init__()
        self.func: nn.Module = neural_field

    def predict(self, t, x):
        return self.predict_batch(t, x.reshape(1, -1)).squeeze()

    def predict_batch(self, t, x):
        x = self.func(t, x)
        return x

    def curl_func_3d(self, t: torch.Tensor, x: torch.Tensor):
        jac = torch.squeeze(vmap(jacrev(self.predict, argnums=(1)), (None, 0))(t, x))
        if len(jac.shape) == 2:
            jac = jac.unsqueeze(0)
        curl = torch.stack(
            (
                jac[:, 2, 1] - jac[:, 1, 2],
                jac[:, 0, 2] - jac[:, 2, 0],
                jac[:, 1, 0] - jac[:, 0, 1],
            ),
            dim=1,
        )
        return curl

    def print_jac_of_field(self, t: torch.Tensor, x: torch.Tensor):
        jac = vmap(jacrev(self.curl_func_3d, argnums=(1)), (None, 0))(
            t, x.unsqueeze(1)
        ).squeeze()
        print(jac)

    def forward(self, t, x):
        out = self.curl_func_3d(t, x)
        return out

    def get_base_func(self, t, x):
        """
        Get the output of the function without removing div
        For testing/visualisation
        """
        return self.predict_batch(t, x)

    def get_div(self, t: torch.Tensor, x: torch.Tensor):
        """
        Get diveregence of the vec field (should be zero (hopefully))
        """
        x = x.reshape(x.shape[0], 1, x.shape[1])
        div_func = jacrev(self.curl_func_3d, argnums=1)
        return torch.sum(
            vmap(lambda t, x: torch.trace(div_func(t, x).squeeze()), in_dims=(None, 0))(
                t, x
            ),
            dim=0,
        )


class DivergenceFreeNeuralField(nn.Module):
    def __init__(self, spatial_dims=3, other_inputs=1, width=32, depth=8):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.other_inputs = other_inputs

        # One full network for each output dim
        networks = nn.ModuleList()
        for i in range(spatial_dims):
            layers = nn.Sequential(nn.Linear(spatial_dims + other_inputs - 1, width))
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
        for i in range(self.spatial_dims):
            # Remove ith dimension
            _x = torch.cat((x[:, 0:i], x[:, i + 1 :], t), dim=1)
            output[:, i] = self.networks[i](_x).squeeze()
            if i < self.spatial_dims - 1:
                output[:, i] += self.trace_params[i] * x[:, i]
            else:
                # The last dimension is the negative trace of the
                # previous matrix if it was one dimension smaller
                # The ensures the trace == 0
                output[:, i] += -trace_param_residual * x[:, i]
        return output


class NeuralField(nn.Module):
    def __init__(self, in_dim, out_dim, width=32, depth=8):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, width))
        for i in range(depth - 2):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, out_dim))

    def forward(self, t, x):
        x = torch.cat((x, torch.zeros((x.shape[0], 1), device=t.device) + t), dim=1)
        for l in self.layers[:-1]:
            x = torch.tanh(l(x))
        return self.layers[-1](x)


class ODEBlock_Backward(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        time_steps, args = torch.unique(t, sorted=True, return_inverse=True)
        args: torch.Tensor
        time_steps: torch.Tensor

        if len(time_steps) == 1 and time_steps[0] == 0.0:
            return x

        for i, _t in enumerate(time_steps):
            if _t == 0.0:
                continue
            x_index = (args == i).nonzero().squeeze(dim=1)
            t_tensor = torch.tensor([_t, 0.0], device=x.device)
            warped = torchdiffeq_odeint(self.odefunc, x[x_index], t_tensor)
            x[x_index] = warped[-1]
        return x


class ODEBlock_Forward(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, start_t: torch.Tensor, end_t: torch.tensor, x: torch.Tensor):
        """
        Integrates all values in x from start_t to end_t using odefunc
        """
        warped = torchdiffeq_odeint(
            func=self.odefunc, y0=x, t=torch.tensor([start_t, end_t], device=x.device, atol=0.01, rtol=0.1)
        )
        return warped[1]

    """ 
    Old code, does not work if there are more than 2 time stamps (during training).
    For example it will work if there is only data at t=0.0 and t=1.0, but no if
    there is data at t=0.0,0.5,1.0 as it only works forward in time.
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        if len(x) == 0:
            return torch.zeros_like(x)

        # Need to sort in order of time
        time_steps, args = torch.unique(t, sorted=True, return_inverse=True)

        if len(time_steps) == 1 and time_steps[0] == 0.0:
            return x

        has_zero = True
        if not torch.any(time_steps == 0.0):
            has_zero = False
            time_steps = torch.cat((torch.tensor([0]).to("cuda:0"), time_steps), dim=0)

        # Morphed points    
        morphed = torchdiffeq_odeint(
            self.odefunc,
            x,
            time_steps,
            rtol=1e-4,
            atol=1e-3,
        )
        if not has_zero:
            morphed = morphed[1:]
        # Morphed points contains an array which is of the form:
        # morphed[time_stamp][index]
        # As this list is in order of time we need to convert it back to how the time steps were before sorting
        # To this we index by the args array, which will give all points at a given time
        # Then indexing by r gives the morphed point at the time given
        r = torch.linspace(0, x.shape[0] - 1, x.shape[0], dtype=torch.long)

        out = morphed[args, r]

        return out"""
    

class DNeRFRadianceField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
        self.warp = MLP(
            input_dim=self.posi_encoder.latent_dim + self.time_encoder.latent_dim,
            output_dim=3,
            net_depth=4,
            net_width=64,
            skip_layer=2,
            output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        )
        self.nerf = VanillaNeRFRadianceField()

    def query_opacity(self, x, timestamps, step_size):
        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        x = x + self.warp(
            torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1)
        )
        return self.nerf.query_density(x)

    def forward(self, x, t, condition=None):
        x = x + self.warp(
            torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1)
        )
        return self.nerf(x, condition=condition)


class ZD_NeRFRadianceField(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # self.warp = ODEBlock_torchdiffeq(NeuralField(4, 3, 32, 6))
        # self.warp = ODEBlock_torchdiffeq(CurlField(NeuralNet))
        self.warp = ODEBlock_Forward(DivergenceFreeNeuralField(3, 1, 16, 6))
        self.nerf = TimeNeRFRadianceField()

    def query_opacity(self, x, timestamps, step_size):
        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        # x = self.warp(t.flatten(), x)
        return self.nerf.query_density(x, t)

    def forward(self, x, t, condition=None):
        out = self.nerf(x, t, condition=condition)
        return out

    def flow_field_pred(
        self, x: torch.Tensor, dirs: torch.Tensor, t_diff=0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_start = torch.rand(1, device=x.device)[0]
        t_end = t_start + torch.rand(1, device=x.device)[0] * t_diff * 2 - t_diff

        t_start_expanded = torch.full(
            size=(x.shape[0], 1), fill_value=t_start, device=x.device
        )
        t_end_expanded = torch.full(
            size=(x.shape[0], 1), fill_value=t_end, device=x.device
        )

        init_rgb, _ = self.forward(x, t_start_expanded, dirs)  # RGB at the starting point
        x_flow = self.warp(t_start, t_end, x)  # Warp point to new location
        end_rgb, _ = self.forward(
            x_flow, t_end_expanded, dirs
        )  # Sample what the nerf thinks the colour should be here

        alive_mask = self.query_density(x, torch.full(size=(x.shape[0], 1), fill_value=t_start, device=x.device)).squeeze(-1) > 0
        return init_rgb[alive_mask], end_rgb[alive_mask]
