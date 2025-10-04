#
import torch
import torch.nn as nn
import math
from typing import Optional, Union, List
from enum import Enum


class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SELU = "selu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"


class PoolingType(Enum):
    MAX = "max"
    AVG = "avg"
    ADAPTIVE = "adaptive"



def get_activation_fn(
    activation: Union[str, ActivationType], leaky_relu_slope: float = 0.01
) -> nn.Module:
    if isinstance(activation, ActivationType):
        activation = activation.value

    if activation == ActivationType.RELU.value:
        return nn.ReLU()
    elif activation == ActivationType.GELU.value:
        return nn.GELU()
    elif activation == ActivationType.SELU.value:
        return nn.SELU()
    elif activation == ActivationType.TANH.value:
        return nn.Tanh()
    elif activation == ActivationType.LEAKY_RELU.value:
        return nn.LeakyReLU(leaky_relu_slope)
    else:
        return nn.ReLU()


class NHiTSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        pooling_kernel_size: int = 1,
        num_coeffs: int = 10,
        activation_fn: Union[str, ActivationType] = ActivationType.RELU,
        pooling_type: Union[str, PoolingType] = PoolingType.MAX,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        leaky_relu_slope: float = 0.01,
    ):
        super(NHiTSBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling_kernel_size = pooling_kernel_size
        self.activation_fn = get_activation_fn(activation_fn, leaky_relu_slope)
        self.num_coeffs = num_coeffs
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.n_theta_backcast = max(self.input_size // self.num_coeffs, 1)
        self.n_theta_forecast = max(self.output_size // self.num_coeffs, 1)
        # Pooling layer
        if pooling_type == PoolingType.MAX.value or pooling_type == PoolingType.MAX:
            self.pooling_layer = nn.MaxPool1d(
                kernel_size=self.pooling_kernel_size,
                stride=self.pooling_kernel_size,
                ceil_mode=True,
            )
        elif pooling_type == PoolingType.AVG.value or pooling_type == PoolingType.AVG:
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.pooling_kernel_size,
                stride=self.pooling_kernel_size,
                ceil_mode=True,
            )
        elif (
            pooling_type == PoolingType.ADAPTIVE.value
            or pooling_type == PoolingType.ADAPTIVE
        ):
            output_size_pooled = max(1, input_size // pooling_kernel_size)
            self.pooling_layer = nn.AdaptiveAvgPool1d(output_size_pooled)

        pooled_input_size = math.ceil(self.input_size / self.pooling_kernel_size)

        # MLP layers with regularization
        layers = []
        for i in range(num_layers):
            in_dim = pooled_input_size if i == 0 else hidden_size
            out_dim = hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)
        self.forecast_head = nn.Linear(hidden_size, self.n_theta_forecast)
        self.backcast_head = nn.Linear(hidden_size, self.n_theta_backcast)

    def forward(self, x):
        x_pooled = self.pooling_layer(x.unsqueeze(1)).squeeze(1)  # Apply pooling
        h = self.mlp(x_pooled)
        theta_f = self.forecast_head(h)
        theta_b = self.backcast_head(h)
        # Reshape coefficients before interpolation
        theta_f_reshape = theta_f.unsqueeze(1)
        theta_b_reshape = theta_b.unsqueeze(1)
        # Interpolate coefficients to match output sizes
        forecast = nn.functional.interpolate(
            theta_f_reshape, size=self.output_size, mode="linear", align_corners=False
        ).squeeze(1)
        backcast = nn.functional.interpolate(
            theta_b_reshape, size=self.input_size, mode="linear", align_corners=False
        ).squeeze(1)

        return backcast, forecast


class NHiTSStack(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super(NHiTSStack, self).__init__()
        self.blocks = blocks

    def forward(self, x):
        residual = x
        output_size = self.blocks[0].output_size
        forecast = torch.zeros(x.size(0), output_size, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class NHiTS(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_blocks: int = 2,
        num_stacks: int = 2,
        stack_pooling_kernel_sizes: Optional[List[int]] = None,
        hidden_size: Union[int, List[int]] = 64,
        num_layers: Union[int, List[int]] = 2,
        pooling_kernel_size: Union[int, List[int]] = 1,
        num_coeffs: Union[int, List[int]] = 10,
        activation_fn: Union[str, ActivationType] = ActivationType.RELU,
        pooling_type: Union[str, PoolingType] = PoolingType.MAX,
        dropout_rate: Union[float, List[float]] = 0.1,
        use_layer_norm: bool = True,
        use_residual_connections: bool = True,
        leaky_relu_slope: float = 0.01,
    ):
        super(NHiTS, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_residual_connections = use_residual_connections

        # Set default pooling kernel sizes if not provided
        if stack_pooling_kernel_sizes is None:
            stack_pooling_kernel_sizes = [4, 2][:num_stacks]
        elif len(stack_pooling_kernel_sizes) != num_stacks:
            stack_pooling_kernel_sizes = stack_pooling_kernel_sizes[:num_stacks]

        # Convert parameters to lists if needed
        def to_list(param, num_stacks):
            if isinstance(param, (int, float, str, ActivationType, PoolingType)):
                return [param] * num_stacks
            elif isinstance(param, list) and len(param) != num_stacks:
                return param[:num_stacks] + [param[-1]] * (num_stacks - len(param))
            return param

        hidden_sizes = to_list(hidden_size, num_stacks)
        num_layers_list = to_list(num_layers, num_stacks)
        pooling_kernel_sizes_list = to_list(
            pooling_kernel_size, num_stacks * num_blocks
        )
        num_coeffs = to_list(num_coeffs, num_stacks * num_blocks)
        activation_fns = to_list(activation_fn, num_stacks * num_blocks)
        pooling_types = to_list(pooling_type, num_stacks * num_blocks)
        dropout_rates = to_list(dropout_rate, num_stacks * num_blocks)

        self.stacks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i in range(num_stacks):
            kernel_size = stack_pooling_kernel_sizes[i]
            downsample_layer = nn.MaxPool1d(
                kernel_size=kernel_size, stride=kernel_size, ceil_mode=True
            )
            self.downsample_layers.append(downsample_layer)
            stack_input_size = math.ceil(self.input_size / kernel_size)
            blocks = nn.ModuleList()

            for j in range(num_blocks):
                block_idx = i * num_blocks + j
                block = NHiTSBlock(
                    input_size=stack_input_size,
                    output_size=output_size,
                    hidden_size=hidden_sizes[i]
                    if isinstance(hidden_sizes, list)
                    else hidden_sizes,
                    num_layers=num_layers_list[i]
                    if isinstance(num_layers_list, list)
                    else num_layers_list,
                    pooling_kernel_size=pooling_kernel_sizes_list[block_idx]
                    if isinstance(pooling_kernel_sizes_list, list)
                    else pooling_kernel_sizes_list,
                    num_coeffs=num_coeffs[block_idx]
                    if isinstance(num_coeffs, list)
                    else num_coeffs,
                    activation_fn=activation_fns[block_idx]
                    if isinstance(activation_fns, list)
                    else activation_fns,
                    pooling_type=pooling_types[block_idx]
                    if isinstance(pooling_types, list)
                    else pooling_types,
                    dropout_rate=dropout_rates[block_idx]
                    if isinstance(dropout_rates, list)
                    else dropout_rates,
                    use_layer_norm=use_layer_norm,
                    leaky_relu_slope=leaky_relu_slope,
                )
                blocks.append(block)

            self.stacks.append(NHiTSStack(blocks))

    def forward(self, x):
        final_forecast = torch.zeros(
            x.size(0), self.output_size, dtype=x.dtype, device=x.device
        )
        for i, stack in enumerate(self.stacks):
            residual = self.downsample_layers[i](x.unsqueeze(1)).squeeze(1)
            stack_forecast = stack(residual)
            final_forecast = final_forecast + stack_forecast
        return final_forecast
