#
import torch
import torch.nn as nn
import numpy as onp
from typing import List, cast

class Model(torch.nn.Module):
    R"""
    Model.
    """
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        ...


class MLP(Model):
    R"""
    MLP.
    """
    def __init__(self, /, *, size: int, shapes: List[int]) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        buf = []
        shapes = [size * size] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        x = torch.flatten(x, start_dim=1)
        for (l, linear) in enumerate(self.linears):
            #
            x = linear.forward(x)
            if l < len(self.linears) - 1:
                #
                x = torch.nn.functional.relu(x)
        return x


#
PADDING = 3


class CNN(torch.nn.Module):
    R"""
    CNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # Create a list of Conv2D layers and shared max-pooling layer.
        # Input and output channles are given in `channels`.
        # ```
        # buf_conv = []
        # ...
        # self.convs = torch.nn.ModuleList(buf_conv)
        # self.pool = ...
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        c1 = nn.Conv2d(in_channels = channels[0], out_channels= channels[1], kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING)
        self.p = nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        c2 = nn.Conv2d(in_channels = channels[1], out_channels= channels[2], kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING)
        self.convs = torch.nn.ModuleList([c1, c2])

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        # ```
        # buf = []
        # ...
        # self.linears = torch.nn.ModuleList(buf)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        size1 = ((size - kernel_size_conv + 2 * PADDING) // stride_size_conv) + 1
        size2 = ((size1 - kernel_size_pool) // stride_size_pool) + 1
        size3 = ((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv) + 1
        size4 = ((size3 - kernel_size_pool) // stride_size_pool) + 1

        self.s = channels[2] * size4 * size4
        fc1 = nn.Linear(self.s, shapes[0])
        fc2 = nn.Linear(shapes[0], shapes[1])
        fc3 = nn.Linear(shapes[1], shapes[2])
        self.linears = torch.nn.ModuleList([fc1, fc2, fc3])
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = onp.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        # CNN forwarding whose activation functions should all be relu.
        # YOU SHOULD FILL IN THIS FUNCTION
        x = torch.relu(self.convs[0](x))
        x = self.p(x)
        x = torch.relu(self.convs[1](x))
        x = self.p(x)

        x = x.view(-1, self.s)

        x = torch.relu(self.linears[0](x))
        x = torch.relu(self.linears[1](x))
        x = self.linears[2](x)

        return x

class CGCNN(Model):
    R"""
    CGCNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # This will load precomputed eigenvectors.
        # You only need to define the proper size.
        # proper_size = ...
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        #
        self.basis: torch.Tensor

        proper_size = 5

        # Loaded eigenvectos are stored in `self.basis`
        with open("rf-{:d}.npy".format(proper_size), "rb") as file:
            #
            onp.load(file)
            eigenvectors = onp.load(file)
        self.register_buffer(
            "basis",
            torch.from_numpy(eigenvectors).to(torch.get_default_dtype()),
        )

        # Create G-invariant CNN like CNN, but is invariant to rotation and
        # flipping.
        # linear is the same as CNN.
        # You only need to create G-invariant Conv2D weights and biases.
        # ```
        # buf_weight = []
        # buf_bias = []
        # ...
        # self.weights = torch.nn.ParameterList(buf_weight)
        # self.biases = torch.nn.ParameterList(buf_bias)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        self.channels = channels
        self.kernel_size = kernel_size_conv
        self.stride = stride_size_conv
        w1 = nn.Parameter(torch.rand(channels[1], channels[0], self.basis.shape[0], 1), requires_grad=True)
        b1 = nn.Parameter(torch.rand(channels[1]),requires_grad=True)

        w2 = nn.Parameter(torch.rand(channels[2], channels[1], self.basis.shape[0], 1), requires_grad=True)
        b2 = nn.Parameter(torch.rand(channels[2]), requires_grad=True)

        self.weights = torch.nn.ParameterList([w1, w2])
        self.biases = torch.nn.ParameterList([b1, b2])

        self.p = nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        size1 = ((size - kernel_size_conv + 2 * PADDING) // stride_size_conv) + 1
        size2 = ((size1 - kernel_size_pool) // stride_size_pool) + 1
        size3 = ((size2 - kernel_size_conv + 2 * PADDING) // stride_size_conv) + 1
        size4 = ((size3 - kernel_size_pool) // stride_size_pool) + 1

        self.s = channels[2] * size4 * size4
        fc1 = nn.Linear(self.s, shapes[0])
        fc2 = nn.Linear(shapes[0], shapes[1])
        fc3 = nn.Linear(shapes[1], shapes[2])
        self.linears = torch.nn.ModuleList([fc1, fc2, fc3])

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for (weight, bias) in zip(self.weights, self.biases):
            #
            (_, ch_ins, b1, b2) = weight.data.size()
            a = 1 / onp.sqrt(ch_ins * b1 * b2)
            weight.data.uniform_(-a, a, generator=rng)
            bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        # CG-CNN forwarding whose activation functions should all be relu.
        # Pay attention that your forwarding should be invariant to rotation
        # and flipping.
        # Thus, if you rotate x by 90 degree (see structures.py), output of
        # this function should not change.
        # YOU SHOULD FILL IN THIS FUNCTION
        # print(self.basis.size())
        new_weights = torch.mul(self.weights[0], self.basis)
        new_weights = new_weights.sum(dim=-2)

        new_weights = new_weights.reshape(
            self.channels[1],
            self.channels[0],
            self.kernel_size,
            self.kernel_size,
        )
        out = nn.functional.conv2d(x, weight=new_weights, bias=None, stride=self.stride, padding=PADDING)
        bias = self.biases[0].view(1, self.channels[1], 1, 1)
        x = out + bias

        x = self.p(x)

        new_weights = torch.mul(self.weights[1], self.basis)
        new_weights = new_weights.sum(dim=-2)
        new_weights = new_weights.reshape(
            self.channels[2],
            self.channels[1],
            self.kernel_size,
            self.kernel_size,
        )
        out = nn.functional.conv2d(x, weight=new_weights, bias=None, stride=self.stride, padding=PADDING)
        bias = self.biases[1].view(1, self.channels[2], 1, 1)
        x = out + bias

        x = self.p(x)

        x = x.view(-1, self.s)

        x = torch.relu(self.linears[0](x))
        x = torch.relu(self.linears[1](x))
        x = self.linears[2](x)

        return x