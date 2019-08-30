import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def dims(self):
        """Get shape of each layer's weights."""
        return [tuple(p.shape) for p in self.parameters()]

    def trainable(self):
        """Check which layers are trainable."""
        return [(tuple(p.shape), p.requires_grad) for p in self.parameters()]

    def layer_stats(self):
        """Check mean and standard deviation of each layer's weights."""
        return [stats(p.data, 3) for p in self.parameters()]

    def plot_weights(self):
        n_layers = len(self.dims())
        fig, ax = plt.subplots(n_layers, figsize=(8, n_layers * 1.25))
        for i, p in enumerate(self.parameters()):
            ax[i].hist(p.data.flatten())
            ax[i].set_title(f'Shape: {p.shape} Stats: {stats(p.data)}')
        plt.tight_layout()
        plt.show()


class GRelu(nn.Module):
    """Generic ReLU."""

    def __init__(self, leak=0.0, max=float('inf'), sub=0.0):
        super().__init__()
        self.leak = leak
        self.max = max
        self.sub = sub

    def forward(self, x):
        """Check which operations are necessary to save computation."""
        x = F.leaky_relu(x, self.leak) if self.leak else F.relu(x)
        if self.sub:
            x -= self.sub
        if self.max:
            x = torch.clamp(x, max=self.max)
        return x

    def __repr__(self):
        return f'GReLU(leak={self.leak}, max={self.max}, sub={self.sub})'


JRelu = GRelu(leak=.1, sub=.4, max=6.0)


def conv_block(c_in, c_out, norm=True, **kwargs):
    """Create a convolutional block (the latter referring to a backward
    strided convolution) optionally followed by a batch norm layer. Note that
    batch norm has learnable affine parameters which remove the need for a
    bias in the preceding conv layer. When batch norm is not used, however,
    the conv layer will include a bias term.

    Useful kwargs include kernel_size, stride, and padding (see pytorch docs
    for nn.Conv2d).

    The activation function is not included in this block since we use this
    to create ResBlock, which must perform an extra addition before the final
    activation.

    Parameters
    -----------
    c_in: int
        # of input channels.
    c_out: int
        # of output channels.
    norm: bool
        If True, include a batch norm layer after the conv layer. If False,
        no norm layer will be used.
    """
    bias = True
    if norm:
        bias = False
        layers = [nn.BatchNorm2d(c_out)]
    layers.insert(0, nn.Conv2d(c_in, c_out, bias=bias, **kwargs))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):

    def __init__(self, c_in, activation=JRelu, f=3, stride=1, pad=1,
                 skip_size=2, norm=True):
        """Residual block to be used in CycleGenerator. Note that f, stride,
        and pad must be selected such that the height and width of the input
        remain the same.

        Parameters
        -----------
        c_in: int
            # of input channels.
        skip_size: int
            Number of conv blocks inside the skip connection (default 2).
            ResNet paper notes that skipping a single layer did not show
            noticeable improvements.
        f: int
            Size of filter (f x f) used in convolution. Default 3.
        stride: int
            # of pixels the filter moves between each convolution. Default 1.
        pad: int
            Pixel padding around the input. Default 1.
        norm: str
            'bn' for batch norm, 'in' for instance norm
        """
        super().__init__()
        self.skip_size = skip_size
        self.layers = nn.ModuleList([conv_block(True, c_in, c_in,
                                                kernel_size=f, stride=stride,
                                                padding=pad, norm=norm)
                                     for i in range(skip_size)])
        self.activation = activation

    def forward(self, x):
        x_out = x
        for i, layer in enumerate(self.layers):
            x_out = layer(x_out)

            # Final activation must be applied after addition.
            if i != self.skip_size - 1:
                x_out = self.activation(x_out)

        return self.activation(x + x_out)


def stats(x):
    """Quick wrapper to get mean and standard deviation of a tensor."""
    return round(x.mean().item(), 4), round(x.std().item(), 4)
