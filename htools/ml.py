import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BaseModel(nn.Module):
    """Parent class for Pytorch models that provides some convenient
    functionality.

    As shown below, the child class should inherit from BaseModel.
    locals() must be passed to the super().__init__() method so we can record
    how the model was initialized, which is convenient for saving and loading.

    Examples
    ---------
    class ConvNet(BaseModel):
        def __init__(self, x_dim, batch_norm=True):
            super().__init__(locals())
            self.x_dim = x_dim
            self.batch_norm = batch_norm
        def forward(self, x):
            ...

    cnn = ConvNet(3)

    While training, we can save weights and other information by calling:
    >>> cnn.save(epoch_num)

    Then later, to load the model:
    >>> cnn_trained = ConvNet.from_path('data/model_4.pth')
    """

    def __init__(self, init_variables):
        super().__init__()
        init_variables.pop('self', None)
        init_variables.pop('__class__', None)
        self._init_variables = init_variables
        self.weight_dir = 'data'

    def dims(self):
        """Get shape of each layer's weights."""
        return [tuple(p.shape) for p in self.parameters()]

    def trainable(self):
        """Check which layers are trainable."""
        return [(tuple(p.shape), p.requires_grad) for p in self.parameters()]

    def weight_stats(self):
        """Check mean and standard deviation of each layer's weights."""
        return [stats(p.data, 3) for p in self.parameters()]

    def plot_weights(self):
        """Plot histograms of each layer's weights."""
        n_layers = len(self.dims())
        fig, ax = plt.subplots(n_layers, figsize=(8, n_layers * 1.25))
        for i, p in enumerate(self.parameters()):
            ax[i].hist(p.data.flatten())
            ax[i].set_title(f'Shape: {tuple(p.shape)} Stats: {stats(p.data)}')
        plt.tight_layout()
        plt.show()

    def save(self, epoch, dir_=None, file='model', overwrite=False,
             verbose=True, **kwargs):
        """Save model weights.

        Parameters
        -----------
        epoch: int
            The epoch of training the weights correspond to.
        dir_: str
            Only needs to be passed in the first time to set the weight
            directory for the model. If never passed in, the default directory
            of 'data' will be used.
        file: str
            The first part of the file name to save the weights to. The epoch
            and file extension will be added automatically.
        overwrite: bool
            If True, will overwrite existing weights files if they share the
            same name. If False, will raise an error if the file already
            exists and save normally otherwise.
        verbose: bool
            If True, print message to notify user that weights have been saved.
        **kwargs: any type
            User can optionally provide additional information to save
            (e.g. optimizer state dict).
        """
        if dir_:
            self.weight_dir = dir_
        os.makedirs(self.weight_dir, exist_ok=True)

        path = os.path.join(self.weight_dir, f'{file}_e{epoch}.pth')
        if os.path.exists(path) and not overwrite:
            raise FileExistsError

        data = dict(weights=self.state_dict(),
                    epoch=epoch,
                    params=self._init_variables)
        torch.save({**data, **kwargs}, path)

        if verbose:
            print(f'Epoch {epoch} weights saved to {path}.')

    @classmethod
    def from_path(cls, path, verbose=True):
        """Factory method to load a model from a file containing saved weights.
        Note that this will return a new model in eval mode, since the intended
        use case is inference.

        Parameters
        -----------
        path: str
            File path to load weights from.
        verbose: bool
            If True, print message notifying user which weights have been
            loaded and what mode the model is in.
        """
        data = torch.load(path)
        model = cls(**data['params'])
        model.load_state_dict(data['weights'])
        model.eval()

        if verbose:
            print(f'Epoch {data["epoch"]} weights loaded from {path}.'
                  f'\nModel parameters: {data["params"]}'
                  '\nCurrently in eval mode.')
        return model

    def load_epoch(self, epoch, mode='train', verbose=True):
        """Load previously saved weights. Note that this differs from
        `from_path()` method in several ways:
        -Nothing is returned since we are not creating a new model.
        -By default, model is returned in training mode, since the intended
        use is to quickly revert to a desired set of weights while trianing.
        -This is called on a model instance, whereas from_path is a class
        method used to construct a new object.

        Parameters
        -----------
        epoch: int
            The epoch of training to load weights from.
        mode: str
            Specifies whether to leave the model in train or eval mode.
            Options: ('train', 'eval').
        verbose: bool
            If True, print message notifying user which weights have been
            loaded and what mode the model is in.

        Examples
        ---------
        model = CNN()
        for epoch in range(epochs):
            # Code for forward and backward pass.
            model.save(epoch)

        model.load_epoch(5)

        """
        fname = [file for file in os.listdir(self.weight_dir)
                 if file.endswith(f'_e{epoch}.pth')][0]
        path = os.path.join(self.weight_dir, fname)
        data = torch.load(path)
        self.load_state_dict(data['weights'])
        getattr(self, mode)

        if verbose:
            print(f'Epoch {epoch} weights loaded from {path}. '
                  f'\nModel parameters: {data["params"]}'
                  f'\nCurrently in {mode} mode.')


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


def stats(x, digits=3):
    """Quick wrapper to get mean and standard deviation of a tensor."""
    return round(x.mean().item(), digits), round(x.std().item(), digits)


# def variable_lr_optimizer(groups, lrs, optimizer=torch.optim.Adam, **kwargs):
#     """Get an optimizer that uses different learning rates for different layer
#     groups. Additional keyword arguments can be used to alter momentum and/or
#     weight decay, for example, but for the sake of simplicity these values
#     will be the same across layer groups.
#
#     Parameters
#     -----------
#     groups: nn.ModuleList
#         For this use case, the model should contain a ModuleList of layer
#         groups in the form of Sequential objects. This variable is then passed
#         in so each group can receive its own learning rate.
#     lrs: list[float]
#         A list containing the learning rates to use for each layer group. This
#         should be the same length as the number of layer groups in the model.
#         At times, we may want to use the same learning rate for all groups,
#         and can achieve this by passing in a list containing a single float.
#     optimizer: torch optimizer
#         The Torch optimizer to be created (Adam by default).
#
#     Examples
#     ---------
#     optim = variable_lr_optimizer(model.groups, [3e-3, 3e-2, 1e-1])
#     """
#     if len(lrs) == 1:
#         lrs *= len(groups)
#
#     data = [{'params': group.parameters(), 'lr': lr}
#             for group, lr in zip(groups, lrs)]
#     return optimizer(data, **kwargs)


def variable_lr_optimizer(model=None, groups=None, lrs=[3e-3],
                          optimizer=torch.optim.Adam, **kwargs):
    """Get an optimizer that uses different learning rates for different layer
    groups. Additional keyword arguments can be used to alter momentum and/or
    weight decay, for example, but for the sake of simplicity these values
    will be the same across layer groups.

    Parameters
    -----------
    model: nn.Module
        model.parameters() generator.
    groups: nn.ModuleList (optional)
        For this use case, the model should contain a ModuleList of layer
        groups in the form of Sequential objects. This variable is then passed
        in so each group can receive its own learning rate.
    lrs: list[float]
        A list containing the learning rates to use for each layer group. This
        should be the same length as the number of layer groups in the model.
        At times, we may want to use the same learning rate for all groups,
        and can achieve this by passing in a list containing a single float.
    optimizer: torch optimizer
        The Torch optimizer to be created (Adam by default).

    Examples
    ---------
    optim = variable_lr_optimizer(model.groups, [3e-3, 3e-2, 1e-1])
    """
    assert bool(model is None) + bool(groups is None) == 1

    if model is not None:
        groups = [model]

    assert len(groups) == len(lrs)

    data = [{'params': group.parameters(), 'lr': lr}
            for group, lr in zip(groups, lrs)]
    return optimizer(data, **kwargs)
