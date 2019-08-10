import torch.nn as nn
import matplotlib.pyplot as plt


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def dims(self):
        """Get shape of each layer's weights."""
        return [p.shape for p in self.parameters()]

    def trainable(self):
        """Check which layers are trainable."""
        return [p.requires_grad for p in self.parameters()]

    def layer_stats(self):
        """Check mean and standard deviation of each layer's weights."""
        return [quick_stats(p.data, 3) for p in self.parameters()]

    def plot_weights(self):
        n_layers = len(self.dims())
        fig, ax = plt.subplots(n_layers, figsize=(8, n_layers * 1.25))
        for i, p in enumerate(self.parameters()):
            ax[i].hist(p.data.flatten())
            ax[i].set_title(f'Shape: {p.shape} Stats: {quick_stats(p.data)}')
        plt.tight_layout()
        plt.show()


def quick_stats(x, digits=4):
    return round(x.mean().item(), digits), round(x.std().item(), digits)
