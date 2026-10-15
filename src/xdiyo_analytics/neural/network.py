"""Shared neural blocks. Imported only when a CNN model is constructed."""
import torch
from torch import nn


def activation(name):
    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[name]()


class HeatmapEncoder(nn.Module):
    def __init__(self, config, maps, height, width):
        super().__init__()
        if min(height, width) < config.pool_size:
            raise ValueError('Pooling cannot exceed grid dimensions.')
        self.config, self.maps, self.height, self.width = config, maps, height, width
        layers = []
        previous = 2 * maps
        for channel in config.channels:
            layers.extend([nn.Conv2d(previous, channel, config.kernel_size, padding=config.kernel_size//2), activation(config.activation)])
            previous = channel
        layers.extend([nn.AvgPool2d(config.pool_size), nn.Flatten(),
                       nn.Linear(previous*(height//config.pool_size)*(width//config.pool_size), config.embedding_dim)])
        self.encoder = nn.Sequential(*layers)
        self.decoder = None
        if config.reconstruction is not None:
            self.decoder = nn.Sequential(nn.Linear(config.embedding_dim, config.reconstruction.hidden_width),
                activation(config.activation), nn.Linear(config.reconstruction.hidden_width, maps*height*width))

    def forward(self, maps, available):
        if len(maps) == 0:
            return maps.new_empty((0, self.config.embedding_dim))
        planes = available[:, :, None, None].expand_as(maps)
        z = self.encoder(torch.cat((maps, planes), dim=1))
        return z * available.any(dim=1, keepdim=True)

    def reconstruction_errors(self, z, maps, available):
        """Per-item observed-map loss; KL is D(target || decoded distribution).

        MSE averages cells. KL sums cells of each unit-mass distribution, then
        both losses average observed maps. Inputs to the encoder are unchanged.
        """
        return self._reconstruction_errors(z, maps, available, validate=True)

    def _reconstruction_errors(self, z, maps, available, *, validate):
        # Only native owned populations may skip validation. Keep normalization,
        # operation order and device arithmetic identical for both paths.
        if self.decoder is None:
            return z.sum(dim=1)*0
        decoded = self.decoder(z).reshape_as(maps)
        if self.config.reconstruction.loss == 'kl':
            target = torch.where(available[:, :, None, None].bool(), maps, 0).flatten(2)
            if validate and (not torch.isfinite(target).all() or (target < 0).any()):
                raise ValueError('KL reconstruction needs finite nonnegative observed maps.')
            # Scale first so count grids cannot overflow their normalization sum.
            maximum = target.amax(dim=-1, keepdim=True)
            if validate and ((maximum.squeeze(-1) == 0) & available.bool()).any():
                raise ValueError('KL reconstruction needs positive mass in every observed map.')
            target = target / torch.where(maximum > 0, maximum, 1)
            total = target.sum(dim=-1, keepdim=True)
            target = target / torch.where(total > 0, total, 1)
            log_probability = decoded.flatten(2).log_softmax(dim=-1)
            # kl_div uses the zero-target convention 0*log(0/q)=0.
            errors = torch.nn.functional.kl_div(log_probability, target, reduction='none').sum(dim=-1) * available
        else:
            errors = (decoded-maps).square().mean(dim=(2, 3)) * available
        return errors.sum(dim=1)/available.sum(dim=1).clamp_min(1)


class HeatmapNetwork(nn.Module):
    """Standalone prediction head with a shared, optionally decoded encoder."""
    def __init__(self, spec, maps, size, outputs):
        super().__init__()
        self.encoder = HeatmapEncoder(spec.encoder, maps, size, size)
        self.head = nn.Sequential(
            nn.Linear(spec.encoder.embedding_dim + maps + 2*len(spec.numeric_features), spec.hidden_width),
            activation(spec.encoder.activation), nn.Linear(spec.hidden_width, outputs))

    def forward(self, maps, masks, numeric):
        z = self.encoder(maps, masks)
        return self.head(torch.cat((z, masks, numeric), dim=1)), z
