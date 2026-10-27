"""Pure PyTorch edge-aware synchronous graph encoder (imported lazily)."""
import torch
from torch import nn


def grouped_softmax(scores, receiver, log_weight, nodes):
    """Stable incoming-edge softmax, retaining even extremely old edges."""
    if not len(receiver):
        return scores
    logits = scores.double() + log_weight.double()[:, None]
    groups = receiver[:, None].expand_as(logits)
    maximum = logits.new_full((nodes, logits.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
    weights = (logits - maximum[receiver]).exp()
    total = logits.new_zeros((nodes, logits.shape[1]))
    total.index_add_(0, receiver, weights)
    return (weights / total[receiver]).to(scores.dtype)


class EdgeLayer(nn.Module):
    def __init__(self, d, heads, edge_dim):
        super().__init__()
        self.d, self.heads = d, heads
        self.q = nn.Linear(d, d * heads, bias=False)
        self.k = nn.Linear(d, d * heads, bias=False)
        self.e = nn.Linear(edge_dim, d * heads, bias=False)
        self.v = nn.Linear(d, d * heads, bias=False)
        self.u = nn.Linear(edge_dim, d * heads, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, d))
        nn.init.xavier_uniform_(self.attention)
        self.output = nn.Linear(d * heads, d, bias=False)

    def forward(self, h, x, receiver, sender, log_weight, gate, *, return_attention=False, sender_state=None):
        if not len(receiver):
            return (h, h.new_empty((0, self.heads))) if return_attention else h
        shape = (-1, self.heads, self.d)
        scores = (torch.nn.functional.leaky_relu(
            (self.q(h[receiver]) + self.k(h[sender] if sender_state is None else sender_state) + self.e(x)).reshape(shape),
            negative_slope=.2) * self.attention).sum(-1)
        alpha = grouped_softmax(scores, receiver, log_weight, len(h))
        values = torch.tanh((self.v(h[sender] if sender_state is None else sender_state) + self.u(x)).reshape(shape))
        messages = h.new_zeros((len(h), self.heads, self.d))
        messages.index_add_(0, receiver, alpha[:, :, None] * values)
        result = h + gate[:, None] * self.output(messages.flatten(1))
        return (result, alpha) if return_attention else result


class TemporalNetwork(nn.Module):
    def __init__(self, config, leagues, edge_dim, fixture_dim):
        super().__init__()
        d = config.embedding_dim
        self.temporal_mode = config.temporal_mode
        self.sender_state_mode = config.sender_state_mode
        self.global_prior = nn.Parameter(torch.zeros(d))
        self.league = nn.Embedding(leagues + 1, d, padding_idx=0)
        self.entry = nn.Embedding(4, d, padding_idx=0)
        nn.init.normal_(self.league.weight, std=.001)
        nn.init.normal_(self.entry.weight, std=.001)
        with torch.no_grad():
            self.league.weight[0].zero_()
            self.entry.weight[0].zero_()
        self.heatmap_config = config.heatmaps
        self.spatial_encoder = None
        if config.heatmaps is not None:
            from ..neural.network import HeatmapEncoder
            spatial = config.heatmaps
            self.spatial_encoder = HeatmapEncoder(spatial.encoder, 2, spatial.grid.grid_size, spatial.grid.grid_size)
            edge_dim += spatial.encoder.embedding_dim + 2
        self.layers = nn.ModuleList([EdgeLayer(d, config.heads, edge_dim) for _ in range(config.layers)])
        self.statistics_config = config.statistics_forecast
        original_dim = 4*d + fixture_dim
        self.head = nn.Sequential(nn.Linear(original_dim, config.hidden_width),
                                  nn.ReLU(), nn.Linear(config.hidden_width, 3))

        self.statistics_head = None
        if config.statistics_forecast is not None:
            from .statistics_network import StatisticsHead
            self.statistics_head = StatisticsHead(config.statistics_forecast, original_dim, config.hidden_width)
            route = config.statistics_forecast.route
            if route != 'auxiliary':
                width = self.statistics_head.summary_dim + (original_dim if route == 'combined' else 0)
                self.head = nn.Sequential(nn.Linear(width, config.hidden_width), nn.ReLU(), nn.Linear(config.hidden_width, 3))

        # Append after all existing modules: common initialization is unchanged.
        self.goal_config = config.goal_auxiliary
        self.goal_head = nn.Linear(original_dim, 2) if self.goal_config is not None else None
        if self.goal_config is not None and self.goal_config.architecture == 'attack_defence':
            from .goal_network import AttackDefenceHead
            self.goal_head = AttackDefenceHead(self.goal_config, d)

    def prior(self, league, entry):
        return self.global_prior + self.league(league) + (league != 0)[:, None] * self.entry(entry)

    def encode(self, data, *, return_attention=False, return_auxiliary=False, _validated_targets=False):
        if self.temporal_mode == 'historical':
            return self._encode_historical(data, return_attention=return_attention, return_auxiliary=return_auxiliary)
        h0 = self.prior(data["league"], data["entry"])
        h = h0
        if 'initial_state' in data:
            if self.sender_state_mode == 'layer_zero':
                raise ValueError('Layer-zero context requires ordinary priors, not a carried initial state.')
            h = data['initial_state']
        x = data['x']
        auxiliary = h.sum()*0
        if self.spatial_encoder is not None:
            maps, masks = data['maps'], data['map_available']
            encoded = []
            denominator = masks.any(dim=1).sum().clamp_min(1)
            for start in range(0, len(maps), self.heatmap_config.batch_size):
                m, a = maps[start:start+self.heatmap_config.batch_size], masks[start:start+self.heatmap_config.batch_size]
                z = self.spatial_encoder(m, a)
                encoded.append(z)
                if return_auxiliary:
                    auxiliary = auxiliary + self.spatial_encoder._reconstruction_errors(
                        z, m, a, validate=not _validated_targets).sum()/denominator
            z = torch.cat(encoded) if encoded else maps.new_empty((0, self.heatmap_config.encoder.embedding_dim))
            x = torch.cat((x, z, masks), dim=1)
        attention = []
        for layer in self.layers:
            # h0 is local to this forward, trainable, and never updated in place.
            # EdgeLayer uses this single sender route for both K and V.
            sender = h0[data['sender']] if self.sender_state_mode == 'layer_zero' else None
            value = layer(h, x, data["receiver"], data["sender"], data["log_weight"], data["gate"],
                          return_attention=return_attention, sender_state=sender)
            if return_attention:
                h, alpha = value
                attention.append(alpha)
            else:
                h = value
        if return_attention:
            return h, attention
        return (h, auxiliary) if return_auxiliary else h

    def _encode_historical(self, data, *, return_attention=False, return_auxiliary=False):
        if 'temporal_layers' not in data or len(data['temporal_layers']) != len(self.layers):
            raise ValueError('Historical attention requires a matching time-indexed input plan.')
        base = data['temporal_base']
        h = self.prior(base['league'], base['entry'])
        attention = []
        for layer, batch in zip(self.layers, data['temporal_layers']):
            result = layer(h[batch['self_index']], batch['x'], batch['receiver'], batch['sender'],
                           batch['log_weight'], batch['gate'], sender_state=h[batch['sender']],
                           return_attention=return_attention)
            if return_attention:
                h, alpha = result
                attention.append(alpha[batch['root_edge_index']])
            else:
                h = result
        if return_attention:
            return h, attention
        return (h, h.sum()*0) if return_auxiliary else h

    def forward(self, data, home, away, fixture, *, return_auxiliary=False, return_statistics=False, return_goals=False, return_goal_details=False, _validated_targets=False):
        result = self.encode(data, return_auxiliary=return_auxiliary, _validated_targets=_validated_targets)
        h, auxiliary = result if return_auxiliary else (result, None)
        a, b = h[home], h[away]
        inputs = torch.cat((a, b, a-b, a*b, fixture), dim=1)
        rates = None
        details = None
        if return_goals or return_goal_details:
            if self.goal_head is None:
                raise ValueError('Goal predictions require goal_auxiliary enabled.')
            if self.goal_config.architecture == 'linear':
                goal_inputs = inputs.detach() if self.goal_config.detach_encoder else inputs
                rates = torch.nn.functional.softplus(self.goal_head(goal_inputs)) + self.goal_config.rate_floor
                details = dict(rates=rates, log_rates=rates.log())
            else:
                ga, gb = (a.detach(), b.detach()) if self.goal_config.detach_encoder else (a, b)
                details = self.goal_head(ga, gb, data['goal_competition'], data['goal_neutral'])
                rates = details['rates']
        forecasts = None
        if self.statistics_head is not None:
            forecasts, summary = self.statistics_head(inputs)
            if self.statistics_config.route == 'combined':
                inputs = torch.cat((inputs, summary), dim=1)
            elif self.statistics_config.route == 'bottleneck':
                inputs = summary
        logits = self.head(inputs)
        if return_goal_details:
            return logits, details
        if return_goals:
            return logits, rates
        if return_statistics:
            return logits, auxiliary, forecasts
        return (logits, auxiliary) if return_auxiliary else logits
