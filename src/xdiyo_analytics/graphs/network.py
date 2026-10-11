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

    def forward(self, h, x, receiver, sender, log_weight, gate, *, return_attention=False):
        if not len(receiver):
            return (h, h.new_empty((0, self.heads))) if return_attention else h
        shape = (-1, self.heads, self.d)
        scores = (torch.nn.functional.leaky_relu(
            (self.q(h[receiver]) + self.k(h[sender]) + self.e(x)).reshape(shape),
            negative_slope=.2) * self.attention).sum(-1)
        alpha = grouped_softmax(scores, receiver, log_weight, len(h))
        values = torch.tanh((self.v(h[sender]) + self.u(x)).reshape(shape))
        messages = h.new_zeros((len(h), self.heads, self.d))
        messages.index_add_(0, receiver, alpha[:, :, None] * values)
        result = h + gate[:, None] * self.output(messages.flatten(1))
        return (result, alpha) if return_attention else result


class TemporalNetwork(nn.Module):
    def __init__(self, config, leagues, edge_dim, fixture_dim):
        super().__init__()
        d = config.embedding_dim
        self.global_prior = nn.Parameter(torch.zeros(d))
        self.league = nn.Embedding(leagues + 1, d, padding_idx=0)
        self.entry = nn.Embedding(4, d, padding_idx=0)
        nn.init.normal_(self.league.weight, std=.001)
        nn.init.normal_(self.entry.weight, std=.001)
        with torch.no_grad():
            self.league.weight[0].zero_()
            self.entry.weight[0].zero_()
        self.layers = nn.ModuleList([EdgeLayer(d, config.heads, edge_dim) for _ in range(config.layers)])
        self.head = nn.Sequential(nn.Linear(4*d + fixture_dim, config.hidden_width),
                                  nn.ReLU(), nn.Linear(config.hidden_width, 3))

    def prior(self, league, entry):
        return self.global_prior + self.league(league) + (league != 0)[:, None] * self.entry(entry)

    def encode(self, data, *, return_attention=False):
        h = self.prior(data["league"], data["entry"])
        attention = []
        for layer in self.layers:
            value = layer(h, data["x"], data["receiver"], data["sender"], data["log_weight"], data["gate"], return_attention=return_attention)
            if return_attention:
                h, alpha = value
                attention.append(alpha)
            else:
                h = value
        return (h, attention) if return_attention else h

    def forward(self, data, home, away, fixture):
        h = self.encode(data)
        a, b = h[home], h[away]
        return self.head(torch.cat((a, b, a-b, a*b, fixture), dim=1))
