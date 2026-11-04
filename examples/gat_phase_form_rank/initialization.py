"""Copy untrained common columns, drop only explicit time, zero new context."""
import torch


def mapped_initial(canonical, source, target):
    if source.config.phase_rank_features is not None or source.config.temporal_mode != 'time_features':
        raise ValueError('Canonical initialization must be the unextended corrected time model.')
    result = {name:value.clone() for name,value in canonical.items()}
    base_edge = source.edge_dim - len(source.temporal_columns)
    keep_edge = base_edge if target.config.temporal_mode == 'snapshot' else source.edge_dim
    for name, value in canonical.items():
        if name.startswith('layers.') and name.endswith(('.e.weight', '.u.weight')):
            new = value.new_zeros((value.shape[0], target.edge_dim))
            new[:, :keep_edge] = value[:, :keep_edge]
            result[name] = new
    offset = 4*source.config.embedding_dim
    if offset != 4*target.config.embedding_dim:
        raise ValueError('Canonical and destination embedding widths differ.')
    for name in ('head.0.weight', 'goal_head.weight'):
        before = canonical[name]
        after = before.new_zeros((before.shape[0], offset+target.fixture_dim))
        after[:, :offset] = before[:, :offset]
        for j, column in enumerate(target.fixture_encoded_columns):
            if column in source.fixture_encoded_columns:
                after[:, offset+j] = before[:, offset+source.fixture_encoded_columns.index(column)]
            elif not column.startswith('phase::'):
                raise ValueError('Unexpected changed ordinary fixture column.')
        result[name] = after
    return result
