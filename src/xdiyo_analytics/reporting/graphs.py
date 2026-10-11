"""Native diagnostics for retained temporal graph models. No model fitting."""
from dataclasses import dataclass
from html import escape
from numbers import Integral, Real
from typing import ClassVar
import numpy as np
import pandas as pd

from .contracts import Artifact, StudyResult
from .model_diagnostics import LearningCurveReporter
from .post_training import PredictionReporter
from .studies import _plotting, _style
from .teams import TeamCatalog
from .embedding_view import embedding_figure


def _graph(model):
    from ..graphs import TemporalGraphAdapter
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        if isinstance(model, TemporalGraphAdapter):
            return model
        model = getattr(model, 'estimator', getattr(model, 'model', None))
    return None


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def _populations(context, issuance_at, result):
    if context.partition not in ('test', 'score') or context.layout != 'match':
        raise ValueError('Graph representations use test/score fixtures in match layout.')
    if context.pooling == 'mean':
        raise ValueError('Graph representations cannot be averaged across fitted folds.')
    for fold, metadata in context.metadata.groupby(level='fold_id', sort=False):
        model = _graph(context.models.get(fold))
        if model is None:
            result.notes.append(f'Fold {fold}: no retained temporal graph model is available.')
            continue
        issue = model.event_store.issue_times(metadata, model.config)
        time = issue.max() if issuance_at is None else pd.Timestamp(issuance_at)
        if time.tzinfo is None or pd.isna(time) or time not in set(issue):
            raise ValueError('issuance_at must be a timezone-aware issuance represented in the selected evaluation fixtures.')
        # One coordinate system and one cutoff for every point in this panel.
        # Teams come only from this evaluation population, not all store nodes.
        teams = sorted(set(metadata.home_id) | set(metadata.away_id))
        queries = pd.DataFrame({'team_id': teams, 'issuance_at': time})
        yield fold, model, queries


@dataclass(kw_only=True)
class GATLossReporter(LearningCurveReporter):
    """Stored cross-entropy losses, separated by fold and restart attempt.

    Training loss is measured before the step's update; validation loss after
    it. AdamW decay and the prior penalty are not included in these data losses.
    Validation appears only when training reserved a validation population.
    """
    metrics: object = ('train_loss', 'validation_loss')

    def run(self, context):
        result = super().run(context)
        result.title = 'GAT training and validation loss'
        result.notes.extend([
            'Training loss is pre-update; validation loss is post-update. Both are weighted cross-entropy data losses, excluding prior penalty and AdamW decay.',
            'No validation curve is fabricated when validation was not configured. Steps and restart attempts are preserved; curves are not averaged across folds.',
        ])
        return result


@dataclass(kw_only=True)
class EmbeddingReporter(PredictionReporter):
    """Project one frozen encoder at one evaluation issuance per fold.

    tSNE (default) and PCA affect reporting only. The latest selected issuance
    is the default. Raw coordinates, support, gates and model versions are
    exported. Never pool separately fitted encoder coordinate systems. tSNE
    uses min(perplexity, n_teams-1), recorded in the projection table. Degenerate
    populations are annotated, with no invented tSNE/PCA geometry.
    """
    mode: str = 'tSNE'
    issuance_at: str | None = None
    perplexity: float = 30.
    learning_rate: object = 'auto'
    max_iter: int = 1000
    init: str = 'pca'
    metric: str = 'euclidean'
    early_exaggeration: float = 12.
    random_state: int = 0
    whiten: bool = False
    show_badges: bool = True
    catalog: object = None
    supported_types: ClassVar[tuple] = ('per_fold', 'overall')

    def run(self, context):
        if self.mode not in ('tSNE', 'PCA'):
            raise ValueError('mode must be tSNE or PCA.')
        if isinstance(self.perplexity, bool) or not isinstance(self.perplexity, Real) or not np.isfinite(self.perplexity) or self.perplexity <= 0:
            raise ValueError('perplexity must be finite and positive.')
        _positive('max_iter', self.max_iter)
        if self.max_iter < 250:
            raise ValueError('tSNE max_iter must be at least 250.')
        result = StudyResult('Team embeddings', notes=[
            'Each fold uses one frozen encoder and a common issuance. Teams are taken from the selected test/score fixtures. The latest issuance is the default.',
            'Projections are independently fitted per fold for visualization only. Axes and distances are not comparable across folds; tSNE is a local-neighborhood visualization, not a ranking.',
            'Support is decayed historical match mass. The gate controls how much graph messages modify the prior. These diagnostics do not add predictors.',
        ])
        catalog = self.catalog if self.catalog is not None else context.resources.get('team_catalog')
        catalog = catalog if isinstance(catalog, TeamCatalog) else TeamCatalog(catalog or {})
        exports, projections, summaries = [], [], []
        go = _plotting()
        for fold, model, queries in _populations(context, self.issuance_at, result):
            values = model.encode_teams(queries)
            values['fold_id'] = fold
            exports.append(values)
            z = values[[f'z_{i}' for i in range(model.config.embedding_dim)]].to_numpy()
            if not np.isfinite(z).all():
                raise ValueError('Retained embeddings contain nonfinite values.')
            summary = dict(fold_id=fold, mode=self.mode, teams=len(z), issuance_at=queries.issuance_at.iloc[0],
                           model_version=model.backend_.model_version)
            if len(z) < 2 or np.all(z == z[0]):
                result.notes.append(f'Fold {fold}: fewer than two distinct representations; projection is unavailable. Raw embeddings remain exported.')
                summaries.append({**summary, 'status': 'degenerate'})
                continue
            if self.mode == 'PCA':
                from sklearn.decomposition import PCA
                count = min(2, z.shape[0], z.shape[1])
                projection = PCA(n_components=count, whiten=self.whiten, svd_solver='full')
                xy = projection.fit_transform(z)
                xy = np.pad(xy, ((0, 0), (0, 2-count)))
                summary.update(explained_variance=projection.explained_variance_ratio_.tolist())
            else:
                from sklearn.manifold import TSNE
                if self.init not in ('pca', 'random'):
                    raise ValueError('tSNE init must be pca or random.')
                effective_init = 'random' if z.shape[1] < 2 else self.init
                perplexity = min(self.perplexity, len(z)-1.)
                projection = TSNE(n_components=2, perplexity=perplexity, learning_rate=self.learning_rate,
                    max_iter=self.max_iter, init=effective_init, metric=self.metric,
                    early_exaggeration=self.early_exaggeration, random_state=self.random_state)
                xy = projection.fit_transform(z)
                summary.update(effective_perplexity=perplexity, effective_init=effective_init,
                               kl_divergence=float(projection.kl_divergence_))
            if not np.isfinite(xy).all():
                raise ValueError('Embedding projection produced nonfinite coordinates.')
            summaries.append({**summary, 'status': 'ok'})
            table = values[['team_id', 'league', 'issuance_at', 'history_cutoff', 'support', 'gate', 'model_version', 'fold_id']].copy()
            table['x'], table['y'] = xy.T
            names, badges = [], []
            for team in table.team_id:
                name, badge, note = catalog.display(team, badges=self.show_badges)
                names.append(name); badges.append(badge)
                if note: result.notes.append(note)
            table['team_name'] = names
            projections.append(table)
            figure = embedding_figure(table, badges,
                f'{self.mode} · Fold {fold}<br>{queries.issuance_at.iloc[0].tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")}')
            result.artifacts.append(Artifact('plotly', figure, f'{self.mode} embeddings · fold {fold}', options={'embedding_fold': str(fold)}))
            support = go.Figure(go.Bar(x=[escape(x) for x in names], y=table.support, marker_color='#2aa198'))
            result.artifacts.append(Artifact('plotly', _style(support, f'Historical support · Fold {fold}', 'Team', 'Decayed match mass'), 'Historical support', options={'embedding_fold': str(fold)}))
        result.tables['embeddings'] = pd.concat(exports, ignore_index=True) if exports else pd.DataFrame()
        result.tables['projection'] = pd.concat(projections, ignore_index=True) if projections else pd.DataFrame()
        result.tables['projection_summary'] = pd.DataFrame(summaries)
        return result


@dataclass(kw_only=True)
class GraphAttentionWeightReporter(PredictionReporter):
    """Incoming edge weights by receiver, layer and head at an exact cutoff.

    Select teams in the rendered plot. Top edges only affect the chart: the full
    attention table and concentration summary remain exportable. Weights sum
    to one over each receiver's full incoming history in each layer/head.
    """
    issuance_at: str | None = None
    top_edges: int = 20
    show_badges: bool = True
    catalog: object = None
    supported_types: ClassVar[tuple] = ('per_fold', 'overall')

    def run(self, context):
        _positive('top_edges', self.top_edges)
        result = StudyResult('Graph attention weights', notes=[
            'Attention is normalized within each receiver, layer and head, including temporal decay. It is not a causal effect or a direct attribution of the fixture prediction.',
            'The chart shows the strongest incoming edges per receiver/layer/head. Exports retain every edge and the full-history entropy. Gates and historical support remain separate.',
        ])
        raw_catalog = self.catalog if self.catalog is not None else context.resources.get('team_catalog')
        catalog = raw_catalog if isinstance(raw_catalog, TeamCatalog) else TeamCatalog(raw_catalog or {})
        exports, summaries = [], []
        go = _plotting()
        for fold, model, queries in _populations(context, self.issuance_at, result):
            table = model.attention_weights(queries)
            if table.empty:
                result.notes.append(f'Fold {fold}: no eligible historical edges; representations use priors.')
                continue
            table['fold_id'] = fold
            exports.append(table)
            figure = go.Figure(); trace_teams = []; buttons = []; images = {}
            for (team, layer, head), group in table.groupby(['team_id', 'layer', 'head'], sort=False):
                p = group.attention.to_numpy(); nonzero = p[p > 0]
                entropy = float(-(nonzero*np.log(nonzero)).sum())
                summaries.append(dict(fold_id=fold, team_id=team, layer=layer, head=head,
                    edges=len(group), weight_sum=float(p.sum()), entropy=entropy,
                    effective_edges=float(np.exp(entropy)), support=group.support.iloc[0], gate=group.gate.iloc[0]))
                name, badge, note = catalog.display(team, badges=self.show_badges)
                if note: result.notes.append(note)
                images[team] = [] if badge is None else [dict(source=badge, x=1, y=1.12, xref='paper', yref='paper', sizex=.06, sizey=.12, xanchor='right', yanchor='top')]
                top = group.nlargest(self.top_edges, 'attention')
                labels = [escape(catalog.display(r.opponent_id)[0])+' · '+r.kickoff_at.strftime('%Y-%m-%d')+' · '+str(r.event_id) for r in top.itertuples()]
                figure.add_trace(go.Bar(x=labels, y=top.attention, name=f'Layer {layer+1} / head {head+1}',
                    visible=not trace_teams or team == trace_teams[0]))
                trace_teams.append(team)
            for team in dict.fromkeys(trace_teams):
                name = escape(catalog.display(team)[0])
                buttons.append(dict(label=name, method='update', args=[{'visible':[t == team for t in trace_teams]},
                    {'title.text':f'Incoming attention · Fold {fold}', 'images':images[team]}]))
            _style(figure, f'Incoming attention · Fold {fold}', 'Historical opponent / kickoff / fixture', 'Normalized weight')
            figure.update_layout(height=750, title=dict(y=.98, yanchor='top'), colorway=['#268bd2','#dc322f','#238b57','#b58900','#805ad5','#2aa198'],
                margin=dict(t=185, b=150),
                legend=dict(orientation='h', x=0, y=1.05, yanchor='bottom'),
                xaxis=dict(tickangle=-40, automargin=True),
                updatemenus=[dict(buttons=buttons, direction='down', x=0, y=1.28, yanchor='bottom')], images=images[trace_teams[0]])
            result.artifacts.append(Artifact('plotly', figure, f'Attention · fold {fold}'))
        result.tables['attention'] = pd.concat(exports, ignore_index=True) if exports else pd.DataFrame()
        result.tables['concentration'] = pd.DataFrame(summaries)
        return result
