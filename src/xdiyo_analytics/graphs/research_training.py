"""Opt-in retained validation reduction and replay inputs on the native loop."""
from dataclasses import dataclass, asdict
from pathlib import Path
from hashlib import sha256
import io
import json
import time
import numpy as np
from .adapter import _Backend, _torch
from .execution import configured_execution
from .goals import poisson_goal_loss

COMPETITIONS = (8,17,18,23,34,35,36,37,38,44,53,54,182)
MONITOR = 'all13_inner_raw_hda_ce_micro'


@dataclass(frozen=True)
class RetainedGATTraining:
    """Explicit population contract. Replay paths require hashes of their bytes.

    A miniature declared competition set is useful for tests, but does not
    represent an all13 reproduction. Paths locate inputs; hashes seal them.
    """
    contract: str = 'retained_all13_raw_ce_v1'
    competitions: tuple[int, ...] = COMPETITIONS
    preprocessing_path: str | None = None
    preprocessing_sha256: str | None = None
    initial_state_path: str | None = None
    initial_state_sha256: str | None = None
    event_store_version: str | None = None
    fitting_population_path: str | None = None
    fitting_population_sha256: str | None = None
    validation_population_path: str | None = None
    validation_population_sha256: str | None = None

    def __post_init__(self):
        if self.contract != 'retained_all13_raw_ce_v1':
            raise ValueError('Unsupported retained training contract.')
        values = tuple(self.competitions)
        if not values or any(type(v) is not int for v in values) or len(set(values)) != len(values):
            raise ValueError('Declare a nonempty ordered set of integer competitions.')
        object.__setattr__(self, 'competitions', values)
        fields = (self.preprocessing_path, self.preprocessing_sha256,
                  self.initial_state_path, self.initial_state_sha256)
        if any(v is not None for v in fields) and not all(v is not None for v in fields):
            raise ValueError('Historical replay requires both preprocessing and initial tensors with SHA256 hashes.')
        population = (self.fitting_population_path,self.fitting_population_sha256,
                      self.validation_population_path,self.validation_population_sha256)
        if any(v is not None for v in population) and not all(v is not None for v in population):
            raise ValueError('Population replay requires both fitting and validation paths and hashes.')
        for value in (self.preprocessing_sha256, self.initial_state_sha256,
                      self.event_store_version,self.fitting_population_sha256,self.validation_population_sha256):
            if value is not None and (not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Replay inputs require lowercase SHA256 hashes.')
        for value in (self.preprocessing_path, self.initial_state_path,self.fitting_population_path,self.validation_population_path):
            if value is not None and (not isinstance(value,str) or not value.strip()):
                raise ValueError('Replay paths must be nonempty strings.')

    def identity(self):
        result=asdict(self)
        result["competitions"]=list(self.competitions)
        for name in ('preprocessing_path','initial_state_path','fitting_population_path','validation_population_path'):result.pop(name)
        return result

    def inputs(self):
        if self.preprocessing_path is None:
            return None, None
        values=[]
        for name,digest in ((self.preprocessing_path,self.preprocessing_sha256),
                            (self.initial_state_path,self.initial_state_sha256)):
            data=Path(name).read_bytes()
            if sha256(data).hexdigest()!=digest:
                raise ValueError('Historical replay input hash changed: '+name)
            values.append(data)
        state=json.loads(values[0])
        if set(state)=={'state'} or 'state' in state:
            state=state['state']
        return state, _torch().load(io.BytesIO(values[1]),map_location='cpu',weights_only=True)

    def validate_population(self, store, context):
        if self.event_store_version is not None and store.version != self.event_store_version:
            raise ValueError('Historical replay requires the exact archived event store.')
        if self.fitting_population_path is None:
            return
        import pandas as pd
        for inputs,path,digest in ((context,self.fitting_population_path,self.fitting_population_sha256),
                                  (context.validation,self.validation_population_path,self.validation_population_sha256)):
            data=Path(path).read_bytes()
            if sha256(data).hexdigest()!=digest:
                raise ValueError('Historical population input hash changed: '+path)
            expected=pd.read_parquet(io.BytesIO(data))
            if inputs is None:
                raise ValueError('Historical replay requires its exact fitting and validation populations.')
            try:
                pd.testing.assert_frame_equal(inputs.metadata,expected,check_exact=True)
            except AssertionError as error:
                raise ValueError('Historical replay population/order differs from the declared archive.') from error
            truth=np.where(expected.goals_for>expected.goals_against,1,
                           np.where(expected.goals_for<expected.goals_against,-1,0))
            if inputs.y.shape[1]!=1 or not np.array_equal(inputs.y.iloc[:,0],truth):
                raise ValueError('Historical replay labels differ from the declared fixture outcomes.')

    def validate_control(self, control):
        if control.monitor != MONITOR or control.direction != 'minimize' or not control.restore_best:
            raise ValueError('Retained training requires raw validation CE selection, minimize and restore_best.')


def summarize(losses, leagues):
    losses=np.asarray(losses,float);leagues=np.asarray(leagues)
    if losses.shape!=leagues.shape or not len(losses) or not np.isfinite(losses).all():
        raise ValueError('Invalid validation population')
    per_league={str(k):float(losses[leagues==k].mean()) for k in np.unique(leagues)}
    return dict(micro=float(losses.mean()),macro=float(np.mean(list(per_league.values()))),per_league=per_league,count=len(losses))


def rps(probability, truth):
    p=np.asarray(probability,dtype=float);y=np.asarray(truth,dtype=int)
    if p.shape!=(len(y),3) or not np.isfinite(p).all() or (p<0).any() or (y<0).any() or (y>2).any():
        raise ValueError('RPS requires aligned finite H/D/A probabilities and class indices')
    if not np.allclose(p.sum(1),1,atol=3e-7,rtol=0):
        raise ValueError('Outcome probabilities must sum to one.')
    p=p/p.sum(1,keepdims=True)
    return .5*np.square(np.cumsum(p,axis=1)[:,:2]-(y[:,None]<=np.arange(2))).sum(1)


def digest_state(value):
    h=sha256()
    def add(v):
        if isinstance(v,(_torch().Tensor,np.ndarray)):
            a=v.detach().cpu().numpy() if isinstance(v,_torch().Tensor) else v
            h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()


class RetainedGATBackend(_Backend):
    def __init__(self, config, store, seed, device, input_cache_policy=None, *, training):
        if not isinstance(training,RetainedGATTraining):
            raise TypeError('training must be RetainedGATTraining.')
        if config.goal_auxiliary is None or config.goal_auxiliary.architecture != 'linear':
            raise ValueError('Retained raw validation requires the materialized linear goal head.')
        self.research_training=training
        preprocessing, self._initial_state = training.inputs()
        super().__init__(config,store,seed,device,input_cache_policy,preprocessor_state=preprocessing)
        self.monitor_name=MONITOR
        self._required_leagues=training.competitions
        self._validation_leagues=None
        self._validation_forward_calls=0

    def initialize(self,context):
        if context.validation is None:
            raise ValueError('Retained training requires a declared validation population.')
        self.research_training.validate_population(self.store,context)
        leagues=context.validation.metadata.competition_id.to_numpy(dtype=np.int64)
        if set(leagues)!=set(self._required_leagues):
            raise ValueError('Validation must contain exactly the declared competitions.')
        weights=context.validation.sample_weight
        if weights is not None:
            weights=np.asarray(weights,dtype=float).reshape(-1)
            if not len(weights) or not np.isfinite(weights).all() or weights[0]<=0 or not np.equal(weights,weights[0]).all():
                raise ValueError('This monitor requires equal per-fixture validation weight')
        super().initialize(context)
        if self._initial_state is not None:
            self.model.load_state_dict(self._initial_state)
        self.initial_digest=digest_state(self.model.state_dict())
        self._validation_leagues=leagues
        self.refresh_version()

    def index(self,groups,index,positions):
        return _torch().as_tensor(positions,device=self.device)

    @configured_execution
    def _loss(self,groups,targets,weights,backward=False):
        if backward or groups is not self.validation_groups:
            return super()._loss(groups,targets,weights,backward)
        if not _torch().equal(weights,weights[0].expand_as(weights)):
            raise ValueError('This monitor requires equal per-fixture validation weight')
        count=len(targets);losses=np.empty(count);goal_losses=np.empty(count)
        probability=np.empty((count,3));rates=np.empty((count,2));seen=np.zeros(count,dtype=np.int64)
        # The caller's no_grad context is also enforced here for diagnostics.
        with _torch().no_grad():
            for index in range(len(groups)):
                with self._group_lease(groups,index) as (group,_):
                    positions,data,home,away,fixture=group
                    ix=self.index(groups,index,positions)
                    logits,goal_rates=self.model(data,home,away,fixture,return_goals=True)
                    losses[positions]=_torch().nn.functional.cross_entropy(logits,targets[ix],reduction='none').cpu().numpy()
                    goal_losses[positions]=poisson_goal_loss(goal_rates,self.validation_goals[ix]).cpu().numpy()
                    probability[positions]=_torch().softmax(logits,dim=1).cpu().numpy();rates[positions]=goal_rates.cpu().numpy()
                    seen[positions]+=1;self._validation_forward_calls+=1
        if not (seen==1).all():
            raise ValueError('Validation fixtures must be scored exactly once')
        self._inner_summary=summarize(losses,self._validation_leagues)
        self._inner_summary['rps']=float(rps(probability,targets.cpu().numpy()).mean())
        self._inner_probability=probability;self._inner_rates=rates
        self._goal_loss=float(goal_losses.mean())
        return self._inner_summary['micro']

    def step(self):
        self._update_started=time.perf_counter()
        return super().step()

    def checkpoint_metrics(self,monitor,values):
        if monitor!=MONITOR:raise ValueError('Only the declared all-league monitor may select this experiment')
        values=super().checkpoint_metrics('train_loss',values)
        values[MONITOR]=values['validation_loss']
        assert values[MONITOR]==self._inner_summary['micro']
        values['all13_inner_raw_hda_ce_macro']=self._inner_summary['macro']
        values['all13_inner_raw_rps_micro']=self._inner_summary['rps']
        values.update({f'inner_raw_ce_league_{league}':value for league,value in self._inner_summary['per_league'].items()})
        if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite metric; no clipping')
        if str(self.device).startswith('cuda'):_torch().cuda.synchronize(self.device)
        values['update_seconds']=time.perf_counter()-self._update_started
        with _torch().no_grad():
            values['league_prior_norm']=float(self.model.league.weight.norm())
            values['entry_prior_norm']=float(self.model.entry.weight.norm())
            values['parameter_norm']=float(_torch().sqrt(sum(v.square().sum() for v in self.model.parameters())))
        return values
