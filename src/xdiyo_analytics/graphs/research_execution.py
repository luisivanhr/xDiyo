"""Explicit acceleration for exact retained families; never selected on load."""
from contextlib import contextmanager
from dataclasses import dataclass, fields
import time
import weakref
import numpy as np
from .adapter import _torch
from .execution import configured_execution
from .research_training import RetainedGATBackend, MONITOR, summarize, rps
from .research_profiles import profile_key
from .streaming import PreparedGroups
from .goals import poisson_goal_loss
from ..training.residency import InputCachePolicy, InputResident


@dataclass(frozen=True)
class RetainedGATExecution:
    profile: str = 'retained_gat_speed_v1'
    omit_diagnostic_replay: bool = True
    pack_validation: bool = True
    cache_indices: bool = True
    borrow_inputs: bool = True
    omit_unused: bool = True

    def __post_init__(self):
        if self.profile!='retained_gat_speed_v1':raise ValueError('Unsupported retained execution profile.')
        if any(type(getattr(self,f.name)) is not bool for f in fields(self) if f.name!='profile'):
            raise ValueError('Each retained speed flag must be an explicit boolean.')

    def validate_config(self,config):
        if profile_key(config) is None:
            raise ValueError('Retained acceleration requires an exact supported V1 research configuration.')


class AcceleratedRetainedGATBackend(RetainedGATBackend):
    def __init__(self,*args,execution,**kwargs):
        if not isinstance(execution,RetainedGATExecution):raise TypeError('research_execution must be RetainedGATExecution.')
        execution.validate_config(args[0] if args else kwargs['config'])
        self.research_execution=execution;self._device_positions=False
        self._index_owners=weakref.WeakKeyDictionary();self._active_indices={}
        self.borrow=None;self.audit_borrow=False
        super().__init__(*args,**kwargs)
        self.input_cache_policy=self.input_cache_policy or InputCachePolicy()

    def _make_network(self):
        if not self.research_execution.omit_unused:return super()._make_network()
        from .research_network import RetainedLeanNetwork
        return RetainedLeanNetwork(self.config,len(self.preprocessor.leagues),
                                   self.preprocessor.edge_dim,self.preprocessor.fixture_dim)

    def initialize(self,context):
        policy=self.input_cache_policy
        if self.research_execution.borrow_inputs:
            # This backend alone owns residency; native public leases stream.
            self.input_cache_policy=InputCachePolicy(0,policy.headroom_bytes)
        try:super().initialize(context)
        finally:self.input_cache_policy=policy
        from .research_cache import freeze_groups
        for name in ('train_groups','validation_groups'):
            freeze_groups(getattr(self,name))
        if self.preprocessor.config!=self.config:raise ValueError('Retained acceleration requires matching preprocessing.')

    def _borrow_cache(self):
        if self.borrow is None or self.borrow.closed:
            from .research_cache import BorrowCache
            self.borrow=BorrowCache(self.device,self.input_cache_policy,audit=self.audit_borrow)
        return self.borrow

    @contextmanager
    def _positions_lease(self,groups,index,positions):
        torch=_torch()
        if not self.research_execution.cache_indices:
            yield torch.as_tensor(positions,device=self.device);return
        owner=self._index_owners.get(groups)
        if owner is None:
            owner=InputResident(self.device,self.input_cache_policy);self._index_owners[groups]=owner
        from .research_cache import owner_key
        key=owner_key(groups,index)[1:]
        def cpu():return torch.as_tensor(np.array(positions,copy=True),dtype=torch.long)
        # InputResident joins the same native pool as public graph leases.
        with owner.lease(key,cpu,lambda:cpu().to(self.device)) as ix:yield ix

    def index(self,groups,index,positions):
        return self._active_indices.get((id(groups),index),None) if (id(groups),index) in self._active_indices else _torch().as_tensor(positions,device=self.device)

    @contextmanager
    def _indexed_group(self,groups,index,group,owned,ix):
        key=(id(groups),index);previous=self._active_indices.get(key)
        self._active_indices[key]=ix
        try:yield ((ix,*group[1:]) if self._device_positions else group),owned
        finally:
            if previous is None:self._active_indices.pop(key,None)
            else:self._active_indices[key]=previous

    @contextmanager
    def _group_lease(self,groups,index,replay=None):
        changes=self.research_execution
        if changes.borrow_inputs and type(groups) is PreparedGroups and replay is None:
            with self._borrow_cache().lease(groups,index,cache_indices=changes.cache_indices) as (group,ix):
                with self._indexed_group(groups,index,group,groups.reconstruction_targets,ix) as value:yield value
        else:
            with super()._group_lease(groups,index,replay) as (group,owned):
                if changes.cache_indices and type(groups) is PreparedGroups and replay is None:
                    with self._positions_lease(groups,index,group[0]) as ix:
                        with self._indexed_group(groups,index,group,owned,ix) as value:yield value
                else:yield group,owned

    @configured_execution
    def _loss(self,groups,targets,weights,backward=False):
        torch=_torch();changes=self.research_execution
        if backward or groups is not self.validation_groups:
            old=self._device_positions;self._device_positions=changes.cache_indices or changes.borrow_inputs
            try:return super()._loss(groups,targets,weights,backward)
            finally:self._device_positions=old
        if not changes.pack_validation:return super()._loss(groups,targets,weights,backward)
        if not torch.equal(weights,weights[0].expand_as(weights)):raise ValueError('Equal validation weights required')
        n=len(targets);losses=np.empty(n);goals=np.empty(n);probability=np.empty((n,3));rates=np.empty((n,2));seen=np.zeros(n,np.int64)
        with torch.no_grad():
            for i in range(len(groups)):
                with self._group_lease(groups,i) as (group,_):
                    positions,data,home,away,fixture=group;ix=self.index(groups,i,positions)
                    logits,goal_rates=self.model(data,home,away,fixture,return_goals=True)
                    ce=torch.nn.functional.cross_entropy(logits,targets[ix],reduction='none')
                    goal=poisson_goal_loss(goal_rates,self.validation_goals[ix]);p=torch.softmax(logits,dim=1)
                    packed=torch.cat((ce[:,None],goal[:,None],p,goal_rates),dim=1).cpu().numpy()
                    losses[positions]=packed[:,0];goals[positions]=packed[:,1];probability[positions]=packed[:,2:5];rates[positions]=packed[:,5:]
                    seen[positions]+=1;self._validation_forward_calls+=1
        if not (seen==1).all():raise ValueError('Validation fixtures must be scored exactly once')
        self._inner_summary=summarize(losses,self._validation_leagues)
        self._inner_summary['rps']=float(rps(probability,targets.cpu().numpy()).mean())
        self._inner_probability=probability;self._inner_rates=rates;self._goal_loss=float(goals.mean())
        return self._inner_summary['micro']

    def checkpoint_metrics(self,monitor,values):
        if not self.research_execution.omit_diagnostic_replay:return super().checkpoint_metrics(monitor,values)
        if monitor!=MONITOR:raise ValueError('Retained acceleration requires raw validation CE selection.')
        values=dict(values)
        for name,new in [('train_loss','pre_update_train_ce'),('train_goal_loss','pre_update_train_goal_loss'),
                         ('train_regularization','pre_update_train_regularization'),('train_objective','pre_update_train_objective')]:
            values[new]=values.pop(name)
        values[MONITOR]=values['validation_loss']
        assert values[MONITOR]==self._inner_summary['micro']
        values['all13_inner_raw_hda_ce_macro']=self._inner_summary['macro'];values['all13_inner_raw_rps_micro']=self._inner_summary['rps']
        values.update({f'inner_raw_ce_league_{league}':v for league,v in self._inner_summary['per_league'].items()})
        if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite computed diagnostic')
        torch=_torch()
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        values['update_seconds']=time.perf_counter()-self._update_started
        with torch.no_grad():
            values['league_prior_norm']=float(self.model.league.weight.norm());values['entry_prior_norm']=float(self.model.entry.weight.norm())
            values['parameter_norm']=float(torch.sqrt(sum(v.square().sum() for v in self.model.parameters())))
        return values

    def close(self):
        if self.borrow is not None:self.borrow.close()
        for owner in self._index_owners.values():owner.close()
        self._index_owners.clear();self._active_indices.clear()
        for name in ('train_groups','validation_groups'):
            groups=getattr(self,name,None)
            if groups is not None:groups._resident.close()
