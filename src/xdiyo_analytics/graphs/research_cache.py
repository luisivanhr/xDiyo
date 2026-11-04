"""Private read-only graph leases with one backend budget and weak owners."""
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy, deepcopy
from types import MappingProxyType
import weakref
import numpy as np
import torch
from .streaming import prepare_group, PreparedGroups, isolated_frame
from ..training.residency import tensor_bytes


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(dict,MappingProxyType)):
        for item in value.values():yield from tensors(item)
    elif isinstance(value,(tuple,list)):
        for item in value:yield from tensors(item)


def immutable(value,device):
    if isinstance(value,torch.Tensor):return value.to(device,copy=True)
    if isinstance(value,dict):return MappingProxyType({k:immutable(v,device) for k,v in value.items()})
    if isinstance(value,(tuple,list)):return tuple(immutable(v,device) for v in value)
    if isinstance(value,np.ndarray):
        result=value.copy();result.setflags(write=False);return result
    return value


def freeze_groups(groups):
    """Snapshot only input ownership, never a learned tensor or boundary carry."""
    if type(groups) is not PreparedGroups:
        raise TypeError('Borrowing requires exact native PreparedGroups.')
    if not hasattr(groups,'_fixed_plan'):
        plan=copy(groups.plan)
        plan.metadata=isolated_frame(plan.metadata);plan.X=isolated_frame(plan.X)
        plan.descriptors=tuple((t,p.copy()) for t,p in plan.descriptors)
        groups._fixed_plan=plan
        groups._fixed_preprocessor=type(groups.preprocessor).from_state(
            deepcopy(groups.preprocessor.config),deepcopy(groups.preprocessor.state()))


def owner_key(owner,index):
    return (weakref.ref(owner),owner._fixed_plan.store.version,owner._time_encoding_key,
            owner._fixture_encoding_key,owner._phase_encoding_key,owner.reconstruction_targets,index)


class BorrowCache:
    """One fit/validation input owner; indices and live transfers are counted.

    max_bytes bounds retained storage. A single required streaming group can
    exceed it; its bytes remain visible as transient and disappear on exit.
    Active leases stay pinned when their owner or cache closes.
    """
    def __init__(self,device,policy,*,audit=False):
        self.device=str(device);self.policy=policy;self.audit=audit
        self.entries=OrderedDict();self.closed_owners=weakref.WeakSet()
        self.resident=self.transient=self.peak=self.hits=self.misses=self.evictions=0
        self.closed=False

    def _drop(self,key):
        entry=self.entries.pop(key);self.resident-=entry['bytes'];self.evictions+=1

    def _trim(self,needed):
        for key,entry in list(self.entries.items()):
            if not entry['pins'] and (key[0]() is None or entry['retired'] or
                    self.resident+self.transient+needed>self.policy.max_bytes):self._drop(key)

    def _room(self,size):
        self._trim(size)
        free=torch.cuda.mem_get_info(self.device)[0] if self.device.startswith('cuda') else float('inf')
        if free<self.policy.headroom_bytes+size:
            self._trim(self.policy.max_bytes)
            free=torch.cuda.mem_get_info(self.device)[0]
        return self.resident+self.transient+size<=self.policy.max_bytes and free>=self.policy.headroom_bytes+size

    @contextmanager
    def lease(self,owner,index,*,cache_indices=False):
        if self.closed or owner in self.closed_owners:raise RuntimeError('Borrow cache or input owner closed')
        freeze_groups(owner)
        key=(*owner_key(owner,index),cache_indices);self._trim(0)
        entry=self.entries.get(key);temporary=extra=0
        if entry is None:
            self.misses+=1;plan,pre=owner._fixed_plan,owner._fixed_preprocessor
            cpu=prepare_group(plan.raw(index,columns=pre.snapshot_columns(plan.store)),pre,'cpu',plan.store,
                              reconstruction_targets=owner.reconstruction_targets)
            ix=torch.as_tensor(np.array(cpu[0],copy=True),dtype=torch.long) if cache_indices else None
            size=tensor_bytes((cpu,ix));cacheable=self._room(size)
            value=immutable((cpu,ix),self.device);del cpu,ix
            entry=dict(value=value,bytes=size,pins=0,retired=False)
            if cacheable:self.entries[key]=entry;self.resident+=size
            else:temporary=size;self.transient+=size
        else:self.hits+=1;self.entries.move_to_end(key)
        entry['pins']+=1;group,ix=entry['value']
        versions=[]
        try:
            if ix is None:
                needed=len(group[0])*8;self._room(needed)
                extra=needed;self.transient+=extra
                ix=torch.as_tensor(np.array(group[0],copy=True),dtype=torch.long,device=self.device)
            self.peak=max(self.peak,self.resident+self.transient)
            versions=[(t,t._version) for t in tensors((group,ix))] if self.audit else []
            yield group,ix
        finally:
            changed=any(t._version!=version for t,version in versions)
            entry['pins']-=1;self.transient-=temporary+extra
            if changed:entry['retired']=True
            if not temporary and entry['retired'] and not entry['pins']:self._drop(key)
            if changed:raise RuntimeError('Audited consumer mutated borrowed static tensors')

    def close_owner(self,owner):
        self.closed_owners.add(owner)
        for key,entry in list(self.entries.items()):
            if key[0]() is owner:
                if entry['pins']:entry['retired']=True
                else:self._drop(key)

    def close(self):
        self.closed=True
        for key,entry in list(self.entries.items()):
            if entry['pins']:entry['retired']=True
            else:self._drop(key)

    def stats(self):
        return {k:getattr(self,k) for k in ('resident','transient','peak','hits','misses','evictions')}
