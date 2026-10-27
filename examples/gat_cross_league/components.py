"""Exact disconnected-component restriction for this frozen experiment only."""
import math
import numpy as np
import pandas as pd
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.store import GraphSnapshot


class ComponentStore(TemporalEventStore):
    """All source history remains available; skip unrelated graph components.

    Components use the entire fixed history, including later edges. This can
    retain more nodes than strictly necessary at a cutoff, but cannot omit a
    currently connected node. Component identity never enters model inputs.
    Global category mappings and the native regularizer remain unchanged.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        parent=np.arange(len(self._source_nodes))
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        for a,b in zip(self._receiver_codes,self._sender_codes):
            ra,rb=root(a),root(b)
            if ra!=rb:parent[rb]=ra
        self._component_for_node=np.array([root(i) for i in range(len(parent))])
        self._component_for_row=self._component_for_node[self._receiver_codes]

    def snapshot(self,issuance,config,teams=(),*,_columns=None,_since=None):
        teams=tuple(teams)
        if not teams:
            return super().snapshot(issuance,config,teams,_columns=_columns,_since=_since)
        if config.temporal_mode=='historical' or config.boundary_memory is not None:
            raise ValueError('This experiment optimization supports snapshot/time_features only')
        issuance=pd.Timestamp(issuance).tz_convert('UTC');cutoff=config.history_cutoff(issuance)
        components={self._component_for_node[self._source_node_index[t]] for t in teams if t in self._source_node_index}
        rows=self._eligible_rows(cutoff)
        rows=rows[np.isin(self._component_for_row[rows],list(components))]
        if _since is not None:
            old=pd.Timestamp(_since).value
            rows=rows[(self._kickoff_ns[rows]>=old)|(self._available_ns[rows]>old)]
        source=self._history if _columns is None else self._history.loc[:,_columns]
        edges=source.iloc[rows].copy()
        codes=self._receiver_codes[rows]
        nodes=tuple(sorted({self._source_nodes[i] for i in np.unique(codes)}|set(teams)))
        index={t:i for i,t in enumerate(nodes)}
        local=np.full(len(self._source_nodes),-1,dtype=np.int64)
        for team,position in index.items():
            code=self._source_node_index.get(team)
            if code is not None:local[code]=position
        receiver=local[codes];sender=local[self._sender_codes[rows]]
        counts=np.zeros(len(nodes),dtype=np.int64);leagues=[];entries=[]
        for i,team in enumerate(nodes):
            member=self._membership_by_team.get(team)
            n=np.searchsorted(member[0],issuance.value,side='right') if member is not None else 0
            counts[i]=n;leagues.append(member[1][n-1] if n else None);entries.append(member[2][n-1] if n else 'unknown')
        age=(issuance-edges.kickoff_at).dt.total_seconds().to_numpy()/86400.
        crossed=(counts[receiver]-self._entries_at_event[rows]).astype(float)
        log_weight=-math.log(2)*age/config.half_life_days+crossed*math.log(config.seasonal_retention)
        support=np.bincount(receiver,weights=np.exp(log_weight),minlength=len(nodes))
        return GraphSnapshot(issuance,cutoff,nodes,edges,receiver,sender,log_weight,support,
            support/(support+config.support_kappa),tuple(leagues),tuple(entries),self.version)
