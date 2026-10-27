"""Native fitting with one all-league validation pass per optimizer update."""
from dataclasses import asdict
from pathlib import Path
import json,time
import numpy as np
import torch
from . import core,metrics
from temporal_team_gat import make_context
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.execution import configured_execution
from xdiyo_analytics.graphs.goals import poisson_goal_loss
from xdiyo_analytics.training.residency import InputCachePolicy
from compare_gat_goals import digest_state

def context_for(run,job):
    population=core.populations(run,job['fold'])
    context=make_context(population['fit'])
    context.validation=make_context(population['inner'])
    return context,population

class AllLeagueBackend(_Backend):
    def __init__(self,run,protocol,store,job):
        self.run,self.job=Path(run),job
        cfg=core.config(job['configuration'])
        assert json.dumps(asdict(cfg),sort_keys=True)==json.dumps(protocol['configurations'][job['configuration']],sort_keys=True)
        state=json.loads((self.run/'shared/preprocessing'/f'{job["fold"]}.json').read_text())['state']
        resources=protocol['resources']
        super().__init__(cfg,store,job['seed'],resources['device'],InputCachePolicy(resources['input_cache_bytes'],resources['headroom_bytes']),preprocessor_state=state)
        self.monitor_name=core.MONITOR
        self._required_leagues=tuple(protocol['competitions'])
        self._validation_leagues=None
        self._validation_forward_calls=0

    def initialize(self,context):
        super().initialize(context)
        self.model.load_state_dict(torch.load(core.initial_path(self.run,self.job),map_location=self.device,weights_only=True))
        self.initial_digest=digest_state(self.model.state_dict())
        self._validation_leagues=context.validation.metadata.competition_id.to_numpy(dtype=np.int64)
        assert set(self._validation_leagues)==set(self._required_leagues),'Validation is missing a declared league'
        self.refresh_version()

    @configured_execution
    def _loss(self,groups,targets,weights,backward=False):
        if backward or groups is not self.validation_groups:
            return super()._loss(groups,targets,weights,backward)
        if not torch.equal(weights,weights[0].expand_as(weights)):
            raise ValueError('This monitor requires equal per-fixture validation weight')
        count=len(targets);losses=np.empty(count);goal_losses=np.empty(count)
        probability=np.empty((count,3));rates=np.empty((count,2));seen=np.zeros(count,dtype=np.int64)
        # The caller's no_grad context is also enforced here for diagnostics.
        with torch.no_grad():
            for index in range(len(groups)):
                with self._group_lease(groups,index) as (group,_):
                    positions,data,home,away,fixture=group
                    ix=torch.as_tensor(positions,device=self.device)
                    logits,goal_rates=self.model(data,home,away,fixture,return_goals=True)
                    losses[positions]=torch.nn.functional.cross_entropy(logits,targets[ix],reduction='none').cpu().numpy()
                    goal_losses[positions]=poisson_goal_loss(goal_rates,self.validation_goals[ix]).cpu().numpy()
                    probability[positions]=torch.softmax(logits,dim=1).cpu().numpy();rates[positions]=goal_rates.cpu().numpy()
                    seen[positions]+=1;self._validation_forward_calls+=1
        assert (seen==1).all(),'Validation fixtures must be scored exactly once'
        self._inner_summary=metrics.summarize(losses,self._validation_leagues)
        self._inner_summary['rps']=float(metrics.rps(probability,targets.cpu().numpy()).mean())
        self._inner_probability=probability;self._inner_rates=rates
        self._goal_loss=float(goal_losses.mean())
        return self._inner_summary['micro']

    def step(self):
        self._update_started=time.perf_counter()
        return super().step()

    def checkpoint_metrics(self,monitor,values):
        if monitor!=core.MONITOR:raise ValueError('Only the declared all-league monitor may select this experiment')
        values=super().checkpoint_metrics('train_loss',values)
        values[core.MONITOR]=values['validation_loss']
        assert values[core.MONITOR]==self._inner_summary['micro']
        values['all13_inner_raw_hda_ce_macro']=self._inner_summary['macro']
        values['all13_inner_raw_rps_micro']=self._inner_summary['rps']
        values.update({f'inner_raw_ce_league_{league}':value for league,value in self._inner_summary['per_league'].items()})
        if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite metric; no clipping')
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        values['update_seconds']=time.perf_counter()-self._update_started
        return values
