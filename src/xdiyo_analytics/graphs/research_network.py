"""Exact retained forward path with unused auxiliary computations omitted."""
import torch
from .network import TemporalNetwork


class RetainedLeanNetwork(TemporalNetwork):
    # __init__ is inherited unchanged; registration and random draws are fixed.
    def encode(self,data,*,return_attention=False,return_auxiliary=False,_validated_targets=False):
        h=self.prior(data['league'],data['entry']);attention=[]
        for layer in self.layers:
            value=layer(h,data['x'],data['receiver'],data['sender'],data['log_weight'],data['gate'],return_attention=return_attention)
            if return_attention:h,alpha=value;attention.append(alpha)
            else:h=value
        if return_attention:return h,attention
        return (h,h.sum()*0) if return_auxiliary else h

    def forward(self,data,home,away,fixture,*,return_auxiliary=False,return_statistics=False,
                return_goals=False,return_goal_details=False,_validated_targets=False):
        result=self.encode(data,return_auxiliary=return_auxiliary,_validated_targets=_validated_targets)
        h,auxiliary=result if return_auxiliary else (result,None)
        a,b=h[home],h[away];inputs=torch.cat((a,b,a-b,a*b,fixture),dim=1)
        rates=details=None
        if return_goals or return_goal_details:
            goal_inputs=inputs.detach() if self.goal_config.detach_encoder else inputs
            rates=torch.nn.functional.softplus(self.goal_head(goal_inputs))+self.goal_config.rate_floor
            if return_goal_details:details=dict(rates=rates,log_rates=rates.log())
        logits=self.head(inputs)
        if return_goal_details:return logits,details
        if return_goals:return logits,rates
        if return_statistics:return logits,auxiliary,None
        return (logits,auxiliary) if return_auxiliary else logits
