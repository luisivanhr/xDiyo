"""Apply the user's explicit provider-label choice to four held review rows."""
from pathlib import Path
import hashlib,json
import pandas as pd
P=Path(__file__).parent
m=pd.read_parquet(P/'v58_full_mapping_REVIEW_NOT_APPROVED.parquet')
c=pd.read_parquet(P/'v58_selected_round_mapping_REVIEW_NOT_APPROVED.parquet')
f=pd.read_parquet(P/'v58_field_lineage_REVIEW_ONLY.parquet')
q=pd.read_csv(P/'v58_held_794_queue_REVIEW.csv',low_memory=False)
e=pd.read_csv(P/'v58_four_sofascore_event_observations_REVIEW.csv')
chosen={73239:4,73217:7,73205:10,73186:13}
assert dict(zip(e.match_id,e.v58_sofascore_displayed_round))==chosen
assert e.match_id.is_unique
for df in (m,c):
 df['v59_selected_season_round']=df.v53_selected_season_round.astype('Int64')
 df['v59_selection_semantics']=pd.Series(pd.NA,index=df.index,dtype='string')
 df['v59_original_draw_round_verified']=pd.Series(pd.NA,index=df.index,dtype='boolean')
 for mid,val in chosen.items():
  mask=df.match_id.eq(mid); assert mask.sum()==1 and df.loc[mask,'v53_selected_season_round'].isna().all()
  assert df.loc[mask,'v58_sofascore_displayed_round'].item()==val
  df.loc[mask,'v59_selected_season_round']=val
  df.loc[mask,'v59_selection_semantics']='USER_SELECTED_SOFASCORE_DISPLAYED_ROUND_FALLBACK'
  df.loc[mask,'v59_original_draw_round_verified']=False
f['v59_selected_value']=f.v53_selected_value.copy()
f['v59_selected_numeric']=f.v53_selected_numeric.copy()
f['v59_selected_text']=f.v53_selected_text.copy()
f['v59_previous_candidate_value']=pd.Series(pd.NA,index=f.index,dtype='string')
f['v59_previous_candidate_origin']=pd.Series(pd.NA,index=f.index,dtype='string')
f['v59_decision_semantics']=pd.Series(pd.NA,index=f.index,dtype='string')
f['v59_original_draw_round_verified']=pd.Series(pd.NA,index=f.index,dtype='boolean')
f['v59_selected_user_fallback']=False
for mid,val in chosen.items():
 mask=f.match_id.eq(mid)&f.field.eq('season_round');assert mask.sum()==1
 assert f.loc[mask,'v53_selected_value'].isna().all() and f.loc[mask,'candidate_value'].isna().all()
 assert f.loc[mask,'v58_sofascore_displayed_round'].item()==val
 f.loc[mask,'v59_previous_candidate_value']=f.loc[mask,'candidate_value']
 f.loc[mask,'v59_previous_candidate_origin']=f.loc[mask,'candidate_origin']
 f.loc[mask,'candidate_value']=str(val)
 f.loc[mask,'candidate_origin']='v59_explicit_user_selected_sofascore_displayed_round_fallback'
 f.loc[mask,'candidate_present']=True
 f.loc[mask,'candidate_agrees_with_prior_lineage']=False
 f.loc[mask,'v59_selected_value']=str(val)
 f.loc[mask,'v59_selected_numeric']=val
 f.loc[mask,'v59_decision_semantics']='USER_SELECTED_SOFASCORE_DISPLAYED_ROUND_FALLBACK; NOT_VERIFIED_ORIGINAL_DRAW_ROUND'
 f.loc[mask,'v59_original_draw_round_verified']=False
 f.loc[mask,'v59_selected_user_fallback']=True
 assert f.loc[mask,'base_field_conflict_gate_status'].str.contains('HELD_AUSTRALIA_ORIGINAL_DRAW_CONFLICT_V46').all()
assert f.v59_selected_user_fallback.sum()==4
assert (f.candidate_value.eq(f.v59_selected_value).fillna(False)|(f.candidate_value.isna()&f.v59_selected_value.isna())).all()
assert len(c)==61488 and c.match_id.is_unique and len(f)==368928 and len(q)==794
assert not f.integration_approved.any() and not f.model_input_approved.any()
assert f.v47_selected_explicit_clear.sum()==3 and c.v45_series_game_number.notna().sum()==60
keys=['v59_selected_season_round','v44_selected_phase_round','v47_selected_phase_key','v44_selected_stage_label','v44_selected_knockout_label','v44_selected_leg_number']
t=c.melt(id_vars='match_id',value_vars=keys,var_name='field',value_name='selected')
for prefix in ['v44_selected_','v47_selected_','v59_selected_']:t.field=t.field.str.replace(prefix,'',regex=False)
t['selected']=t.selected.astype('string')
j=f[['match_id','field','v59_selected_value']].merge(t,on=['match_id','field'],validate='one_to_one')
assert len(j)==368928 and (j.v59_selected_value.eq(j.selected).fillna(False)|(j.v59_selected_value.isna()&j.selected.isna())).all()
for df,name in [(m,'v59_full_mapping_REVIEW_NOT_APPROVED.parquet'),(c,'v59_selected_round_mapping_REVIEW_NOT_APPROVED.parquet'),(f,'v59_field_lineage_REVIEW_ONLY.parquet')]:df.to_parquet(P/name,index=False)
c.to_csv(P/'v59_selected_round_mapping_REVIEW_NOT_APPROVED.csv',index=False)
q.to_csv(P/'v59_held_794_queue_REVIEW.csv',index=False)
out={'fixture_rows':len(c),'field_rows':len(f),'selected_ledger_mismatches':0,'user_selected_provider_fallbacks':chosen,'original_draw_verified_for_fallbacks':0,'integration_approvals':0,'model_input_approvals':0,'all_prior_gates_preserved_by_builder':True}
for name in ['v59_full_mapping_REVIEW_NOT_APPROVED.parquet','v59_selected_round_mapping_REVIEW_NOT_APPROVED.parquet','v59_selected_round_mapping_REVIEW_NOT_APPROVED.csv','v59_field_lineage_REVIEW_ONLY.parquet','v59_held_794_queue_REVIEW.csv']:
 out[name+'_sha256']=hashlib.sha256((P/name).read_bytes()).hexdigest()
(P/'v59_validation.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
