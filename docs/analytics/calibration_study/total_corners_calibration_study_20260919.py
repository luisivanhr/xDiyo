from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/corners_lasso_first_grid_search--fcd9789c/runs/bd20f431-2828-4e43-b0b4-026eeda650aa'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
run = json.loads((RUN / 'run.json').read_text(encoding='utf-8'))
training = json.loads((RUN / 'training.json').read_text(encoding='utf-8'))
frames = []
for fold_dir in sorted(RUN.glob('fold-*')):
    if (fold_dir / 'output-0.parquet').exists():
        part = pd.concat([pd.read_parquet(fold_dir / 'metadata.parquet'),
            pd.read_parquet(fold_dir / 'targets.parquet').iloc[:, 0].rename('observed'),
            pd.read_parquet(fold_dir / 'output-0.parquet').iloc[:, 0].rename('predicted')], axis=1)
        part['fold_id'] = int(fold_dir.name.split('-')[-1]); frames.append(part)
df = pd.concat(frames, ignore_index=True)

def rank_corr(a, b):
    return float(np.corrcoef(pd.Series(a).rank(method='average'), pd.Series(b).rank(method='average'))[0, 1])

def stats(g):
    y, p = g.observed.to_numpy(float), g.predicted.to_numpy(float)
    mse = float(np.mean((p-y)**2)); base = float(np.mean((y-y.mean())**2))
    slope = float(np.cov(p, y, ddof=1)[0,1] / np.var(p, ddof=1)); intercept = float(y.mean() - slope*p.mean())
    return {'n': len(g), 'observed_mean': y.mean(), 'observed_sd': y.std(ddof=1),
      'observed_q05': np.quantile(y,.05), 'observed_q25': np.quantile(y,.25), 'observed_median': np.median(y), 'observed_q75': np.quantile(y,.75), 'observed_q95': np.quantile(y,.95),
      'predicted_mean': p.mean(), 'predicted_sd': p.std(ddof=1), 'predicted_q05': np.quantile(p,.05), 'predicted_q25': np.quantile(p,.25), 'predicted_median': np.median(p), 'predicted_q75': np.quantile(p,.75), 'predicted_q95': np.quantile(p,.95),
      'mse': mse, 'mae': float(np.mean(abs(p-y))), 'rmse': np.sqrt(mse), 'r2_vs_mean_baseline': 1-mse/base, 'mean_baseline_mse': base, 'pearson': float(np.corrcoef(y,p)[0,1]), 'spearman': rank_corr(y,p), 'descriptive_slope_y_on_prediction': slope, 'descriptive_intercept': intercept}

metric_rows = [dict(scope='pooled', **stats(df))]
for fold_id, g in df.groupby('fold_id', sort=True): metric_rows.append(dict(scope=f'fold-{fold_id}', **stats(g)))
pd.DataFrame(metric_rows).to_csv(OUT/'metrics.csv', index=False)

s = df.sort_values(['predicted', 'event_id']).copy(); bins = []; start = 0
for q in range(10):
    end = round((q+1)*len(s)/10)
    if q < 9:
        while end < len(s) and s.iloc[end-1].predicted == s.iloc[end].predicted: end += 1
    if end <= start: continue
    z = s.iloc[start:end]; bins.append({'bin': len(bins)+1, 'prediction_min': z.predicted.min(), 'prediction_max': z.predicted.max(), 'mean_prediction': z.predicted.mean(), 'mean_observed': z.observed.mean(), 'observed_sd': z.observed.std(ddof=1), 'n': len(z), 'mean_observed_se_iid': z.observed.std(ddof=1)/np.sqrt(len(z))}); start = end
if start < len(s):
    z = s.iloc[start:]; old = bins[-1]; n0 = old['n']; n1 = len(z); old['prediction_max'] = z.predicted.max(); old['mean_prediction'] = (old['mean_prediction']*n0+z.predicted.sum())/(n0+n1); old['mean_observed'] = (old['mean_observed']*n0+z.observed.sum())/(n0+n1); old['n'] = n0+n1
pd.DataFrame(bins).to_csv(OUT/'prediction_deciles.csv', index=False)

all_bins = []
for fold_id, g in df.groupby('fold_id', sort=True):
    s = g.sort_values(['predicted', 'event_id']); start = 0; fb = []
    for q in range(10):
        end = round((q+1)*len(s)/10)
        if q < 9:
            while end < len(s) and s.iloc[end-1].predicted == s.iloc[end].predicted: end += 1
        if end <= start: continue
        z=s.iloc[start:end]; fb.append({'fold_id':fold_id,'bin':len(fb)+1,'prediction_min':z.predicted.min(),'prediction_max':z.predicted.max(),'mean_prediction':z.predicted.mean(),'mean_observed':z.observed.mean(),'observed_sd':z.observed.std(ddof=1),'n':len(z)}); start=end
    all_bins.extend(fb)
pd.DataFrame(all_bins).to_csv(OUT/'prediction_deciles_by_fold.csv', index=False)

def make_plot(g, path, title):
    W,H=1100,820; margin=(105,70,55,100); img=Image.new('RGB',(W,H),'white'); draw=ImageDraw.Draw(img); lo=min(g.predicted.min(),g.observed.min()); hi=max(g.predicted.max(),g.observed.max()); x0,y0=margin[0],margin[1]; x1,y1=W-margin[2],H-margin[3]
    px=lambda v: x0+(v-lo)/(hi-lo)*(x1-x0); py=lambda v: y1-(v-lo)/(hi-lo)*(y1-y0)
    draw.line((x0,y1,x1,y1),fill='black',width=2); draw.line((x0,y1,x0,y0),fill='black',width=2); draw.line((px(lo),py(lo),px(hi),py(hi)),fill='#555555',width=2)
    for xp,yp in zip(g.predicted,g.observed): draw.ellipse((px(xp)-2,py(yp)-2,px(xp)+2,py(yp)+2),fill='#3465a4')
    draw.text((x0+240,20),title,fill='black'); draw.text((x0+280,H-55),'Predicted total corners',fill='black'); draw.text((15,350),'Observed total corners',fill='black'); img.save(path)
make_plot(df, OUT/'observed_vs_predicted_identity.png', 'Total corners: observed vs predicted (pooled)')
for fold_id, g in df.groupby('fold_id', sort=True): make_plot(g, OUT/f'observed_vs_predicted_identity_fold_{fold_id}.png', f'Total corners: observed vs predicted (fold {fold_id})')

features = run.get('config',{}).get('preparation',{}).get('recipe',{}).get('features',{})
conv = [{'fold_id': x.get('fold_id'), 'n_iter': x.get('summary',{}).get('n_iter'), 'termination_reason': x.get('summary',{}).get('termination_reason'), 'n_features': len(x.get('feature_columns',[]))} for x in training]
ids = ['source_league','source_season','competition_id','season_id','event_id']
manifest = {'run_path': str(RUN), 'run_id': run['run_id'], 'created_at': run['created_at'], 'name': run['name'], 'n_rows': len(df), 'folds': sorted(df.fold_id.unique().tolist()), 'feature_definition_count': len(features), 'training_model_records': conv, 'selected_alpha_by_fold': {str(k): v['config']['alpha'] for k,v in run.get('config',{}).get('selection',{}).items()}, 'identifier_columns': ids, 'identifier_rows': int(df[ids].drop_duplicates().shape[0]), 'train_predictions_retained': False}
(OUT/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
