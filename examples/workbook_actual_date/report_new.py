"""Final-only native consolidated report adapter. Does not train or publish.

Generate only after analyze_results and completion of native diagnostic/ticket
exports. Rendering creates an unapproved candidate, never a release receipt.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from analyze_results import digest,require

LABELS={'original13':'Original 13 leagues','first_divisions':'Added first divisions','first_and_second_divisions':'Added first and second divisions','time_B/N':'B/N','time_C/E1':'C/E1','time_B/CD_DW':'B/CD_DW','all_tested':'All tested fixtures','original13_population':'Common original 13'}
COLS={'stage':'Training dataset','family':'Model','population':'Test population','ce':'Outcome CE','rps':'Normalized RPS','draw_ce':'Draw CE','brier':'Multiclass Brier','goal_rmse':'Goal RMSE (C/E1)','accuracy':'Accuracy (%)','seed':'Seed','fold':'Test fold','profit':'Profit (units)','stake':'Stake (units)','roi':'ROI (%)','win_rate':'Win rate (%)','daily_pnl_sharpe':'Daily unit-PnL Sharpe-like statistic','max_drawdown_units':'Maximum drawdown (units)'}

def display(d):
    d=d.copy()
    for c in ('stage','family','population'):
        if c in d:d[c]=d[c].replace(LABELS)
    for c in ('accuracy','roi','win_rate'):
        if c in d:d[c]*=100
    return d.rename(columns=COLS)

def report(base=ROOT/'results'):
    from xdiyo_analytics.reporting import AnalysisReport,StudyRun,StudyResult,Artifact
    import plotly.graph_objects as go
    base=Path(base);analysis=base/'analysis';receipt=json.loads((analysis/'ANALYSIS_COMPLETE.json').read_text())
    for name,h in receipt['files'].items():require(digest(analysis/name)==h,'Analysis artifact changed: '+name)
    manifest=json.loads((base/'ANALYSIS_MANIFEST.json').read_text())
    # Report cannot silently omit required native tickets/latent diagnostics.
    support=json.loads((base/'REPORT_SUPPORT.json').read_text())
    for section in ('tickets','diagnostics'):
        require(section in support,'Native '+section+' export absent')
        require((base/support[section]).is_file(),'Missing native '+section+' entry page')
    out=base/'report';out.mkdir(exist_ok=True);studies=[]
    def add(name,artifacts,notes=(),n=0):
        tables={a.title or name:a.data for a in artifacts if a.kind=='table'}
        studies.append(StudyRun(name=name,type='overall',partition='outer',layout='match',row_positions=np.arange(n),n_matches=n,result=StudyResult(name,artifacts=artifacts,tables=tables,notes=list(notes))))
    summary=pd.read_csv(analysis/'scores_seed_mean.csv')
    paired=pd.read_csv(analysis/'paired_intervals.csv');primary=paired.query('weeks==4')
    add('Overview',[Artifact('table',display(summary.query("population=='original13'")[[ 'stage','family','ce','rps']]),'Common original-13 predictive scores')],[
        'This comparison asks whether larger training datasets improve each architecture. The primary comparison uses exactly paired original-13 fixtures; all-tested scores describe changing populations and cannot alone establish a training-data benefit.',
        'Three independently scored seeds, four expanding-history test periods, three model families. Probabilities are not ensembled. Negative paired CE/RPS changes favor the larger dataset. These repeatedly inspected periods are exploratory.',
        'Every stage uses the same workbook source, identity rules, statistic definitions and actual-playing-date cutoff. Expanded stages change the available histories and league dimensions. Comparisons to older native-source runs also change source and timing and are not controlled sample-size comparisons.',
        'Historical financial results are simulations, not placed bets, executable quotes or future-return claims.'])
    add('Predictive performance',[Artifact('table',display(summary),'Seed-mean fixture-weighted scores'),Artifact('table',display(pd.read_csv(analysis/'scores_equal_fold_mean.csv')),'Equal-fold mean scores'),Artifact('table',primary,'Paired common-13 differences and pointwise 95% intervals')],[
        'Natural-log CE without clipping; normalized RPS is half the sum of two H/D/A cumulative squared errors. Accuracy uses H/D/A argmax; Brier is sum of three squared class errors. Goal RMSE pools squared home/away errors within each seed then takes the square root, only for C/E1. Seed means average scores, not probabilities.',
        'Pooled scores weight fixtures; equal-fold scores weight each fold equally. All-tested league composition differs by dataset. Paired calendar block intervals average seed losses first, preserve original fold-by-league fixture weights, and share calendar draws across leagues. Four-week blocks primary; two/eight-week sensitivity. Conditional on fitted seeds, without multiple-comparison correction.'])
    daily=pd.read_parquet(analysis/'financial_daily.parquet');fig=go.Figure()
    for (stage,family,seed),g in daily.groupby(['stage','family','seed'],sort=True):
        label=f'{LABELS[family]} / {LABELS[stage]} / seed {seed}'
        fig.add_trace(go.Scatter(x=g.day,y=g.cumulative_profit,name=label,legendgroup=family,hovertemplate='%{x|%Y-%m-%d}<br>%{y:.4f} units<extra>'+label+'</extra>'))
    fig.add_hline(y=0,line_color='white',line_width=2)
    fig.update_layout(title='Continuous simulated cumulative profit',xaxis=dict(rangeslider=dict(visible=False)),yaxis=dict(title='Profit (stake units)',rangemode='tozero'),hovermode='closest',legend=dict(groupclick='toggleitem',itemdoubleclick='toggleothers'))
    add('Continuous cumulative profit',[Artifact('plotly',fig,'Original-13 simulated profit')],['Same full UTC calendar for every curve, inactive days included. No resets at season boundaries. Each seed is a separate simulation, never three simultaneous stakes.'])
    add('Financial results',[Artifact('table',display(pd.read_csv(analysis/'financial_seed.csv')),'Original-13 financial results by seed')],[
        'Opening draw odds; draw probability >=20%; strict positive expected value; one-unit flat stakes. Missing/invalid prices excluded and counted. Settled draw returns odds minus one, loss minus one. No modeled commissions, void refunds or bankroll compounding; the eligible fixture cohort must already exclude unsupported settlements.',
        'Archived opening-snapshot bytes and fixture identities are hash verified, but quote observation timestamps are unverified. Settlement availability is not quote observation time. These retrospective simulations are not certified executable returns.',
        'ROI = profit / actual stake. Daily unit-PnL Sharpe-like statistic = sqrt(365) × mean daily PnL / sample standard deviation (ddof=1), inactive days included; undefined at zero variance. It is not a bankroll-normalized investment return. Absolute drawdown uses the continuous cumulative-profit peak including initial zero.'])
    add('Season and league detail',[Artifact('table',display(pd.read_csv(analysis/'scores_fold_seed.csv')),'Per-fold and seed scores'),Artifact('table',display(pd.read_csv(analysis/'scores_league_fold_seed.csv')),'Per-league, fold and seed scores'),Artifact('table',paired,'Calendar-block sensitivity')])
    # Native exporters own their assets; links retain the existing relative tree.
    add('Tickets and GAT diagnostics',[Artifact('text','Native ticket viewer: ../'+support['tickets']+'\nNative embeddings and attention: ../'+support['diagnostics'])],[
        'Open the supplied native entry pages. Independent fitted latent spaces retain independent projections; no cross-model coordinate alignment is implied. Diagnostic issuance is distinct from fixture forecast issuance.'])
    for role in ('selection','refit'):
        for path in sorted((base/role).rglob('history.csv')):
            h=pd.read_csv(path);fit=json.loads((path.parent/'fit.json').read_text());wide=h.pivot(index='step',columns='metric',values='value');arts=[]
            for name,cols in [('Outcome cross-entropy',('train_loss','validation_loss')),('Poisson goal loss',('train_goal_loss','validation_goal_loss')),('Combined objective',('train_objective','validation_objective'))]:
                figure=go.Figure()
                for col in cols:
                    if col in wide:figure.add_trace(go.Scatter(x=wide.index,y=wide[col],name=col.replace('_',' '),mode='lines'))
                if not len(figure.data):continue
                if role=='selection':
                    require('train_loss' in wide and 'validation_loss' in wide,'Missing mandatory selection curves')
                    figure.add_vline(x=fit['summary']['retained_step'],line_dash='dash')
                figure.update_layout(title=name,xaxis_title='Completed optimizer update',yaxis_title='Loss')
                arts.append(Artifact('plotly',figure,name))
            add(('Selection: ' if role=='selection' else 'Training-only refit: ')+str(path.parent.relative_to(base/role)),arts,[
                'Curves are saved native measurements; no test loss is used for checkpoint selection. C/E1 uses auxiliary Poisson supervision, not quantile loss. See fit receipt for update budget and termination; selection marks retained checkpoint.'],n=0)
    add('Methods and evidence',[Artifact('text',json.dumps(support.get('methods',{}),indent=2))],[
        'Candidate report: rendering, ticket/badge/latent-selector checks, independent numerical review, package-union and Drive verification remain separate release gates. Source inspection alone is not rendered verification.'])
    AnalysisReport(studies=studies,title='GAT dataset expansion: four-fold comparison').to_html(out/'INDEX.html')
    (out/'REPORT_QA.json').write_text(json.dumps(dict(status='CANDIDATE_NOT_RELEASED',rendered_browser='PENDING',independent_review='PENDING',native_support=support,analysis_receipt_sha256=digest(analysis/'ANALYSIS_COMPLETE.json')),indent=2))
    return out

if __name__=='__main__':report()
