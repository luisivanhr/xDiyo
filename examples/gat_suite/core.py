"""Immutable populations, common preprocessing, canonical initialization and manifests."""
from dataclasses import asdict, replace
from pathlib import Path
from hashlib import sha256
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import torch

from xdiyo_analytics.graphs import TemporalGraphConfig, TemporalEventStore, GoalAuxiliary
from xdiyo_analytics.graphs.adapter import _Backend
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor
from xdiyo_analytics.graphs.streaming import GroupPlan
from xdiyo_analytics.graphs.config import MATCH_KEYS
from xdiyo_analytics.graphs.goals import goal_labels
from xdiyo_analytics.graphs.network import TemporalNetwork
from xdiyo_analytics.graphs.execution import prepare_determinism
from xdiyo_analytics.training.residency import InputCachePolicy
from temporal_team_gat import make_context
from compare_gat_temporal import digest, write_json, frame_hash
from compare_gat_goals import digest_state

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ('01_supervised_window', '02_temporal_historical', '03_structured_poisson', '04_shared_random_effects')
MODES = ('snapshot', 'time_features', 'historical')
SEEDS = (17, 42, 73)
ARMS = ('A_recent', 'B_expanded', 'C_flexible', 'S_structured_poisson', 'N_independent_nb', 'G_shared_gamma')
PARENT = ROOT/'experiments/gat_goal_auxiliary_cuda_20261026'


def config(mode='snapshot', arm='A_recent', competitions=(35,)):
    goal = GoalAuxiliary(beta=0. if arm in ARMS[:2] else .1)
    if arm in ARMS[3:]:
        likelihood = dict(S_structured_poisson='poisson', N_independent_nb='independent_nb', G_shared_gamma='shared_gamma')[arm]
        goal = GoalAuxiliary(architecture='attack_defence', likelihood=likelihood, competitions=competitions)
    return TemporalGraphConfig(fixture_gap_encoding='legacy_standardized_gap_v1', timezone='Europe/Berlin', temporal_mode=mode, deterministic=True, goal_auxiliary=goal,
        time_encoding='rational_decay_v1', age_scale_days=180., gap_scale_days=7., time_scales_days=(7.,30.,180.))


def matrix():
    rows = []
    for study in range(1, 5):
        modes = ('snapshot',) if study == 1 else MODES[1:] if study == 2 else MODES
        arms = ARMS[:3] if study < 3 else ARMS[3:4] if study == 3 else ARMS[4:]
        for fold in ('fold_1', 'fold_2'):
            for mode in modes:
                for seed in SEEDS:
                    for arm in arms:
                        relative = f'{STUDIES[study-1]}/{fold}/{mode}/{arm}/seed_{seed}'
                        rows.append(dict(job_id=f'{fold}__{mode}__{arm}__{seed}', study=study,
                            fold=fold, mode=mode, arm=arm, seed=seed, path=relative, status='planned', provenance='new'))
    assert len(rows) == 108 and len({r['job_id'] for r in rows}) == 108
    return rows


def load(run, *, frozen=False):
    run = Path(run)
    protocol = json.loads((run/'MASTER_PROTOCOL.json').read_text())
    if frozen:
        receipt = json.loads((run/'FREEZE.json').read_text())
        if digest(run/'MASTER_PROTOCOL.json') != receipt['master_sha256']:
            raise ValueError('Master protocol changed after freeze.')
        for name, expected in json.loads((run/'SOURCE_MANIFEST.json').read_text()).items():
            if digest(ROOT/name) != expected: raise ValueError(f'Frozen source changed: {name}')
        for name, expected in receipt['shared_hashes'].items():
            if digest(run/name) != expected: raise ValueError(f'Frozen shared artifact changed: {name}')
    data = run/'shared/data'
    store = TemporalEventStore(pd.read_parquet(data/'history.parquet'), pd.read_parquet(data/'memberships.parquet'),
        schedule=pd.read_parquet(data/'schedule.parquet'), provenance=protocol['provenance'])
    if store.version != protocol['store_version']: raise ValueError('Archive identity changed.')
    return protocol, store


def frames(run, fold):
    return {key:pd.read_parquet(Path(run)/'shared/data'/f'{fold}_{key}.parquet') for key in ('recent','expanded','inner','outer')}


def initialize_state(cfg, preprocessor, seed, snapshot=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TemporalNetwork(cfg, len(preprocessor.leagues), preprocessor.edge_dim,
                                preprocessor.fixture_dim)
    if snapshot is not None:
        target = model.state_dict()
        for name, value in snapshot.items():
            if target[name].shape == value.shape:
                target[name] = value.clone()
            elif name.endswith(('.e.weight', '.u.weight')):
                assert target[name].shape[1] == value.shape[1]+14
                target[name].zero_(); target[name][:,:value.shape[1]] = value
            else:
                raise ValueError(f'Unexpected canonical tensor mismatch: {name}')
        model.load_state_dict(target)
    return model


def prepare(run):
    run = Path(run)
    if (run/'FREEZE.json').exists(): raise ValueError('Already frozen; create a new revision instead.')
    old = json.loads((PARENT/'protocol.json').read_text())
    data = run/'shared/data'
    for name in ('history.parquet','memberships.parquet','schedule.parquet'):
        if digest(data/name) != old['data_hashes'][name]: raise ValueError('Archive data mismatch.')
    store = TemporalEventStore(pd.read_parquet(data/'history.parquet'), pd.read_parquet(data/'memberships.parquet'),
        schedule=pd.read_parquet(data/'schedule.parquet'), provenance=old['provenance'])
    schedule = store.schedule
    targets = schedule[schedule.source_league.eq('Bundesliga')].sort_values(['kickoff_at',*MATCH_KEYS]).reset_index(drop=True)
    counts = targets.groupby('source_season').size().to_dict()
    assert counts == dict(zip(('19_20','20_21','21_22','22_23','23_24','24_25'),(306,306,305,306,306,305)))
    competitions = tuple(map(int, sorted(targets.competition_id.unique())))
    cfg = config(competitions=competitions)
    folds = {}
    for fold, seasons in [('fold_1',('21_22',('20_21','21_22'),'22_23','23_24')),
                          ('fold_2',('22_23',('20_21','21_22','22_23'),'23_24','24_25'))]:
        recent, expanded, inner, outer = seasons
        populations = {k:targets[targets.source_season.isin(v if isinstance(v,tuple) else (v,))].reset_index(drop=True)
                       for k,v in dict(recent=recent,expanded=expanded,inner=inner,outer=outer).items()}
        receipt = {}
        for name, frame in populations.items():
            issue = store.issue_times(frame,cfg)
            store.assert_targets_unobserved(frame,issue,cfg)
            available = store.label_availability(frame,cfg)
            ys = goal_labels(store,frame,available.max(),make_context(frame).y.iloc[:,0])
            frame.to_parquet(data/f'{fold}_{name}.parquet',index=False)
            receipt[name] = dict(count=len(frame),identity_sha256=frame_hash(frame),labels_sha256=digest_state(ys),
                first_cutoff=str(min(cfg.history_cutoff(t) for t in issue)),last_label=str(available.max()),
                round_count=len(frame.groupby(['competition_id','season_id','round'])),weight_policy='uniform native fixture weights')
        assert pd.Timestamp(receipt['expanded']['last_label']) <= pd.Timestamp(receipt['inner']['first_cutoff'])
        assert pd.Timestamp(receipt['inner']['last_label']) <= pd.Timestamp(receipt['outer']['first_cutoff'])
        ctx = make_context(populations['recent'])
        plan = GroupPlan(store,cfg,ctx,store.issue_times(ctx.metadata,cfg))
        numeric, leagues = plan.fitting_inputs()
        largest = max(cfg.history_cutoff(t) for t,_ in plan.descriptors)
        pre = GraphPreprocessor(cfg).fit(store.eligible(largest),(),ctx.metadata,numeric,extra_leagues=leagues)
        write_json(run/'shared/preprocessing'/f'{fold}.json',dict(state=pre.state(),
            fit_population=receipt['recent'],largest_history_cutoff=str(largest),edge_order=pre.edge_columns,fixture_order=pre.fixture_columns))
        folds[fold] = receipt
        for seed in SEEDS:
            snap = initialize_state(cfg,pre,seed)
            snapshot = snap.state_dict()
            for mode in MODES:
                mc = replace(cfg,temporal_mode=mode)
                pp = GraphPreprocessor.from_state(mc,pre.state())
                model = snap if mode=='snapshot' else initialize_state(mc,pp,seed,snapshot)
                folder=run/'shared/initialization'/fold/mode/f'seed_{seed}'; folder.mkdir(parents=True,exist_ok=True)
                torch.save(model.state_dict(),folder/'canonical.pt')
                write_json(folder/'receipt.json',dict(full_state=digest_state(model.state_dict()),
                    common_state=digest_state({k:v for k,v in model.state_dict().items() if not k.startswith('goal_head.')}),
                    named_tensors={k:digest_state(v) for k,v in model.state_dict().items()},temporal_columns_initially_zero=mode!='snapshot'))
                for arm in ARMS[3:]:
                    sc = config(mode,arm,competitions)
                    structured = initialize_state(sc,pp,seed)
                    state = structured.state_dict()
                    for name,value in model.state_dict().items():
                        if not name.startswith('goal_head.'): state[name]=value.clone()
                    structured.load_state_dict(state)
                    expanded_frame = populations['expanded']
                    structured.goal_head.initialize_means(torch.tensor(expanded_frame.competition_id.to_numpy(dtype=np.int64)),
                        torch.tensor(expanded_frame[['goals_for','goals_against']].to_numpy(dtype=float)))
                    torch.save(structured.state_dict(),folder/f'{arm}.pt')
                    common_state={k:v for k,v in structured.state_dict().items() if not k.startswith('goal_head.')}
                    assert digest_state(common_state)==digest_state({k:v for k,v in model.state_dict().items() if not k.startswith('goal_head.')})
                    write_json(folder/f'{arm}_receipt.json',dict(full_state=digest_state(structured.state_dict()),
                        common_state=digest_state(common_state),named_tensors={k:digest_state(v) for k,v in structured.state_dict().items()},
                        fitting_initialization_population=receipt['expanded']['identity_sha256']))
    rng = np.random.default_rng(20261026)
    for fold in folds:
        frame=frames(run,fold)['outer']
        ordering=frame.groupby('round').kickoff_at.min().sort_values().index.tolist()
        folder=run/'shared/bootstrap'/fold; folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'rounds.json',ordering)
        for length in (2,4,8):
            starts=rng.integers(0,len(ordering),size=(20000,int(np.ceil(len(ordering)/length))))
            indices=((starts[:,:,None]+np.arange(length))%len(ordering)).reshape(20000,-1)[:,:len(ordering)]
            counts=np.zeros((20000,len(ordering)),dtype=np.int16)
            np.add.at(counts,(np.arange(20000)[:,None],indices),1)
            np.save(folder/f'blocks_{length}.npy',counts)
    rows=matrix()
    protocol=dict(schema=1,assignment_sha256=digest(run/'HANDOFF.md'),created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        source_parent=dict(protocol_sha256=digest(PARENT/'protocol.json'),frozen_source_sha256=digest(PARENT/'frozen_source.zip')),
        provenance=old['provenance'],store_version=store.version,archive_counts=targets.groupby('source_season').size().to_dict(),
        competitions=competitions,folds=folds,base_config=asdict(cfg),matrix=rows,seeds=SEEDS,modes=MODES,
        training=dict(max_steps=200,patience=15,min_delta=0.,monitor='validation_loss',restore_best=True,scheduler=None),
        bootstrap=dict(replicates=20000,seed=20261026,primary_length=4,sensitivity_lengths=[2,8],circular=True),
        configurations={mode:{arm:asdict(config(mode,arm,competitions)) for arm in ARMS} for mode in MODES},
        initialization=dict(common='canonical named snapshot tensors per fold/seed; same within each mode across arms',
            temporal='copy ordinary projection columns; initialize all 14 added columns to zero, trainable',
            historical='copy complete compatible time_features state; query semantics may change predictions',
            flexible='native linear initialization, copied to A/B/C',
            structured='zero shared bias-free attack/defence maps; scoring/home intercepts from expanded fitting means',
            dispersion='alpha=0.1 by inverse softplus; scalar AdamW lr=0.05, weight_decay=0',
            intercept_optimizer='native learning rate, zero weight decay',warm_start=False),
        baseline_policy=dict(outcome=['fitting_frequency','last_10_available_matches_multinomial_logistic'],
            logistic=dict(C=1.,max_iter=500,random_state=17,transform_fit='recent covariates only, shared across A/B'),
            goals=['fitting_mean_poisson','expanded_centered_attack_defence_poisson','expanded_constant_mean_independent_nb','expanded_constant_mean_shared_gamma'],
            team_effect_penalty=.001,constant_dispersion_allows_poisson_boundary=True),
        calibration=dict(method='draw_intercept',bounds=[-20.,20.],fit_population='inner'),
        resources=dict(device='cuda:0',threads=1,input_cache_bytes=4*2**30,headroom_bytes=2**30,rss_limit_gib=6.,cuda_fraction=.8),
        reuse_policy='All 108 are new fits; earlier pilots have no exact suite preprocessing/initialization receipts.',
        unknown_neutrality='ordinary home term',neutrality_counts={'known_neutral':int(targets.get('neutral_venue',pd.Series(dtype=float)).eq(1).sum()),
            'unknown':int(targets.get('neutral_venue',pd.Series(np.nan,index=targets.index)).isna().sum())},
        limitations=['Retrospective development periods already inspected; no fresh holdout claims.',
            'Retrospective schedule/membership archive; July 1 membership assumption.',
            'Fold training and inner roles overlap; outer seasons are distinct, not independent untouched replications.'])
    write_json(run/'MASTER_PROTOCOL.json',protocol)
    pd.DataFrame(rows).to_csv(run/'RUN_MANIFEST.csv',index=False)
    write_json(run/'preflight/populations.json',folds)
    print(json.dumps(dict(stage='prepared',jobs=len(rows),folds=folds,neutrality=protocol['neutrality_counts'])),flush=True)


def freeze(run):
    run=Path(run)
    if (run/'FREEZE.json').exists(): raise ValueError('Already frozen.')
    required=['mathematical_checks.txt','regression_checks.txt','profile_time_features.json','profile_historical.json','pairing.json']
    for name in required:
        if not (run/'preflight'/name).exists(): raise ValueError(f'Missing preflight: {name}')
    paths=list((ROOT/'src/xdiyo_analytics').rglob('*.py'))+list((ROOT/'examples/gat_suite').glob('*.py'))
    paths += [ROOT/'examples/run_gat_suite.py',ROOT/'examples/temporal_team_gat.py',ROOT/'examples/compare_gat_temporal.py',ROOT/'examples/compare_gat_goals.py']
    paths += list((ROOT/'tests/analytics').glob('*gat_suite*'))+list((ROOT/'tests/analytics').glob('test_graph_goal_auxiliary.py'))+list((ROOT/'tests/analytics').glob('test_graph_temporal_attention.py'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))}
    write_json(run/'SOURCE_MANIFEST.json',hashes)
    with zipfile.ZipFile(run/'frozen_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in hashes: archive.write(ROOT/name,name)
    write_json(run/'FREEZE.json',dict(master_sha256=digest(run/'MASTER_PROTOCOL.json'),source_archive_sha256=digest(run/'frozen_source.zip'),
        shared_hashes={p.relative_to(run).as_posix():digest(p) for p in sorted((run/'shared').rglob('*')) if p.is_file()},
        frozen_at=pd.Timestamp.now(tz='UTC').isoformat()))
    import psutil, scipy, sklearn
    write_json(run/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
        host_memory=psutil.virtual_memory()._asdict(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),source_sha256=digest(run/'frozen_source.zip'),
        deterministic=True,precision='highest',cublas_workspace=':4096:8'))
