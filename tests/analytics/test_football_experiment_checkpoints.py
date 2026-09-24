import json
import numpy as np
import pandas as pd
import pytest
from football_experiment_samples import prepared,ridge,post
from football_checkpoint_samples import NativeFactory
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.training import CheckpointPolicy

def test_capable_native_adapter_resumes_optimizer_rng_preprocessing_and_cursor(tmp_path):
    preparation=prepared(holdout=True);events=[];gate=tmp_path/'resume.flag'
    factory=NativeFactory(events,gate=gate);model=Candidate('Native SGD',factory,config=factory.cache_key())
    experiment=FootballExperiment('Interrupted native fit',output_dir=tmp_path)
    with pytest.raises(RuntimeError,match='native checkpoint'):
        experiment.run(preparation,model=model,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    pointers=list((experiment.store.path/'checkpoints').rglob('latest.json'));assert len(pointers)==1
    pointer=pointers[0];published=pointer.parent/json.loads(pointer.read_text())['directory']
    assert json.loads((published/'metadata.json').read_text())['cursor']==3
    immutable={p.name:p.read_bytes() for p in published.iterdir()}
    assert not experiment.store.read_runs()
    gate.write_text('resume')
    resumed=experiment.run(preparation,model=model,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    assert events==[('start',0),('step',1),('step',2),('step',3),('start',3),('step',4),('step',5),('step',6)]
    assert immutable=={p.name:p.read_bytes() for p in published.iterdir()}
    reference_events=[];reference=FootballExperiment('Uninterrupted native reference',output_dir=tmp_path).run(
        preparation,model=Candidate('Native SGD',NativeFactory(reference_events),config=factory.cache_key()),
        checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    pd.testing.assert_frame_equal(resumed.training.folds[0].predictions['predict'],reference.training.folds[0].predictions['predict'])
    pd.testing.assert_frame_equal(resumed.training.folds[0].training_history,reference.training.folds[0].training_history)
    np.testing.assert_array_equal(resumed.training.folds[0].model.weights,reference.training.folds[0].model.weights)
    assert resumed.training.folds[0].training_summary['steps']==6
    count=len(events)
    cached=experiment.run(preparation,model=model,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    assert cached.reused and cached.training.folds[0].model is None and len(events)==count

@pytest.mark.parametrize('unsupported',['skip','raise'])
def test_ordinary_estimator_checkpoint_support_is_explicit(tmp_path,unsupported):
    experiment=FootballExperiment('Ordinary sklearn',output_dir=tmp_path)
    if unsupported=='raise':
        with pytest.raises(TypeError,match='fit_resumable'):
            experiment.run(prepared(holdout=True),model=ridge(),checkpoint_policy=CheckpointPolicy(unsupported='raise'),post_analysis=post())
    else:
        result=experiment.run(prepared(holdout=True),model=ridge(),checkpoint_policy=CheckpointPolicy(),post_analysis=post())
        assert result.training.folds[0].model is not None
    assert not list((experiment.store.path/'checkpoints').rglob('latest.json'))

@pytest.mark.parametrize('every',[0,-1,1.5,True])
def test_invalid_checkpoint_frequency(every):
    with pytest.raises(ValueError):CheckpointPolicy(every)

def test_offered_checkpoint_frequency_and_reuse_false_namespace(tmp_path):
    preparation=prepared(holdout=True);events=[]
    factory=NativeFactory(events)
    experiment=FootballExperiment('Offered frequency',output_dir=tmp_path)
    candidate=Candidate('Native',factory,config=factory.cache_key())
    one=experiment.run(preparation,model=candidate,checkpoint_policy=CheckpointPolicy(every=2),post_analysis=post())
    folders=list((experiment.store.path/'checkpoints').iterdir());assert len(folders)==1
    assert len([p for p in folders[0].iterdir() if p.is_dir()])==3
    two=experiment.run(preparation,model=candidate,checkpoint_policy=CheckpointPolicy(every=2),post_analysis=post(),reuse=False)
    assert one.record['run_group']!=two.record['run_group']
    assert [e for e in events if e[0]=='start']==[('start',0),('start',0)]
    assert len(list((experiment.store.path/'checkpoints').iterdir()))==2

def test_failed_checkpoint_writer_keeps_the_last_published_state(tmp_path,monkeypatch):
    events=[];factory=NativeFactory(events);experiment=FootballExperiment('Writer interruption',output_dir=tmp_path)
    candidate=Candidate('Native',factory,config=factory.cache_key());preparation=prepared(holdout=True)
    original=np.savez;calls=[]
    def interrupted(path,**state):
        calls.append(path)
        if len(calls)==2:raise OSError('injected checkpoint disk failure')
        original(path,**state)
    with monkeypatch.context() as patch:
        patch.setattr(np,'savez',interrupted)
        with pytest.raises(OSError,match='disk failure'):
            experiment.run(preparation,model=candidate,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    pointer=next((experiment.store.path/'checkpoints').rglob('latest.json'))
    checkpoint=pointer.parent/json.loads(pointer.read_text())['directory']
    assert json.loads((checkpoint/'metadata.json').read_text())['cursor']==1
    before={p.name:p.read_bytes() for p in checkpoint.iterdir()}
    result=experiment.run(preparation,model=candidate,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    assert ('start',1) in events and result.training.folds[0].training_summary['steps']==6
    assert before=={p.name:p.read_bytes() for p in checkpoint.iterdir()}

def test_invalid_native_checkpoint_pointer_fails_before_resume(tmp_path):
    gate=tmp_path/'gate';factory=NativeFactory([],gate=gate)
    experiment=FootballExperiment('Invalid pointer',output_dir=tmp_path);preparation=prepared(holdout=True)
    candidate=Candidate('Native',factory,config=factory.cache_key())
    with pytest.raises(RuntimeError,match='native checkpoint'):
        experiment.run(preparation,model=candidate,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
    pointer=next((experiment.store.path/'checkpoints').rglob('latest.json'))
    pointer.write_text(json.dumps({'directory':'../../outside'}));gate.write_text('resume')
    with pytest.raises(ValueError,match='Invalid native checkpoint pointer'):
        experiment.run(preparation,model=candidate,checkpoint_policy=CheckpointPolicy(),post_analysis=post())
