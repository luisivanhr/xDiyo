"""Construct an explicit frozen-history recipe. Preparation only; never fits a model."""
from xdiyo_analytics.data.catalogue import select_data
from xdiyo_analytics.ui import default_recipe
from xdiyo_analytics.ui.recipe import node


def make_recipe(root, snapshots, train_ids, evaluation_ids, *, mode='extend_history'):
    selection=select_data(root,snapshots=snapshots,mode=mode,requirements=['goals'],
                          train_ids=train_ids,evaluation_ids=evaluation_ids)
    recipe=default_recipe()
    recipe['data_selection']={'root':str(root),'selection':selection}
    recipe['features']={'goals_for20':node('features.RollingMean',source=node('features.MatchScore',side='for'),window=20,min_periods=1),
                        'goals_against20':node('features.RollingMean',source=node('features.MatchScore',side='against'),window=20,min_periods=1)}
    recipe['labels']={'outcome':node('labels.Outcome',perspective='home')}
    recipe['target']='outcome'
    recipe['name']='Frozen reported-goals history example'
    recipe['model']=node('sklearn.linear_model.LogisticRegression',max_iter=1000)
    recipe['prediction_methods']=['predict','predict_proba']
    recipe['post_reporters']={'performance':node('reporting.PerformanceReporter',type='overall',partition='score',metrics=['accuracy'])}
    return recipe
