from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.estimator_checks import parametrize_with_checks
from xdiyo_analytics.training import SpatialPCA, SpatialClusters
from xdiyo_analytics.ui import catalog_for_ui


@parametrize_with_checks([SpatialPCA(n_components=1),SpatialClusters(n_clusters=1)])
def test_sklearn_contract(estimator,check):
    check(estimator)


@pytest.mark.parametrize('cls,kwargs',[(SpatialPCA,{'n_components':2}),(SpatialClusters,{'n_clusters':2})])
@pytest.mark.parametrize('n',[2,3,7])
def test_shared_basis_fold_fit_and_roundtrip(cls,kwargs,n):
    rng=np.random.default_rng(3)
    columns=[f'{side}_{i}' for side in ('home','away') for i in range(n*n)]
    X=pd.DataFrame(rng.random((14,2*n*n)),columns=columns,index=range(30,44))
    X['other']=np.arange(len(X));blocks=[columns[:n*n],columns[n*n:]]
    X.loc[30,blocks[0]]=np.nan
    train=X.iloc[:8].copy(); test=X.iloc[8:].copy()
    transformer=cls(blocks=blocks,**kwargs).fit(train)
    assert transformer.n_training_maps_==15
    imputer=transformer.imputer_.statistics_.copy()
    original=transformer.transform(test)
    altered=test*100
    transformer.transform(altered)
    np.testing.assert_array_equal(transformer.imputer_.statistics_,imputer)
    assert original.shape==(6,5)
    np.testing.assert_array_equal(original[:,0],test.other)
    # Both team blocks transformed by exactly the same stored representation.
    duplicate=test.copy();duplicate[blocks[1]]=test[blocks[0]].to_numpy()
    output=transformer.transform(duplicate)
    np.testing.assert_allclose(output[:,1:3],output[:,3:5])
    pd_output=deepcopy(transformer).set_output(transform='pandas').transform(test)
    assert pd_output.index.equals(test.index)
    np.testing.assert_array_equal(pd_output.columns,transformer.get_feature_names_out())
    np.testing.assert_allclose(pickle.loads(pickle.dumps(transformer)).transform(test),original)
    catalog=catalog_for_ui();restored=catalog.build(catalog.encode(transformer))
    np.testing.assert_allclose(restored.fit(train).transform(test),original)
    assert not hasattr(clone(transformer),'encoder_')
    fresh=cls(blocks=blocks,**kwargs).fit(X.iloc[2:8])
    transformer.fit(X.iloc[2:8])
    np.testing.assert_allclose(transformer.transform(test),fresh.transform(test))


def test_pipeline_search_and_invalid_blocks():
    X=np.random.default_rng(8).normal(size=(24,4));y=X[:,0]+X[:,1]
    pipeline=Pipeline([('spatial',SpatialPCA(n_components=1)),('model',Ridge())])
    search=GridSearchCV(pipeline,{'spatial__n_components':[1,2]},cv=TimeSeriesSplit(2),error_score='raise').fit(X,y)
    assert search.best_estimator_.predict(X).shape==(24,)
    for blocks in ([[0,1,2]],[[0,1,2,3],[0,1,2,3]],[[0,1,2,9]]):
        with pytest.raises(ValueError):SpatialPCA(blocks=blocks).fit(X)
    with pytest.raises(ValueError,match='observed training'):
        SpatialPCA().fit(np.full((8,4),np.nan))
