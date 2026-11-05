"""Check all published movement features without fitting or reading statistics."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from xdiyo_analytics.data.catalogue import select_data, resolve_selection, selected_history, save_selection
from xdiyo_analytics.data.catalogue.storage import file_hash
from xdiyo_analytics.features import TeamMovement, evaluate_features


def run(root):
    folder=Path(__file__).parent
    publication=json.loads((folder/'publication.json').read_text(encoding='utf-8'))
    version=publication['version']
    selection=select_data(root,snapshots={'international_dataset':version},requirements=['goals','movements'])
    data=resolve_selection(root,selection)
    # These predictors do not consume stats. The publication script independently
    # compares the complete unchanged statistics and odds payloads after reload.
    data.tables.pop('statistics',None)
    history=selected_history(data)
    values=evaluate_features(history,{'was_promoted':TeamMovement('promoted'),'was_relegated':TeamMovement('relegated')})
    assert len(history)==2*publication['matches'] and values.notna().all().all()
    assert (values.was_promoted.to_numpy()==history.team_got_promoted.to_numpy()).all()
    assert (values.was_relegated.to_numpy()==history.team_got_demoted.to_numpy()).all()
    save_selection(selection,Path(root)/'international-movements-selection.json')
    result=dict(version=version,fixtures=len(data.matches),history_rows=len(history),all_flags_match=True,
                unknown_feature_values=int(values.isna().sum().sum()),selection_fingerprint=selection['fingerprint'],
                python=sys.version,pandas=pd.__version__,script_sha256=file_hash(__file__))
    (folder/'native-feature-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('catalogue');run(parser.parse_args().catalogue)
