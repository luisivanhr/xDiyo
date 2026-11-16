"""Small real-data pipeline checks for the reviewed expansion; no fitting."""
import json
import hashlib
from pathlib import Path
import pandas as pd

from xdiyo_analytics.data import load_season, load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import TeamMovement, SeasonCountdown, evaluate_features
from xdiyo_analytics.data._source import _open_source
from xdiyo_analytics.labels import Outcome, create_labels
from xdiyo_analytics.datasets import assemble_dataset


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root='data/xDiyo_data'):
    root = Path(root)
    receipt_path = Path('docs/analytics/data/library_preparation_20261116.json')
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['script_sha256'] == digest('examples/prepare_library_20261116.py')
    assert receipt['review_sha256'] == digest('docs/analytics/data/library_movement_review_20261116.json')
    total = 0
    for entry in receipt['seasons']:
        stem = entry['stem']
        source = _open_source(root, stem)
        assert digest(source.path) == entry['source_manifest_sha256']
        assert digest(root / (stem + '.manifest.json')) == entry['publication_sha256']
        assert digest(root / (stem + '.parquet')) == entry['season_sha256']
        for name, info in entry['tables'].items():
            assert digest(source.table_path(name)) == info['sha256']
        loaded = load_season(root, stem, tables=['matches', 'team_seasons'], verify_hashes=True, include_awarded=True)
        assert len(loaded.matches) == entry['matches']
        assert loaded.matches.annual_season.eq(entry['calendar']['annual_season']).all()
        assert 'season_phase' not in loaded.matches  # do not inject a numeric feature input
        assert not loaded['team_seasons'][['got_promoted', 'got_demoted']].isna().any().any()
        total += len(loaded.matches)
    assert total == receipt['matches']
    checks = []
    for stem in ['Greek_Super_League_20_21', 'Colombia_Primera_A_Apertura_21_21',
                 'Colombia_Primera_A_Clausura_21_21', 'Brazil_Serie_B_17_17', 'Liga_Profesional_26_26']:
        data = load_season(root, stem, tables=['matches', 'statistics', 'team_seasons'],
                           verify_hashes=True, include_awarded=True)
        history = build_team_history(data)
        features = evaluate_features(history, {'promoted': TeamMovement('promoted'),
            'relegated': TeamMovement('relegated')}, keyed=True)
        label = create_labels(history, {'target': Outcome(perspective='home')})['target']
        assembled = assemble_dataset(features, label, layout='match')
        assert not assembled.X.isna().any().any()
        assert len(history) == 2 * len(data.matches)
        if stem.startswith('Greek'):
            ids = {1215338386170877, 1007896398245367, 1138374924318711}
            assert ids <= set(data.matches.event_id)
            assert ids <= set(data.statistics.event_id)
            pub = json.loads((Path(root) / (stem + '.manifest.json')).read_bytes())
            review = pub['reconciliation']['archive_cohort']['reviewed_source_gaps']['review']
            assert review['provider_event_ids_verified'] is False
        checks.append(dict(stem=stem, matches=len(data.matches), team_rows=len(history),
            feature_rows=len(features), assembled_rows=len(assembled.X), columns=list(assembled.X),
            source_manifest_sha256=digest(data.provenance['manifest_path'])))
    combined = load_seasons(root, ['17_17'], leagues=['Colombia_Primera_A_Apertura',
        'Colombia_Primera_A_Clausura'], tables=['matches', 'team_seasons'], verify_hashes=True,
        include_awarded=True)
    assert combined.matches.annual_season.unique().tolist() == ['2017']
    assert set(combined.matches.season_half) == {1, 2}
    assert combined.matches.event_id.is_unique
    history = build_team_history(combined)
    assert len(history) == len(combined.matches) * 2
    japan = build_team_history(load_seasons(root, ['16_16'], leagues=['J1'], include_awarded=True))
    features = {'clock': SeasonCountdown(team_count=18)}
    clock = evaluate_features(japan, features, keyed=True)
    without_calendar = japan.drop(columns=['annual_league', 'annual_season', 'annual_phase', 'season_half'])
    pd.testing.assert_frame_equal(clock, evaluate_features(without_calendar, features, keyed=True))
    modules = ['src/xdiyo_analytics/data/season_calendar.py', 'src/xdiyo_analytics/data/loading.py',
               'src/xdiyo_analytics/histories/team.py', 'examples/prepare_library_20261116.py',
               'tests/analytics/test_library_consolidation.py']
    output = dict(script_sha256=digest(__file__), source_hashes={p: digest(p) for p in modules},
                  publication_receipt_sha256=digest(receipt_path),
                  validated_publications=len(receipt['seasons']), validated_matches=total,
                  checks=checks, japan_2016_countdown_parity_rows=len(clock),
                  combined_colombia_2017_matches=len(combined.matches))
    Path('docs/analytics/data/library_pipeline_smoke_20261116.json').write_text(
        json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    verify()
