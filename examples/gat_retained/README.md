# Retained architecture recipes

The six modules explicitly name the original family and fold. CD_DW uses W10 in fold1 and W20 in fold2. Seeds 17, 42 and 73 are separate choices; there is no default seed.

Set `PYTHONPATH` to the integration checkout's `src` and `examples` directories. For example:

```powershell
python -m gat_retained.ce1_fold2 --research-root C:/Users/luisi/Documents/Programming/Python/xDiyo --seed 17 --prepare-only
```

Remove `--prepare-only` and supply a new `--output` directory outside the protected research checkout to run an explicitly requested replay. The default device is CUDA. Preparation verifies the recorded bytes, exact original ComponentStore, phase source, ordered populations, canonical preprocessing and initial weights. It never changes those inputs. All recipes use native TemporalGraphModel construction and the native checkpoint/training loop, 200 updates maximum, patience 15, no scheduler/restarts, and raw inner outcome CE selection.

These archived populations are distinct from fresh UI recipes. The ordinary UI path uses native TemporalEventStore and its named-data guards. An architecture factory alone does not reproduce an original fit. Replay output has new source lineage; historical exports and old native checkpoints retain their original authority.
