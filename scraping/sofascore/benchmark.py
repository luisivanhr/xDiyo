"""Reproducible, local legacy migration check and single-run read benchmark."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

from .cli import load_config
from .export import publish_tables, read_tables
from .legacy import read_legacy_csv, normalize_legacy, legacy_view, legacy_lineups_view, read_compatibility_views
from .store import atomic_write, json_bytes


def peak_working_set():
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in (
            "PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    info=Counters()
    info.cb=ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
        return None
    return info.PeakWorkingSetSize


def measure(mode,path):
    start=time.perf_counter()
    if mode=="csv":
        df,_=read_legacy_csv(path)
    elif mode=="parquet_full":
        df,_=read_compatibility_views(path,include_rosters=True,restore_legacy_xg=True)
    else:
        tables=read_tables(path,names={"matches","statistics","pregame","legacy_fields"} if mode=="parquet_scalars" else None)
        df=legacy_view(tables,restore_legacy_xg=True)
    return {"mode":mode,"rows":len(df),"elapsed_seconds":time.perf_counter()-start,
            "process_peak_working_set_bytes":peak_working_set()}


def compare_legacy(original,lineups,tables):
    import pandas as pd
    reconstructed=legacy_view(tables,restore_legacy_xg=True).set_index("match_id")
    original=original.set_index("match_id")
    assert original.index.equals(reconstructed.index),"Match IDs/order changed"
    checked=0
    for column in original:
        if column=="index" or column.startswith("Unnamed:") or column.endswith("_available"):
            continue
        assert column in reconstructed,column
        if column.endswith("_shots"):
            for eid,shots in original[column].items():
                result=reconstructed.at[eid,column]
                assert len(shots)==len(result),(eid,column,"shot count")
                for before,after in zip(shots,result):
                    for key,value in before.items():
                        assert value==after.get(key),(eid,column,key,value,after.get(key))
        else:
            try:
                pd.testing.assert_series_equal(original[column],reconstructed[column],check_dtype=False,check_names=False)
            except AssertionError:
                raise AssertionError(f"Legacy scalar/nested column differs: {column}") from None
        checked+=1
    if lineups is not None:
        restored=legacy_lineups_view(tables).set_index("match_id")
        for side in ("home","away"):
            for eid,before in lineups.set_index("match_id")[side].items():
                if before != restored.at[eid,side]:
                    raise AssertionError(f"Legacy lineup differs: match_id={eid}, side={side}")
    return {"match_rows":len(original),"compared_match_columns":checked,"lineup_rows":len(lineups) if lineups is not None else 0,
            "result":"all_supported_legacy_values_and_nested_order_match"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--measure",choices=["csv","parquet_full","parquet_scalars"])
    args=parser.parse_args(argv)
    if args.measure:
        print(json.dumps(measure(args.measure,args.input)))
        return
    if not args.config or not args.output:
        parser.error("--config and --output are required for a benchmark")
    scope,profile,_,_=load_config(args.config)
    original,lineups=read_legacy_csv(args.input)
    tables,lineage=normalize_legacy(args.input,scope)
    scope=replace(scope,rounds=tuple(lineage["imported_rounds"]))
    report={"method":"Single local read per mode in a fresh process; filesystem cache not cleared; includes hash validation and view reconstruction",
            "source":lineage,"source_csv_bytes":args.input.stat().st_size,
            "compatibility_check":compare_legacy(original,lineups,tables),"formats":{}}
    lineup_path=args.input.with_name(args.input.stem+"_lineups.csv")
    if lineup_path.exists(): report["source_csv_bytes"]+=lineup_path.stat().st_size
    for compression in ("zstd","snappy"):
        manifest=publish_tables(tables,args.output/compression,scope=scope,profile=profile,provenance=lineage,compression=compression)
        metadata=json.loads(manifest.read_text())
        report["formats"][compression]={"manifest":str(manifest),"parquet_bytes":sum(r["bytes"] for r in metadata["tables"].values()),"reads":[]}
        for mode in ("parquet_full","parquet_scalars"):
            out=subprocess.check_output([sys.executable,"-m","scraping.sofascore.benchmark","--measure",mode,"--input",str(manifest)],text=True)
            report["formats"][compression]["reads"].append(json.loads(out))
    out=subprocess.check_output([sys.executable,"-m","scraping.sofascore.benchmark","--measure","csv","--input",str(args.input)],text=True)
    report["csv_read"]=json.loads(out)
    atomic_write(args.output/"benchmark.json",json_bytes(report))
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
