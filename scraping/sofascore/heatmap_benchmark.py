"""Compare flat points with typed nested points, without making API requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .benchmark import peak_working_set
from .store import atomic_write, json_bytes

POINT_FIELDS = ["point_order", "x", "y", "weight", "source_item_json"]


def pack(table):
    table = table.combine_chunks()
    keys = [c for c in table.column_names if c not in POINT_FIELDS]
    # Existing canonical exports preserve each group's point order contiguously.
    columns = [table[k].to_pylist() for k in keys]
    groups = list(zip(*columns))
    starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i-1]]
    if len({groups[i] for i in starts}) != len(starts):
        raise ValueError("Point groups are not contiguous; sort explicitly before this benchmark")
    indices = pa.array(starts, type=pa.int64())
    point_struct = pa.StructArray.from_arrays([table[c].chunk(0) for c in POINT_FIELDS], names=POINT_FIELDS)
    values = pa.ListArray.from_arrays(pa.array([*starts, table.num_rows], type=pa.int32()), point_struct)
    return pa.Table.from_arrays([pc.take(table[k], indices) for k in keys]+[values], names=keys+["points"])


def unpack(nested, order):
    points = pc.list_flatten(nested["points"]).combine_chunks()
    parents = pc.list_parent_indices(nested["points"])
    arrays = [pc.struct_field(points,c) if c in POINT_FIELDS else pc.take(nested[c],parents) for c in order]
    return pa.Table.from_arrays(arrays,names=order)


def measure(path, layout):
    start = time.perf_counter()
    table = pq.ParquetFile(path).read()
    points = table.num_rows if layout == "flat" else len(pc.list_flatten(table["points"]))
    return {"layout":layout,"points":points,"seconds":time.perf_counter()-start,
            "arrow_bytes":table.nbytes,"process_peak_working_set_bytes":peak_working_set()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--measure",choices=["flat","nested"])
    args = parser.parse_args(argv)
    if args.measure:
        print(json.dumps(measure(args.input,args.measure)))
        return
    if not args.output:
        parser.error("--output is required")
    flat = pq.ParquetFile(args.input).read()
    nested = pack(flat)
    if not unpack(nested,flat.column_names).equals(flat,check_metadata=False):
        raise ValueError("Nested layout changed point fields, grouping or ordering")
    args.output.mkdir(parents=True,exist_ok=True)
    report = {"source":str(args.input.resolve()),"point_count":flat.num_rows,"groups":nested.num_rows,
              "parity":"all fields and original point order preserved", "variants":[],
              "method":"One full Arrow read per variant in fresh processes; filesystem caches not cleared. Does not measure Python/model view reconstruction."}
    for compression in ("zstd","snappy"):
        for layout, table in (("flat",flat),("nested",nested)):
            path = args.output/f"{layout}_{compression}.parquet"
            pq.write_table(table,path,compression=compression)
            result = subprocess.check_output([sys.executable,"-m","scraping.sofascore.heatmap_benchmark",
                "--input",str(path),"--measure",layout],text=True)
            report["variants"].append({"compression":compression,"file_bytes":path.stat().st_size,**json.loads(result)})
    atomic_write(args.output/"benchmark.json",json_bytes(report))
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
