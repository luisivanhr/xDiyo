"""Bounded, synthetic end-to-end RSS audit against the pinned eager adapter.

Requires psutil. Each mode/population runs in a fresh subprocess. No production
experiment is loaded or fitted. Phase peaks are sampled every 2 ms; the process
high-water RSS is also reported. A conservative available-RAM guard runs first.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

from benchmark_gat_preparation_memory import make_inputs


class MemoryPhases:
    def __init__(self, process):
        self.process, self.phase, self.stats = process, None, {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def sample(self):
        phase = self.phase
        if phase is not None:
            record = self.stats[phase]
            record["peak_rss_bytes"] = max(record["peak_rss_bytes"], self.process.memory_info().rss)

    def _sample(self):
        while not self.stop.wait(.002):
            self.sample()

    def switch(self, phase):
        now = time.perf_counter()
        if self.phase is not None:
            self.sample()
            self.stats[self.phase]["seconds"] += now-self.started
        if phase is not None:
            self.stats.setdefault(phase, dict(peak_rss_bytes=0, seconds=0.))
        self.phase, self.started = phase, now
        self.sample()

    @contextmanager
    def track(self, phase):
        previous = self.phase
        self.switch(phase)
        try:
            yield
        finally:
            self.switch(previous)

    def close(self):
        self.stop.set()
        self.thread.join()


def worker(args):
    import psutil
    import torch
    from xdiyo_analytics.graphs.adapter import _Backend
    torch.set_num_threads(1)
    if args.mode == "reference":
        source = subprocess.check_output(["git", "show", f"{args.reference}:src/xdiyo_analytics/graphs/adapter.py"], text=True)
        module = types.ModuleType("xdiyo_analytics.graphs._memory_reference")
        module.__package__ = "xdiyo_analytics.graphs"
        exec(compile(source, "reference_adapter.py", "exec"), module.__dict__)
        _Backend = module._Backend
    store, cfg, train, prediction = make_inputs(args)
    cfg = replace(cfg, embedding_dim=32, hidden_width=64)
    process = psutil.Process()
    baseline = process.memory_info().rss
    source = store.history
    raw_per_graph = int(source.memory_usage(index=True, deep=True).sum())
    edges_per_graph = len(source)
    all_groups = args.issuances + args.validation_issuances + args.prediction_issuances
    # Deliberately conservative: raw copies, transforms, autograd and runtime.
    live_groups = all_groups if args.mode == "reference" else 1
    extra_estimate = 512*1024**2 + live_groups*(2*raw_per_graph + edges_per_graph*256) + edges_per_graph*32768
    available = psutil.virtual_memory().available
    if extra_estimate > .65*available:
        raise SystemExit(f"Headroom guard refused allocation: estimate={extra_estimate}, available={available}")
    del source
    backend = _Backend(cfg, store, 17, "cpu")
    sampler = MemoryPhases(process)
    sampler.thread.start()
    try:
        with sampler.track("preprocessing_initialization"):
            backend.initialize(train)
        loss = backend._loss
        def measured_loss(groups, targets, weights, backward=False):
            if groups is backend.validation_groups:
                with sampler.track("validation"):
                    return loss(groups, targets, weights, backward)
            return loss(groups, targets, weights, backward)
        backend._loss = measured_loss
        with sampler.track("fitting"):
            metrics = backend.step()
        with sampler.track("prediction"):
            predictions = backend.predict(prediction)
        high_water = process.memory_info()
        if sys.platform == "win32":
            peak = high_water.peak_wset
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    finally:
        sampler.close()
    digest, edge_counts, prepared_bytes = sha256(), [], []
    def audit_group(group):
        positions, data, home, away, fixture = group
        digest.update(positions.tobytes())
        edge_counts.append(len(data["receiver"]))
        size = 0
        for tensor in (*data.values(), home, away, fixture):
            digest.update(tensor.numpy().tobytes())
            size += tensor.numel()*tensor.element_size()
        prepared_bytes.append(size)
    for population in (backend.train_groups, backend.validation_groups or []):
        for index in range(len(population)):
            audit_group(population[index])
    for frame in predictions.values():
        digest.update(frame.to_numpy().tobytes())
    digest.update(json.dumps(backend.preprocessor.state(), sort_keys=True).encode())
    digest.update(json.dumps(metrics, sort_keys=True).encode())
    for state in backend.optimizer.state.values():
        for value in state.values():
            digest.update(value.cpu().numpy().tobytes() if isinstance(value, torch.Tensor) else repr(value).encode())
    resident = sum(prepared_bytes) if isinstance(backend.train_groups, list) else 0
    print(json.dumps(dict(mode=args.mode, reference=args.reference,
        issuances=args.issuances, validation_issuances=args.validation_issuances,
        prediction_issuances=args.prediction_issuances, history_matches=args.history_matches,
        train_validation_directed_edges=sum(edge_counts), prediction_directed_edges=edges_per_graph*args.prediction_issuances,
        maximum_live_graph_edges=edges_per_graph, prepared_population_bytes=sum(prepared_bytes),
        maximum_group_tensor_bytes=max(prepared_bytes), retained_group_tensor_bytes=resident,
        baseline_rss_bytes=baseline, peak_rss_bytes=peak, phases=sampler.stats,
        additional_memory_guard_bytes=extra_estimate, available_bytes_at_guard=available,
        parity_sha256=digest.hexdigest(), model_version=backend.model_version, store_version=store.version,
        torch=torch.__version__, pandas=__import__('pandas').__version__, python=sys.version,
        seed=17, threads=1, device="cpu", config=__import__('dataclasses').asdict(cfg))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "reference"))
    parser.add_argument("--reference", default="f8962af")
    parser.add_argument("--history-matches", type=int, default=2000)
    parser.add_argument("--issuances", type=int, default=120)
    parser.add_argument("--validation-issuances", type=int, default=24)
    parser.add_argument("--prediction-issuances", type=int, default=24)
    parser.add_argument("--schedule-issuances", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    results = []
    for mode in ("reference", "native"):
        command = [sys.executable, __file__, "--mode", mode, "--reference", args.reference]
        for name in ("history_matches", "issuances", "validation_issuances", "prediction_issuances", "schedule_issuances"):
            command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
        result = json.loads(subprocess.check_output(command, text=True))
        results.append(result)
        print(json.dumps(result), flush=True)
    assert results[0]["parity_sha256"] == results[1]["parity_sha256"]
    assert results[0]["model_version"] == results[1]["model_version"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
