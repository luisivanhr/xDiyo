# Retain reusable inputs during larger sequential scans

Native graph and CNN adapters accept an optional admission policy:

```python
adapter.configure_input_cache(max_bytes=8 * 2**30,
                              headroom_bytes=2**30,
                              admission='retain')
```

The default remains `admission='lru'`. The opt-in `retain` policy keeps already
cached immutable inputs when a miss cannot be admitted, and streams that miss.
It evicts entries only as needed for the current consumer's byte budget or
physical headroom. This avoids replacing every reusable input during repeated
sequential scans larger than the cache. Consumer tensors and metadata remain
isolated copies; activations, gradients and model state are never cached.

For this policy, available CUDA memory includes driver-free bytes and unused
blocks already reserved by PyTorch's allocator. Reserved memory is not counted
as live cached tensor bytes. Allocation can still fail because of fragmentation
or unrelated allocations. The policy does not reserve a GPU or change training
math, input order, observations, loss or precision. An input larger than the
configured budget follows the existing streaming fallback.

The shared pool enforces the smallest participating owner's byte budget and
largest headroom requirement. Owners release their entries when closed. CPU
adapters retain their normal uncached path. Native checkpoints keep the legacy
two-field execution profile for default LRU. Opting into `retain` adds the
admission value to that profile and requires a matching profile on resume.

`test_input_residency_admission.py` checks bounded partial scans, copy isolation,
shared owners, physical headroom, oversized inputs, consumer exceptions,
allocator accounting and profile compatibility. CPU simulations establish these
contracts; they do not establish a CUDA speedup.
