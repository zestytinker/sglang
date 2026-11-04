# MLX backend: Metal OOM when admitting a batch of new requests

On an Apple M2 (16 GB), the SGLang server crashes with a Metal out-of-memory error
when many concurrent requests are admitted at once. The KV cache is almost empty at
the time. Capping the KV pool with `--max-total-tokens` avoids the crash.

Possibly related upstream issue: https://github.com/sgl-project/sglang/issues/39675
(not confirmed to be the same root cause).

## Environment

| | |
|---|---|
| Machine | Apple M2, 16 GB unified memory |
| OS | macOS 26.6.2 |
| SGLang | source checkout at upstream commit `b702f0d` (2026-11-04), installed with `python[srt_mps]` |
| Python | 3.12.12 |
| torch / mlx / transformers | 2.13.0 / 0.32.4 / 5.19.0 |
| Model | `mlx-community/Qwen3-0.6B-4bit` |

## Steps to reproduce

1. Start the server with default memory settings:

   ```bash
   SGLANG_USE_MLX=1 python -m sglang.launch_server --model-path mlx-community/Qwen3-0.6B-4bit --disable-cuda-graph
   ```

2. Send 16 concurrent streaming requests of 100 output tokens each, using
   `batching-experiment.py` in this branch:

   ```bash
   python batching-experiment.py --batches 1 4 16 --max-tokens 100
   ```

3. Batch sizes 1 and 4 succeed. During batch 16 the server process exits and the
   client gets `peer closed connection without sending complete message body`.

## Observed

| Run | Free memory at startup | KV pool (auto-sized) | Extra flags | Batch 16 |
|---|---|---|---|---|
| A | 8.65 GB | 6.01 GB (56,232 slots) | none | crash |
| B | 8.11 GB | 5.63 GB (52,698 slots) | `--max-running-requests 8` | crash |
| C | 9.21 GB | 2.19 GB (20,000 slots) | `--max-total-tokens 20000` | OK, 294.5 tok/s total |

- KV usage was near zero at the crash: `token usage: 0.00` (run A) and `0.01` (run B).
- Both crashes happen in the prefill (extend) step that admits queued requests:
  run A right after `#queue-req: 13`, run B with `#running-req: 8, #queue-req: 8`.
- Limiting concurrency to 8 (run B) did not help; limiting the KV pool size (run C) did.

Startup log line showing the auto-sizing (run A):

```
Auto-sized attention KV pool: sys_available=8.65 GB, mlx_limit=11.8 GB, mlx_used=0.31 GB, kv_budget=6.01 GB, bytes_per_slot=114688, pool_size=56232
```

Scheduler traceback (run A; run B ends at the same frame):

```
Scheduler hit an exception: Traceback (most recent call last):
  File "python/sglang/srt/managers/scheduler.py", line 6642, in run_scheduler_process
  File "python/sglang/srt/managers/scheduler.py", line 2066, in run_event_loop
  File "python/sglang/srt/managers/scheduler.py", line 6489, in dispatch_event_loop
  File "python/sglang/srt/managers/scheduler.py", line 6501, in _dispatch_event_loop_once
  File "python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py", line 274, in event_loop_overlap_mlx
  File "python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py", line 165, in _launch_fresh
  File "python/sglang/srt/hardware_backend/mlx/tp_worker.py", line 460, in async_forward_batch_generation_mlx
  File "python/sglang/srt/hardware_backend/mlx/tp_worker.py", line 567, in _async_extend_batch
    mx.async_eval(*async_args)
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

## Likely cause (hypothesis)

The KV pool is sized to about 70% of the memory free at startup. When more memory
is free, the pool grows to 5-6 GB, which leaves too little headroom on a 16 GB
machine for the temporary buffers of a multi-request prefill. Earlier the same day,
with 3.8 GB free, the pool was 2.66 GB.

## Workaround

Cap the KV pool explicitly:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server --model-path mlx-community/Qwen3-0.6B-4bit --disable-cuda-graph --max-total-tokens 20000
```
