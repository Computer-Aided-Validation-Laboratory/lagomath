# Scalar baseline

Untouched kernel baseline captured 2026-11-19 before rename or numerical changes.

* Git: `d4ebf9010496d4196dc8e8b87b8b4583a008441e`
* Zig: 0.16.0; ReleaseFast; native default CPU features
* CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
* Linux-7.0.0-43-generic-x86_64-with-glibc2.39
* Cases: 184; f32/f64; seven samples each
* Command: `zig build bench -Doptimize=ReleaseFast`
* Summarise: `.venv/bin/python benchmarks/python/run_benchmarks.py --label zig-before --existing benchmarks/results/zig-before.jsonl`
* Raw/summary files: `benchmarks/results/zig-before.{jsonl,summary.json}` (machine-local, ignored)
* Clock: CLOCK_MONOTONIC. Allocations, setup and JSON parsing excluded. Compiler barriers included; tiny-kernel results include dispatch/barrier overhead.
* No explicit SIMD; compiler auto-vectorisation allowed and not instruction-audited. Laptop frequencies and scheduling are uncontrolled; timings are advisory.

Selected medians (ns/call):

| Case | ns |
|---|---:|
| bench.dot.Vec.3.f64 | 3.91 |
| bench.dot.VecView.1048576.f64 | 659448.00 |
| bench.matmul.Mat.6.f64 | 30.36 |
| bench.lu.Mat.6.f64 | 105.81 |
| bench.cholesky.Mat.6.f64 | 101.29 |

## Completed shared workload

The final harness covers 870 workloads in f32/f64: fixed and borrowed vectors, long slice arithmetic/reductions, square and rectangular matrices, outer products, specialised determinant/inverse kernels, standalone factors, one versus sixteen RHS, pivot-heavy/signed/dense input families, and NDArray ranks 1–5 with reads/writes and equal-size streaming shapes. Both languages use identical JSON workload definitions, validated checksums and seven samples per case, x86-64 baseline targets and disabled assertions.

The original kernels come from an isolated archive of `d4ebf9010496d4196dc8e8b87b8b4583a008441e`; only the current harness and its mechanical old-name substitutions are supplied. Kernel algorithms in the archive remain unchanged. Warmups and output/input liveness barriers apply equally to the original and final harnesses. Allocation, input generation, metadata construction, JSON and reuse factorisation/RHS preparation are outside timing; dispatch, barriers and API-required copies/resets are included.

Reproduction commands:

```sh
.venv/bin/python tools/run_original_baseline.py
.venv/bin/python benchmarks/python/run_benchmarks.py --label zig-phaseA
.venv/bin/python benchmarks/python/run_benchmarks.py --backend mojo --label mojo-phaseA
.venv/bin/python benchmarks/python/compare_results.py benchmarks/results/zig-phaseA.summary.json benchmarks/results/mojo-phaseA.summary.json
```

Commands print compact summaries and save detailed records to ignored benchmarks/results and build/logs. Complete final summaries are versioned as [original Zig](baselines/zig-original-current-harness.json), [final Zig](baselines/zig-final.json), [Mojo](baselines/mojo-final.json), and [Zig/Mojo comparison](baselines/zig-mojo-comparison.json). They record workload, kernel/harness source hashes, compiler/CPU/host flags and min/median/mean/max/standard deviation.

Selected medians, ns per invocation (batch cases include all sixteen solves):

| Case | Original Zig | Scaled Zig | Mojo |
|---|---:|---:|---:|
| bench.dot.Vec.3.f64 | 2.70 | 2.91 | 10.92 |
| bench.dot.VecView.1048576.f64 | 668326.60 | 676674.60 | 683541.80 |
| bench.matmul.Mat.6.f64 | 22.25 | 22.45 | 71.54 |
| bench.lu.Mat.6.f64 | 102.10 | 158.10 | 213.11 |
| bench.cholesky.Mat.6.f64 | 102.10 | 124.04 | 207.40 |
| bench.matmulRect.MatView.6x12x8.f64 | 58.61 | 58.62 | 141.07 |
| bench.inv.Mat.4x4.f64 | 26.15 | 26.85 | 143.07 |
| bench.luBatch16.Mat.6x6.f64 | 411.79 | 886.80 | 1331.94 |
| bench.gauss.Mat.6x6.pivot_required.f64 | 107.31 | 124.03 | 189.56 |
| bench.getFlatIdx.NDArray.2x4x8x8x8.f64 | 3.30 | 3.21 | 10.32 |

The original and rename-only 184-case runs are retained separately. Their different workload hashes must not be compared with the final 870-case summaries. Global scaling intentionally adds solver work; this final comparison measures the resulting algorithms and Span port, rather than isolating the rename. Laptop frequencies/scheduling are uncontrolled, so timing ratios are advisory. Compiler auto-vectorisation is enabled by default and machine instructions have not been audited; no explicit SIMD kernels are present.
