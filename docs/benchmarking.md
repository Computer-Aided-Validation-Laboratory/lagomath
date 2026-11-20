# Scalar benchmarks

benchmarks/cases/manifest.json is the shared workload contract for Zig and Mojo: IDs, representation, dtype, dimensions, deterministic input rule, warmups, samples, repetitions and analytical checksum. The 870 shared workloads cover fixed vectors, long/awkward vector views, long slice arithmetic/reductions, fixed/view square and rectangular matrices, specialised inverses/determinants, standalone factorisation, factor reuse with sixteen RHS, and NDArray ranks 1–5 with flat/indexed reads/writes and helper/streaming access.

```sh
.venv/bin/python benchmarks/python/run_benchmarks.py --label zig-current
.venv/bin/python benchmarks/python/run_benchmarks.py --backend mojo --label mojo-current
.venv/bin/python benchmarks/python/compare_results.py benchmarks/results/zig-current.summary.json benchmarks/results/mojo-current.summary.json
```

For a short smoke run append `--samples 2 --runs 2`. CLI overrides write a separate generated fixture module under build/; committed JSON is never edited. For longer measurements increase samples/repetitions. Coverage, unique sample numbers and checksums are validated before a summary is accepted. Summary JSON contains min, median, mean, max, standard deviation, compiler/host/CPU/target/mode and workload hash. Different workload hashes cannot be compared.

Zig uses ReleaseFast and -Dcpu=baseline; Mojo uses -O3, --target-cpu x86-64 and ASSERT=none. Compilers may auto-vectorise scalar source; generated instruction selection is not audited. This is a pre-explicit-SIMD baseline, not a promise of scalar machine instructions.

Monotonic native timing excludes JSON parsing, input generation, allocation, shape construction and factorisation in reuse cases. Fixed-prefix NDArray cases measure access through a preconstructed descriptor; metadata allocation is outside timing. Kernel dispatch, compiler barriers, copies required by the value API, and resetting accumulation destinations remain included. These costs matter for tiny kernels. Python/cffi call overhead is not mixed with native timings.

Raw machine-local runs are ignored under benchmarks/results. dev/scalar_baseline.md records initial runs, including the original-commit baseline and the rename-only run. An expanded harness was also run against an isolated archive of the original commit before comparison with scaled solvers. Laptop frequencies/scheduling are uncontrolled, so CI timings are advisory and not hard speed gates. Broad performance claims require stable hardware and longer runs.

Successful benchmark runs print one summary line. Build diagnostics are captured in build/logs; raw timing records remain in benchmarks/results. Failures show a bounded diagnostic tail and the full log path. RHS batches and their factors are prepared outside the timed loop.

Native runners are quiet by default. Pass `--jsonl` to a native Mojo binary or `zig build bench -- --jsonl` for explicit machine-readable records. The Python wrapper supplies that flag, captures the records and validates them. The comparison tool writes full comparisons to JSON and prints one line; `--verbose` opts into per-case output. Source hashes cover both the kernels and benchmark harnesses.
