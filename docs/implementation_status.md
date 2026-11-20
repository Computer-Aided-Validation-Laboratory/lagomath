# Phase 0/A implementation status

The agreed Phase 0/A work is implemented and locally verified on Linux x86-64 with Zig 0.16.0 and Mojo 1.2.0 (2016d0cd).

* Canonical Zig type names/files and compatibility aliases; src/root.zig remains in place. Both languages expose the explicit vector norms and matching mathematical entrypoints.
* Gaussian/LU/Cholesky global scaling, legitimate numerical failure errors in standard kernels, and detailed validation through Checked entrypoints.
* Span-backed native Mojo views, slices and permutation scratch. Raw pointers occur only in cabi.mojo. Native ownership checking rejects conflicting tracked borrows; the ABI validates foreign overlap before wrapping Spans.
* Preserved NDArray positive-dimension, oversized backing, tail/plane/prefix and split element/metadata ownership contracts. Mojo allocated conveniences borrow caller-owned List storage. No NDView.
* 16,009 shared numerical cases and five assertion-contract cases. Structured nonfinite outputs, whole-buffer mutation checks, reconstruction, independent SciPy repeated-RHS references, residuals and controlled diagonal conditioning are included.
* All 140 historical named numerical tests now execute shared case groups. Original literals and assertion counts are archived; a checked migration registry maps names to shared IDs. The complete Zig unit suite has 144 checks, including import/type identity and allocator cleanup mechanics.
* An enforced registry of 69 canonical operation entries and 172 method entries, with mathematical versus mechanical applicability, source inventories, case IDs, benchmark IDs and ABI exposure.
* Eight native/ABI build configurations in build/coverage-matrix.json. ABI N/A entries explain operations outside the intentionally narrow C interface and the ABI's finite-input contract.
* Sixteen C exports, four library configurations, C/cffi/POD layout checks, export/dependency inspection and relocated Mojo runtime loading.
* 870 shared benchmark workloads, matching native runners, complete scalar statistics and original-commit/Zig/Mojo baselines. Warmups, setup, aliases, RHS batching and output liveness are explicit. No explicit SIMD.
* Nine paired examples, independent Zig/Mojo consumers, source-whitelist packaging verification, pinned uv environment, developer documentation and CI configuration.
* Quiet native/default verification and benchmark commands. Explicit --jsonl/--verbose switches provide detailed records; wrappers save logs/artifacts and bound failure diagnostics.

Local acceptance: 30 pytest tests passed, Zig Debug/ReleaseFast tests passed, formatting passed, whitelisted Zig consumers passed in both modes, paired examples ran, and ABI exports/runtime dependencies were inspected. Full benchmark summaries are versioned in dev/baselines; raw records and detailed verification logs remain under ignored build/ and benchmarks/results/.

Hosted CI has not been run by this session. Verified binary support is Linux x86-64; universal binaries, Python wheels, GPU, explicit SIMD, QR and NDView remain outside this milestone.
