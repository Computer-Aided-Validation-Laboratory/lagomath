# Phase 0/A decisions

Approved by the maintainer on 2026-11-19:

* Native names match in Zig and Mojo, including the camelCase Checked suffix.
* Vectors provide normL1, normL2 and normL2Square.
* Standard kernels assert programmer preconditions and assume finite input. Solvers return numerical failures in both standard and Checked forms. Checked variants additionally validate input and storage contracts.
* Solvers use global matrix magnitude scaling, preserving Cholesky symmetry. Numerical changes follow the untouched performance baseline.
* NDArray requires positive dimensions. Retain its backing-slice and split data/metadata ownership contracts, including existing slicing behaviour. No NDView.
* Keep src/root.zig; flat Zig type files acquire the type suffix.
* JSON benchmark infrastructure and the untouched Zig baseline precede API migration.

Implementation choices preserve the existing conveniences and Zig aliases rather than removing them. The verified initial binary target is Linux x86-64, using the working rabbit-mojo toolchain as the reference. The narrow C ABI covers representative f32/f64 arithmetic, matrix products, two solvers and a portable f64 matrix POD. Python extension wheels and cross-platform Mojo binaries remain outside the advertised support.

Canonical float32/float64 operations are recorded in api_manifest.json; the full source inventories are retained alongside that registry. Tests and performance expectations are separate versioned JSON contracts. Default benchmark timings are advisory, with no hard CI speed threshold. The original-commit baseline is reproducible using tools/run_original_baseline.py. Historical numerical tests now delegate to shared fixtures; the method registry and migration ledger document coverage. See shared_test_contract.md.

Native Mojo borrowed storage uses Span throughout, including sliceops arguments and permutation scratch. Raw pointers are confined to the C ABI adapter. Span ownership checks reject native overlap; C ABI overlap validation occurs before foreign buffers become Spans. Allocating Mojo convenience constructors receive caller-owned List storage to preserve split data/metadata ownership with tracked lifetimes.
