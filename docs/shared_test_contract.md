# Shared numerical verification

The versioned fixtures in tests/cases contain flat row-major inputs, shapes, dtype, representation, source, operation, case ID, and either expected output plus tolerance or a symbolic expected error. all.json is a generated, compact bundle of the suite files for native embedding. The manifest records generator/reference versions and SHA-256 integrity hashes. Regeneration is deterministic under uv.lock; pytest verifies byte-for-byte reproducibility.

`tests/python/generate_regression.py` computes frozen references using NumPy. SciPy supplies independent repeated-RHS solver references. Analytical cases and mathematical identities are also committed. NaN and signed infinity input and expected output values use structured special tags; non-standard JSON numbers are prohibited. tests/cases/contracts.json separately describes assertion contracts executed in child processes, so a trap cannot corrupt the rest of a numerical run.

The Zig runner embeds the same all.json via an explicitly imported fixture module and validates outputs/errors natively. zig build test runs it quietly alongside the original unit suite. Keeping that module beside the fixtures avoids parent-directory @embedFile restrictions. Mojo uses Python only to parse JSON and emit JSON Lines; native Mojo types/kernels execute the mathematics. Python transport and allocations are outside benchmarks. Fixed-size cases instantiate fixed-size types, while view cases borrow caller buffers.

A result record contains id, backend, representation, status and result or error. The verifier requires exactly one record per applicable case, matching representation, result length, values and error. Missing/duplicate/unknown IDs fail. C ABI coverage records explicitly mark functions outside the narrow ABI as N/A, rather than passing them. Native f32/f64 verification and C ABI verification share the same expected numerical values.

The original suite had 140 named numerical tests plus two import smoke tests. The named tests now execute shared case groups through the same Zig adapter. historical_inputs.json archives the original literals, array roles, assertion counts and source hashes. Representative literal input arrays are re-exercised only for operations associated with their historical test, using independently generated NumPy references; procedural constructor/callback/ownership checks map to explicit method cases. legacy_registry.json maps every historical name to shared IDs, and legacy_cases.json is a derived compact bundle for those unit adapters. All those IDs also run in native Mojo. Native allocator and borrow/lifetime checks remain language-specific mechanical tests.

The method registry includes every stable type method and rejects unregistered public methods. Numerical entries require shared cases; formatting/resource/type factories are identified separately. Compatibility aliases retain the canonical implementation's contract.

Default verification wrappers capture verbose native JSONL/compiler output in artifacts and print compact summaries. Direct native runners also default to a one-line success summary. Request machine-readable records explicitly with `zig build cases -- --jsonl` or `build/mojo-cases --jsonl`; wrappers capture those records. `zig build test` runs numerical verification silently.

## Keeping the suite pragmatic

Vector fixtures cover every supported operation/representation/precision on small
inputs. Additional dot and add cases exercise singleton, fixed-size, and borrowed
loop boundaries at lengths 1, 8, 16, 17, 31, 32 and 33. Explicit empty, zero,
nonfinite and zero-divisor cases remain separate. Large workload sizes belong in
benchmarks rather than an exhaustive correctness cross-product.

Historical replay shares the migration registry's operation mapping. It retains
one original input per operation/representation/precision within each named test,
so unrelated RHS and scratch arrays do not spawn whole numerical suites. The
original archive remains unchanged. Matrix/solver shape, conditioning, scaling,
pivoting, error, ownership and method-specific cases remain in the shared suite.

To add coverage, identify a distinct behaviour or failure mode and add a small
independent reference or analytical case. Do not multiply every existing array by
every operation. Suite files remain readable; all.json and legacy_cases.json are
derived transport bundles, and must be regenerated rather than edited by hand.
