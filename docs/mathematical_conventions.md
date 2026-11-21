# Mathematical contracts

All storage is contiguous and row-major. Fixed-size Vec and Mat have value semantics. VecView and MatView borrow caller storage. Mojo views carry an origin parameter, which keeps the owner alive; Span tracks the owner lifetime and mutability. Read-only sources use immutable Spans. No numerical kernel allocates output or scratch implicitly.

## Norms

Every vector representation and sliceops provides normL1 (sum of absolute values), normL2 (Euclidean length), and normL2Square (sum of squares). The old Zig type methods norm and sliceops.norm retain their historical squared-norm behaviour for compatibility; vecops.normStack/normSlice retain Euclidean semantics. Use the explicit new names in new code. No explicit SIMD implementations are included.

## Validation and numerical failures

Standard kernels assert programmer preconditions and require finite input. Gaussian, LU and Cholesky solvers still return numerical failures: SingularMatrix or NotPositiveDefinite. Checked variants additionally validate shapes, finite input, and the storage/permutation contracts representable by their arguments. Checked ends the name in both languages. C entrypoints always validate their boundary arguments.

Zig assertions run in Debug/ReleaseSafe and are normally disabled in ReleaseFast. Mojo assertions are controlled independently by `-D ASSERT=all` or `-D ASSERT=none`; optimisation/debug symbols alone do not enable them. Tests explicitly enable assertions. Benchmarks disable them. See the [Mojo assertion documentation](https://mojolang.org/1.1.0/docs/tools/feature-toggles/).

Standard solver entrypoints do not run dedicated NaN/Inf scans. Computing the global magnitude is part of scaling; Cholesky's diagonal validity check is intrinsic to factorisation. Invalid preconditions have no promised numerical result in unchecked kernels.

## Scaling and factor reuse

Gaussian elimination normalises A and b by max(abs(A)); division is used rather than multiplication by a potentially overflowing reciprocal. LU normalises A during factorisation, then restores U so P*A=L*U remains the public contract. Its factored solve normalises the RHS and U consistently. Cholesky uses one positive global scale, then restores L by sqrt(scale), preserving A=L*L^T and symmetry.

Gaussian/LU use partial pivoting and a relative pivot threshold `epsilon(dtype)*N` on the normalised system. This intentionally replaces the old absolute `epsilon*1e-4` threshold. Numerical singularity is scale-dependent relative to machine precision, not a proof of exact mathematical rank. Cholesky retains the existing lower-triangle interpretation; callers promise symmetry. Both representations use the same rules. Small specialised inverses retain their existing determinant/checked-determinant contracts.

## Aliasing

Exact in-place elementwise operations and AXPY are supported. Partially overlapping views are not supported. Matrix products, transpose outputs and solver input/output/scratch buffers must be disjoint; checked Zig solver adapters and both C adapters validate these restrictions. Read-only sources may share storage. Mojo rejects conflicting tracked borrows at compile time. Use the named in-place methods for native mutation. Raw pointers appear only at C ABI boundaries; both C adapters check foreign-buffer overlap before constructing views.

## NDArray compatibility

Rank and every dimension must be positive. init retains the entire caller-provided backing slice, even if longer than the shape product. dims and strides are separate allocated metadata in Zig; deinit frees only metadata. initFlat and dupe allocate element storage that the caller frees separately. Mojo retains the data/metadata distinction with Span-backed data and List metadata. Its initFlat and dupe take a mutable caller-owned List to receive allocated elements and return a view tied to that List. The List releases elements; deinit releases metadata only. Keep the storage List alive while any view borrows it.

getPlaneSlice returns the backing tail from the plane's start. getSlice retains its historical contiguous-slice interpretation. fixedPrefixView borrows data and creates independent metadata. These are compatibility contracts, not a general strided-view API. NDView is deferred.

## Precision and tolerances

Canonical verification covers f32 and f64. Existing Zig integer arithmetic/indexing remains available; integer construction/access/arithmetic is verified separately in shared i32 cases; generic floating-point algorithms require f32/f64. Shared numerical fixtures specify their own absolute and relative tolerances. Current bounded, well-conditioned regression families use atol=2e-5/rtol=2e-4 for f32 and atol=2e-12/rtol=2e-11 for f64; analytical indexing, zero and ownership cases use zero tolerance. Failure cases use symbolic codes. Controlled diagonal stress families reach condition numbers through 1e12 in f64 and 1e4 in f32. Solver cases also verify scaled residuals. Other ill-conditioned families require their own tolerance/input contract, not a global tolerance increase.

The historical Vec3Ops.cross middle component is `a.x*b.z-a.z*b.x`. This differs in sign from the conventional cross product. Both languages preserve this existing contract during Phase 0/A, and shared signed inputs explicitly test it.
