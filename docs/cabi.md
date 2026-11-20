# C ABI version 1

include/lagomath.h is authoritative. Python cffi ABI declarations are derived directly from it by removing preprocessor/C++ linkage lines. The tests compile a C sizeof/alignof/offsetof probe for the exposed lm_mat22_f64 POD and compare it with each producer and cffi.

The optional libraries are liblagomath_zig.so and liblagomath_mojo.so. Exported f32/f64 functions cover dot, matvec, matmul, AXPY, Gaussian solve and Cholesky solve. A compact f64 matrix transpose demonstrates POD layout. All numerical output uses caller storage; scalars use out parameters. Dimensions/counts are signed 64-bit integers, in elements. Matrices are contiguous row-major. ABI version/layout queries return uint32_t; statuses return int32_t.

Statuses: 0 OK, 1 invalid argument, 2 shape/scratch mismatch, 3 singular, 4 not positive definite, 5 nonfinite, 6 invalid aliasing. Solves require N*N+N floating-point scratch elements; there is no allocation. Square dimensions must be positive. Negative counts and unsupported counts above 1e9 are rejected before pointer access. Dot permits null inputs for length zero but still requires its scalar result pointer. Empty AXPY permits null pointers. Other null pointers are rejected. Actual pointer validity, buffer capacity, alignment and dtype remain caller obligations.

AXPY permits exactly identical source/destination pointers, and rejects partial overlap. Matrix products and solves reject overlapping outputs/scratch/inputs. Input-input aliases are allowed. POD transpose copies the input first and supports exact in-place use. C boundary validation always runs, independent of optimisation mode.

Build and test from the repository root:

```sh
zig build cabi -Doptimize=Debug
zig build cabi -Doptimize=ReleaseFast --prefix build/cabi-release
.venv/bin/python tools/build_mojo.py
.venv/bin/python -m pytest -q tests/python/test_cabi.py
```

The pytest session builds Debug and ReleaseFast Zig libraries, O0/assertion-enabled and O3 Mojo libraries, and validates each in a fresh process. For an already built library, `.venv/bin/python tests/python/cabi_worker.py /absolute/path/to/library.so` performs the same cffi checks without rebuilding. Machine-local coverage reports are written beneath build/.

The verified binary target is Linux x86-64. No Windows/macOS Mojo binary support or Python extension wheel is claimed. The relocatable Mojo library ships alongside build/.libs containing the three compiler runtime libraries; patchelf sets relative RPATHs. A test copies this bundle and loads it in an isolated Python process with no Mojo environment variables. This verifies the bundled runtime path, not universal binary portability. Runtime redistribution/licensing requires separate packaging review before publishing binary artefacts.
