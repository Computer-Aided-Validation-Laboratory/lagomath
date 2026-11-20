# Lagomath

CPU vector, matrix and small-system mathematics in Zig and Mojo. For rabbits.

The development API uses `Vec`, `Mat`, `VecView`, `MatView` and `NDArray`. Fixed-size values use inline storage; views borrow contiguous caller storage. Numerical kernels use caller-owned output/scratch and have no hidden heap allocation. Gaussian elimination, LU and Cholesky support factor reuse and global magnitude scaling. Explicit SIMD, GPU, NDView, QR and statistics are outside this milestone.

Zig 0.16.0 and Mojo 1.2.0 are the verified compilers. Zig remains a source-only standard-library dependency: Python, Mojo and the optional C libraries are not required by a Zig consumer. The verified Mojo/C binary environment is Linux x86-64.

## Import with one namespace

Zig:

```zig
const lm = @import("lagomath");
const Vec3 = lm.Vec(3, f64);
const v = Vec3.initSlice(&.{ 3, 4, 0 });
const length = v.normL2();
```

Mojo:

```mojo
import lagomath as lm

var v = lm.Vec[3, DType.float64]()
v.set(0, 3)
v.set(1, 4)
var length = v.normL2()
```

Use `lm.vecops`, `lm.matops`, `lm.rowops` and `lm.linsolve` for operations. Both languages use the same canonical names, including the `Checked` suffix. All vector representations provide `normL1`, `normL2` and `normL2Square`.

Standard kernels assert programmer preconditions and assume finite input. Standard solvers return legitimate numerical failures; `Checked` variants additionally validate input/storage contracts. The optional C ABI always validates its boundary arguments. See [mathematical contracts](docs/mathematical_conventions.md).

## Zig source dependency

For local development, declare a path dependency in the consumer's build.zig.zon:

```zig
.dependencies = .{
    .lagomath = .{ .path = "../lagomath" },
},
```

Expose `b.dependency("lagomath", .{ .target = target, .optimize = optimize }).module("lagomath")` through the consumer module's imports. [integration/consumer](integration/consumer) is a complete separate build using only the public package entrypoint. Normal release consumers should pin a published archive with `zig fetch --save-exact`; this development API has not been published as a new release tag.

```sh
zig build test -Doptimize=Debug
zig build test -Doptimize=ReleaseFast
zig build run-example
zig build run -Doptimize=Debug --build-file integration/consumer/build.zig
zig build run -Doptimize=ReleaseFast --build-file integration/consumer/build.zig
```

root.zig remains in src/. Type definitions are now vectype.zig, mattype.zig, vecviewtype.zig, matviewtype.zig and ndarraytype.zig. Old source filenames and VecStack/MatStack/VecSlice/MatSlice names remain compatibility shims. Existing convenience namespaces remain available. Ambiguous legacy norm names retain their original semantics; use the explicit norm names for new code.

## Development environment and shared verification

```sh
uv sync --locked
.venv/bin/mojo --version
.venv/bin/python tests/python/validate_cases.py
.venv/bin/python tools/audit_api.py --check
.venv/bin/python -m pytest -q tests/python
```

pytest builds native runners and both C libraries, then verifies the same frozen JSON cases in Zig Debug/ReleaseFast, Mojo O0/O3 and all four C ABI build configurations. Child processes isolate assertion and library failures. Numerical inputs/answers, integrity metadata and regeneration tools live under tests/. To deliberately regenerate references:

```sh
.venv/bin/python tests/python/generate_regression.py
.venv/bin/python tools/audit_api.py
```

Successful native commands and wrappers print compact summaries; detailed records require --jsonl or --verbose and are captured by wrappers in build/logs or JSON artifacts. Review changed fixtures and the API manifest before committing them. [The test contract](docs/shared_test_contract.md) explains the current migration status, including the historical test migration ledger and enforced method coverage.

## Paired examples

The nine numbered examples teach types, borrowed views, NDArray, vector operations, matrix operations, row operations, Gaussian elimination, LU reuse and Cholesky reuse:

```sh
zig build examples
zig build run-ex07_GAUSS
.venv/bin/python tools/run_examples.py
.venv/bin/mojo -I src/mojo examples/mojo/ex07_GAUSS.mojo
.venv/bin/mojo -I src/mojo integration/mojo-consumer/main.mojo
```

Every example imports Lagomath as `lm`. Native Mojo consumers use `-I src/mojo`; no Python extension wheel is required. See [Mojo build details](docs/mojo_build.md).

NDArray retains Riley's split ownership and historical slicing contracts. In Zig, free initFlat/dupe element storage separately from deinit metadata. Every dimension must be positive. Mojo views use Span; allocating helpers receive a caller-owned List and borrow its lifetime. The NDARRAY examples demonstrate the contract.

## C libraries and scalar performance

```sh
zig build cabi -Doptimize=ReleaseFast
.venv/bin/python tools/build_mojo.py
.venv/bin/python -m pytest -q tests/python/test_cabi.py
.venv/bin/python benchmarks/python/run_benchmarks.py --label zig-current
.venv/bin/python benchmarks/python/run_benchmarks.py --backend mojo --label mojo-current
```

Use `--samples 2 --runs 2` for benchmark smoke runs. Raw runs and coverage reports remain machine-local under ignored build/ and benchmarks/results/. No explicit SIMD implementations are included; normal compiler auto-vectorisation is permitted. Initial measurements are recorded in [the scalar baseline](dev/scalar_baseline.md).

See [implementation status](docs/implementation_status.md), [the C ABI](docs/cabi.md), [benchmark controls](docs/benchmarking.md), [architecture decisions](docs/architecture.md), and [the machine-readable API registry](docs/api_manifest.json). Mojo libraries require their bundled runtime .libs directory. Universal binary distribution and Python wheels are not claimed.
