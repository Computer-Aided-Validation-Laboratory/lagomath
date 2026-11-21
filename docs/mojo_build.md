# Mojo source use and builds

The verified toolchain is Mojo 1.2.0 (2016d0cd), installed in the uv-managed Python 3.14 environment. uv.lock pins development/reference tooling. Native types use std.collections.Array for fixed inline storage and origin-carrying std.collections.Span for borrowed views. There are no explicitly vectorised kernels, GPU operations or MAX dependencies.

Use `import lagomath as lm` and pass `-I src/mojo`. Types are lm.Vec, lm.Mat, lm.VecView, lm.MatView and lm.NDArray. Operations are namespaced identically where applicable: lm.vecops, lm.matops, lm.rowops, lm.linsolve. The flat entry module is src/mojo/lagomath.mojo; no __init__.mojo is needed for this verified import arrangement.

```sh
uv sync --locked
.venv/bin/python tools/build_mojo.py
.venv/bin/mojo -I src/mojo integration/mojo-consumer/main.mojo
.venv/bin/python tools/run_examples.py
```

Native JSON runners use Python JSON transport; native kernels and the C libraries do not call Python for mathematics. A source consumer need not use that runner. tools/build_mojo.py builds assertion-enabled O0/O3 runners and O0/O3 C libraries. The C libraries depend on the bundled Mojo runtimes; see cabi.md. Keep the library and .libs directory together when relocating.

Borrowed origins keep the storage owner alive through a view's last use. VecView and MatView receive Span values; NDArray.init receives a Span plus dimensions. Construct native views with `lm.VecView(Span(storage))` or `lm.MatView(Span(storage), rows, cols)`. Use `.as_imm()` for read-only Spans. Native Checked calls rely on the borrow checker to reject overlapping tracked input/output/scratch. The C ABI separately validates foreign addresses before wrapping them in Spans. There are no raw-pointer fields, casts, allocation/free calls or indexing in native Mojo code.

For allocated NDArray data, pass a caller-owned `List` to `initFlat(storage, dims)` or `array.dupe(storage)`. The returned view borrows that List. Metadata remains independent; deinit does not free the caller's elements. No untracked manual-allocation exception is needed.

Successful build/example wrappers emit one-line summaries. Compiler diagnostics and example output are saved in build/logs; failures show a bounded diagnostic tail with the log path.
