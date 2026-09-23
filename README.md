# Lagomath

Lightweight matrix and vector mathematics in Zig. For rabbits.

Lagomath is a source-only Zig library for reusable numerical infrastructure:

- fixed-size vectors and matrices (`VecStack`, `MatStack`);
- borrowed vector and matrix views (`VecSlice`, `MatSlice`);
- owned N-dimensional array metadata and storage helpers (`NDArray`); and
- generic contiguous-slice operations (`sliceops`).

It depends only on Zig's standard library and supports Zig 0.16.x. Lagomath
is distributed as Zig source. Consumers import the `lagomath` module and Zig
compiles the required source as part of the consuming compilation. Lagomath
does not require linking against a separate precompiled library.

## Add Lagomath to a project

For a released version, run this from the root of the consuming project. It
pins the dependency to the `2026.9.0` release tag and records the content hash
in `build.zig.zon`:

```sh
zig fetch --save-exact \
  https://github.com/Computer-Aided-Validation-Laboratory/lagomath/archive/refs/tags/2026.9.0.tar.gz
```

`zig fetch` writes the dependency's URL and Zig-computed hash for you. Commit
the resulting `build.zig.zon` change. The hash is the immutable package
identity; do not manually replace it. To upgrade later, rerun the command with
the desired release tag, for example `2026.10.0`, then test the consumer.

The command requires that the corresponding GitHub release tag has been
published. Until then, use a local path dependency for simultaneous Lagomath
and consumer development:

```zig
.dependencies = .{
    .lagomath = .{ .path = "../lagomath" },
},
```

Then obtain and expose its module from your `build.zig`:

```zig
const lagomath_dependency = b.dependency("lagomath", .{
    .target = target,
    .optimize = optimize,
});

const executable = b.addExecutable(.{
    .name = "my-program",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{
                .name = "lagomath",
                .module = lagomath_dependency.module("lagomath"),
            },
        },
    }),
});
_ = executable;
```

Your Zig source imports only the public package module:

```zig
const lagomath = @import("lagomath");
```

Use the version-pinned release dependency for normal consumers; the local path
form is only for simultaneous development.

## Tutorial

Fixed-size values are stack allocated and use compile-time dimensions. This
small matrix-vector product allocates nothing:

```zig
const lagomath = @import("lagomath");

const Mat22 = lagomath.MatStack(2, 2, f64);
const Vec2 = lagomath.VecStack(2, f64);

const matrix = Mat22.initRows(.{ .{ 1, 2 }, .{ 3, 4 } });
const vector = Vec2.initSlice(&.{ 5, 6 });
const product = matrix.mulVec(vector); // { 17, 39 }
```

For existing contiguous storage, make a non-owning slice view:

```zig
var values = [_]f64{ 1, 2, 3, 4 };
const matrix = lagomath.MatSlice(f64).init(&values, 2, 2);
const diagonal_sum = matrix.trace(); // 5
```

`NDArray.initFlat` allocates its element storage and its shape metadata. The
caller owns both and must free the element slice and call `deinit` with the
same allocator:

```zig
const std = @import("std");
const lagomath = @import("lagomath");

fn makeArray(alloc: std.mem.Allocator) !void {
    var array = try lagomath.NDArray(f64).initFlat(alloc, &.{ 2, 2 });
    defer alloc.free(array.slice);
    defer array.deinit(alloc);

    array.set(&.{ 1, 0 }, 7);
    const value = array.get(&.{ 1, 0 });
    _ = value;
}
```

For reusable contiguous kernels, use `sliceops` with caller-provided output
storage:

```zig
var output = [_]f64{ 0, 0, 0 };
try lagomath.sliceops.add(f64, &.{ 1, 2, 3 }, &.{ 4, 5, 6 }, &output);
// output is { 5, 7, 9 }
```

## Development and validation

Run the complete Lagomath unit suite, including local test blocks from every
public module:

```sh
zig build test
zig build test -Doptimize=ReleaseFast
```

Run the small user-facing example:

```sh
zig build run-example
```

The independent package-boundary integration project is in
[`integration/consumer`](integration/consumer). It has its own build files,
declares Lagomath as a normal dependency, and imports only
`@import("lagomath")`:

```sh
cd integration/consumer
zig build run
zig build run -Doptimize=ReleaseFast
```

Before contributing, format and run all local checks:

```sh
zig fmt --check .
zig build test -Doptimize=Debug
zig build test -Doptimize=ReleaseFast
zig build run-example
cd integration/consumer && zig build run
```

Continuous integration runs these formatting, unit-test, example, and
consumer-integration checks on pull requests and pushes to `main`.
