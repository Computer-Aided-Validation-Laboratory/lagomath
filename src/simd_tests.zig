const std = @import("std");
const m = @import("mattype.zig").Mat;
const v = @import("vectype.zig").Vec;
const so = @import("linsolve_simd.zig");
const mm = @import("matops_simd.zig");
test "SIMD solvers match an analytic SPD solution" {
    var a = m(9, 9, f64).initFill(0.125);
    for (0..9) |i| a.set(i, i, 10);
    const b = v(9, f64).initOnes();
    const x = try so.solveLUMatSIMD(9, f64, a, b);
    const y = try so.solveCholeskyMatSIMD(9, f64, a, b);
    const z = try so.solveGaussMatSIMD(9, f64, a, b);
    const product = mm.mulMatMatSIMD(9, 9, 9, f64, a, a);
    const trans = mm.transposeMatSIMD(9, 9, f64, product);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), x.vec[0], 1e-14);
    try std.testing.expectApproxEqAbs(x.vec[0], y.vec[0], 1e-14);
    try std.testing.expectApproxEqAbs(x.vec[0], z.vec[0], 1e-14);
    try std.testing.expectEqual(product.mat[0][1], trans.mat[1][0]);
}

const core = @import("sliceops_simd.zig");
const plain = @import("sliceops.zig");
test "SIMD unaligned blocks tails in-place and sentinels" {
    inline for (.{ f32, f64 }) |T| {
        for ([_]usize{ 0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63 }) |n| {
            var pa: [66]T = @splat(99);
            var pb: [66]T = @splat(99);
            var po: [66]T = @splat(99);
            const a = pa[1 .. n + 1];
            const b = pb[1 .. n + 1];
            const out = po[1 .. n + 1];
            for (0..n) |i| {
                a[i] = @as(T, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 8;
                b[i] = @as(T, @floatFromInt(i % 7 + 1)) / 4;
            }
            try std.testing.expectEqual(plain.dot(T, a, b), core.dotSIMD(T, a, b));
            try std.testing.expectEqual(plain.normL1(T, a), core.normL1SIMD(T, a));
            try std.testing.expectEqual(plain.normL2Square(T, a), core.normL2SquareSIMD(T, a));
            try std.testing.expect(core.allFiniteSIMD(T, a));
            if (n > 0) {
                try std.testing.expectEqual(plain.sum(T, a), core.sumSIMD(T, a));
                try std.testing.expectEqualDeep(plain.max(T, a), core.maxSIMD(T, a));
                try std.testing.expectEqualDeep(plain.min(T, a), core.minSIMD(T, a));
            }
            core.addSIMD(T, a, b, out);
            for (0..n) |i| try std.testing.expectEqual(a[i] + b[i], out[i]);
            core.subSIMD(T, a, b, out);
            for (0..n) |i| try std.testing.expectEqual(a[i] - b[i], out[i]);
            core.mulSIMD(T, a, b, out);
            for (0..n) |i| try std.testing.expectEqual(a[i] * b[i], out[i]);
            core.divSIMD(T, a, b, out);
            for (0..n) |i| try std.testing.expectEqual(a[i] / b[i], out[i]);
            core.axpySIMD(T, 0.5, a, b, out);
            for (0..n) |i| try std.testing.expectEqual(0.5 * a[i] + b[i], out[i]);
            core.mulScalSIMD(T, out, 2, out);
            for (0..n) |i| try std.testing.expectEqual(a[i] + 2 * b[i], out[i]);
            core.fillSIMD(T, out, -2);
            for (out) |value| try std.testing.expectEqual(@as(T, -2), value);
            try std.testing.expectEqual(@as(T, 99), po[0]);
            for (po[n + 1 ..]) |value| try std.testing.expectEqual(@as(T, 99), value);
        }
        var nonfinite: [17]T = @splat(1);
        for (0..17) |i| {
            nonfinite[i] = std.math.nan(T);
            try std.testing.expect(!core.allFiniteSIMD(T, &nonfinite));
            nonfinite[i] = 1;
            nonfinite[i] = std.math.inf(T);
            try std.testing.expect(!core.allFiniteSIMD(T, &nonfinite));
            nonfinite[i] = 1;
        }
        nonfinite[0] = std.math.nan(T);
        try std.testing.expect(std.math.isNan(core.maxAbsSIMD(T, &nonfinite)));
        try std.testing.expect(std.math.isNan(core.minSIMD(T, &nonfinite).val));
        try std.testing.expectEqual(@as(usize, 0), core.maxSIMD(T, &nonfinite).idx);
    }
}
test "SIMD rectangular products and transpose preserve output guards" {
    inline for (.{ f32, f64 }) |T| {
        var a: [5 * 9]T = undefined;
        var b: [9 * 11]T = undefined;
        var guarded: [5 * 11 + 2]T = @splat(99);
        for (&a, 0..) |*value, i| value.* = @as(T, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 8;
        for (&b, 0..) |*value, i| value.* = @as(T, @floatFromInt(i % 7 + 1)) / 4;
        const out = guarded[1 .. guarded.len - 1];
        mm.mulMatSIMD(T, 5, 9, 11, &a, &b, out);
        for (0..5) |r| for (0..11) |c| {
            var expected: T = 0;
            for (0..9) |k| expected += a[r * 9 + k] * b[k * 11 + c];
            try std.testing.expectEqual(expected, out[r * 11 + c]);
        };
        try std.testing.expectEqual(@as(T, 99), guarded[0]);
        try std.testing.expectEqual(@as(T, 99), guarded[guarded.len - 1]);
        var original: [9 * 17]T = undefined;
        for (&original, 0..) |*value, i| value.* = @floatFromInt(i);
        var trans: [9 * 17 + 2]T = @splat(99);
        mm.transposeSIMD(T, 9, 17, &original, trans[1..154]);
        for (0..9) |r| for (0..17) |c| {
            try std.testing.expectEqual(original[r * 17 + c], trans[1 + c * 9 + r]);
        };
        try std.testing.expectEqual(@as(T, 99), trans[0]);
        try std.testing.expectEqual(@as(T, 99), trans[154]);
        var x: [2]T = .{ 1, 2 };
        var y: [1]T = .{3};
        var z: [2]T = .{ 99, 99 };
        try std.testing.expectError(error.LengthMismatch, core.addCheckedSIMD(T, &x, &y, &z));
        try std.testing.expectEqual(@as(T, 99), z[0]);
    }
}
test "SIMD NDArray arithmetic uses the existing backing-storage contract" {
    const NDArray = @import("ndarraytype.zig").NDArray;
    const ops = @import("ndarrayops_simd.zig");
    inline for (.{ f32, f64 }) |T| {
        var aa: [15]T = @splat(2);
        var bb: [15]T = @splat(4);
        var out: [15]T = @splat(0);
        var a = try NDArray(T).init(std.testing.allocator, &aa, &.{ 3, 5 });
        defer a.deinit(std.testing.allocator);
        var b = try NDArray(T).init(std.testing.allocator, &bb, &.{ 3, 5 });
        defer b.deinit(std.testing.allocator);
        var dst = try NDArray(T).init(std.testing.allocator, &out, &.{ 3, 5 });
        defer dst.deinit(std.testing.allocator);
        ops.addNDArraySIMD(T, &a, &b, &dst);
        for (out) |value| try std.testing.expectEqual(@as(T, 6), value);
        ops.subNDArraySIMD(T, &a, &b, &dst);
        for (out) |value| try std.testing.expectEqual(@as(T, -2), value);
        ops.mulElemWiseNDArraySIMD(T, &a, &b, &dst);
        for (out) |value| try std.testing.expectEqual(@as(T, 8), value);
        ops.divElemWiseNDArraySIMD(T, &a, &b, &dst);
        for (out) |value| try std.testing.expectEqual(@as(T, 0.5), value);
        ops.fillNDArraySIMD(T, &dst, 3);
        ops.mulScalNDArraySIMD(T, &dst, -2);
        for (out) |value| try std.testing.expectEqual(@as(T, -6), value);
    }
}
test "SIMD across independent short vectors and permutation gathers" {
    const vectors = @import("vecops_simd.zig");
    const rows = @import("rowops_simd.zig");
    const View = @import("vecviewtype.zig").VecView;
    inline for (.{ f32, f64 }) |T| {
        var a: [17 * 3]T = undefined;
        var b: [17 * 3]T = undefined;
        var out: [17]T = undefined;
        for (&a, 0..) |*value, i| value.* = @as(T, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
        for (&b, 0..) |*value, i| value.* = @as(T, @floatFromInt(i % 7 + 1)) / 4;
        vectors.dotBatchSIMD(3, T, &a, &b, &out);
        for (out, 0..) |value, i| {
            const want = a[i * 3] * b[i * 3] + a[i * 3 + 1] * b[i * 3 + 1] + a[i * 3 + 2] * b[i * 3 + 2];
            try std.testing.expectEqual(want, value);
        }
        var perm: [17]usize = undefined;
        for (&perm, 0..) |*value, i| value.* = 16 - i;
        var gathered: [17]T = undefined;
        rows.applyPermutationVecViewSIMD(T, &perm, View(T).init(&out), View(T).init(&gathered));
        for (gathered, 0..) |value, i| try std.testing.expectEqual(out[16 - i], value);
    }
}
test "SIMD matrix subviews preserve storage outside their bounds" {
    const View = @import("matviewtype.zig").MatView;
    inline for (.{ f32, f64 }) |T| {
        var a: [5]T = .{ 1, 2, 3, 4, 99 };
        var b: [5]T = .{ 1, 0, 0, 1, 88 };
        var out: [5]T = @splat(77);
        const av = View(T).init(a[0..4], 2, 2);
        const bv = View(T).init(b[0..4], 2, 2);
        const dst = View(T).init(out[0..4], 2, 2);
        mm.mulMatMatViewSIMD(T, av, bv, dst);
        for (0..4) |i| try std.testing.expectEqual(a[i], out[i]);
        try std.testing.expectEqual(@as(T, 77), out[4]);
        try std.testing.expectEqual(@as(T, 4), mm.maxAbsMatViewSIMD(T, av));
        mm.fillMatViewSIMD(T, dst, 2);
        for (out[0..4]) |value| try std.testing.expectEqual(@as(T, 2), value);
        try std.testing.expectEqual(@as(T, 77), out[4]);
    }
}
