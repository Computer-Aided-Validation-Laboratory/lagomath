//! Opt-in f32/f64 SIMD. Unaligned full-width blocks and scalar tails never
//! access outside a slice. Exact aliasing is supported for elementwise output;
//! Partial input/output overlaps are unsupported; exact in-place aliases are supported.
//! Reductions accumulate SIMD lanes before folding; roundoff can differ from scalar.
const std = @import("std");
const scalar = @import("sliceops.zig");
const assert = std.debug.assert;
fn widthSIMD(comptime T: type) comptime_int {
    if (T != f32 and T != f64) @compileError("SIMD kernels require f32 or f64");
    return 32 / @sizeOf(T);
}
fn loadSIMD(comptime T: type, a: []const T, i: usize) @Vector(widthSIMD(T), T) {
    return a[i..][0..widthSIMD(T)].*;
}
fn storeSIMD(comptime T: type, a: []T, i: usize, v: @Vector(widthSIMD(T), T)) void {
    a[i..][0..widthSIMD(T)].* = v;
}
pub fn dotSIMD(comptime T: type, a: []const T, b: []const T) T {
    assert(a.len == b.len);
    const W = widthSIMD(T);
    var acc: @Vector(W, T) = @splat(0);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) acc += loadSIMD(T, a, i) * loadSIMD(T, b, i);
    var result: T = @reduce(.Add, acc);
    for (i..a.len) |j| result += a[j] * b[j];
    return result;
}
pub fn sumSIMD(comptime T: type, a: []const T) T {
    assert(a.len > 0);
    const W = widthSIMD(T);
    var acc: @Vector(W, T) = @splat(0);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) acc += loadSIMD(T, a, i);
    var result: T = @reduce(.Add, acc);
    for (a[i..]) |v| result += v;
    return result;
}
pub fn meanSIMD(comptime T: type, a: []const T) T {
    return sumSIMD(T, a) / @as(T, @floatFromInt(a.len));
}
pub fn normL1SIMD(comptime T: type, a: []const T) T {
    const W = widthSIMD(T);
    var acc: @Vector(W, T) = @splat(0);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) acc += @abs(loadSIMD(T, a, i));
    var result: T = @reduce(.Add, acc);
    for (a[i..]) |v| result += @abs(v);
    return result;
}
pub fn normL2SquareSIMD(comptime T: type, a: []const T) T {
    return dotSIMD(T, a, a);
}
pub fn normL2SIMD(comptime T: type, a: []const T) T {
    return @sqrt(normL2SquareSIMD(T, a));
}
pub fn maxAbsSIMD(comptime T: type, a: []const T) T {
    assert(a.len > 0);
    if (std.math.isNan(a[0])) return @abs(a[0]);
    const W = widthSIMD(T);
    var acc: @Vector(W, T) = @splat(0);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) acc = @max(acc, @abs(loadSIMD(T, a, i)));
    var result: T = @reduce(.Max, acc);
    for (a[i..]) |v| result = @max(result, @abs(v));
    return result;
}
pub fn allFiniteSIMD(comptime T: type, a: []const T) bool {
    const W = widthSIMD(T);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) {
        if (!@reduce(.And, @abs(loadSIMD(T, a, i)) <= @as(@Vector(W, T), @splat(std.math.floatMax(T))))) return false;
    }
    for (a[i..]) |v| if (!std.math.isFinite(v)) return false;
    return true;
}
fn extremaSIMD(comptime T: type, comptime maximum: bool, a: []const T) scalar.ValIdx(T) {
    assert(a.len > 0);
    const W = widthSIMD(T);
    var result = scalar.ValIdx(T){ .val = a[0], .idx = 0 };
    var i: usize = 1;
    while (a.len - i >= W) : (i += W) {
        const v = loadSIMD(T, a, i);
        const best: @Vector(W, T) = @splat(result.val);
        const mask = if (maximum) v > best else v < best;
        if (@reduce(.Or, mask)) {
            for (i..i + W) |j| if (if (maximum) a[j] > result.val else a[j] < result.val) {
                result = .{ .val = a[j], .idx = j };
            };
        }
    }
    for (i..a.len) |j| if (if (maximum) a[j] > result.val else a[j] < result.val) {
        result = .{ .val = a[j], .idx = j };
    };
    return result;
}
pub fn minSIMD(comptime T: type, a: []const T) scalar.ValIdx(T) {
    return extremaSIMD(T, false, a);
}
pub fn maxSIMD(comptime T: type, a: []const T) scalar.ValIdx(T) {
    return extremaSIMD(T, true, a);
}
fn binarySIMD(comptime T: type, comptime op: enum { add, sub, mul, div }, a: []const T, b: []const T, out: []T) void {
    assert(a.len == b.len and out.len == a.len);
    const W = widthSIMD(T);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) {
        const x = loadSIMD(T, a, i);
        const y = loadSIMD(T, b, i);
        storeSIMD(T, out, i, switch (op) {
            .add => x + y,
            .sub => x - y,
            .mul => x * y,
            .div => x / y,
        });
    }
    for (i..a.len) |j| out[j] = switch (op) {
        .add => a[j] + b[j],
        .sub => a[j] - b[j],
        .mul => a[j] * b[j],
        .div => a[j] / b[j],
    };
}
pub fn addSIMD(comptime T: type, a: []const T, b: []const T, out: []T) void {
    binarySIMD(T, .add, a, b, out);
}
pub fn subSIMD(comptime T: type, a: []const T, b: []const T, out: []T) void {
    binarySIMD(T, .sub, a, b, out);
}
pub fn mulSIMD(comptime T: type, a: []const T, b: []const T, out: []T) void {
    binarySIMD(T, .mul, a, b, out);
}
pub fn divSIMD(comptime T: type, a: []const T, b: []const T, out: []T) void {
    binarySIMD(T, .div, a, b, out);
}
pub fn mulScalSIMD(comptime T: type, a: []const T, alpha: T, out: []T) void {
    assert(a.len == out.len);
    const W = widthSIMD(T);
    const v: @Vector(W, T) = @splat(alpha);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) storeSIMD(T, out, i, loadSIMD(T, a, i) * v);
    for (i..a.len) |j| out[j] = a[j] * alpha;
}
pub fn fillSIMD(comptime T: type, a: []T, value: T) void {
    const W = widthSIMD(T);
    const v: @Vector(W, T) = @splat(value);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) storeSIMD(T, a, i, v);
    @memset(a[i..], value);
}
pub fn axpySIMD(comptime T: type, alpha: T, a: []const T, b: []const T, out: []T) void {
    assert(a.len == b.len and out.len == a.len);
    const W = widthSIMD(T);
    const v: @Vector(W, T) = @splat(alpha);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) storeSIMD(T, out, i, v * loadSIMD(T, a, i) + loadSIMD(T, b, i));
    for (i..a.len) |j| out[j] = alpha * a[j] + b[j];
}
pub fn swapSIMD(comptime T: type, a: []T, b: []T) void {
    assert(a.len == b.len);
    const W = widthSIMD(T);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) {
        const x = loadSIMD(T, a, i);
        const y = loadSIMD(T, b, i);
        storeSIMD(T, a, i, y);
        storeSIMD(T, b, i, x);
    }
    for (i..a.len) |j| std.mem.swap(T, &a[j], &b[j]);
}
// Products are vectorised; subtraction order matches the scalar solver exactly.
pub fn subtractProductsSIMD(comptime T: type, initial: T, a: []const T, b: []const T, divisor: T) T {
    assert(a.len == b.len);
    const W = widthSIMD(T);
    const scale: @Vector(W, T) = @splat(divisor);
    var result = initial;
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) {
        const products = loadSIMD(T, a, i) / scale * loadSIMD(T, b, i);
        inline for (0..W) |j| result -= products[j];
    }
    for (i..a.len) |j| result -= (a[j] / divisor) * b[j];
    return result;
}
pub fn addCheckedSIMD(comptime T: type, a: []const T, b: []const T, out: []T) scalar.SliceError!void {
    if (a.len != b.len or out.len != a.len) return error.LengthMismatch;
    addSIMD(T, a, b, out);
}
pub fn subCheckedSIMD(comptime T: type, a: []const T, b: []const T, out: []T) scalar.SliceError!void {
    if (a.len != b.len or out.len != a.len) return error.LengthMismatch;
    subSIMD(T, a, b, out);
}
pub fn mulCheckedSIMD(comptime T: type, a: []const T, b: []const T, out: []T) scalar.SliceError!void {
    if (a.len != b.len or out.len != a.len) return error.LengthMismatch;
    mulSIMD(T, a, b, out);
}
pub fn divCheckedSIMD(comptime T: type, a: []const T, b: []const T, out: []T) scalar.SliceError!void {
    if (a.len != b.len or out.len != a.len) return error.LengthMismatch;
    divSIMD(T, a, b, out);
}
pub fn mulScalCheckedSIMD(comptime T: type, a: []const T, alpha: T, out: []T) scalar.SliceError!void {
    if (a.len != out.len) return error.LengthMismatch;
    mulScalSIMD(T, a, alpha, out);
}
pub fn divScalSIMD(comptime T: type, a: []const T, scale: T, out: []T) void {
    assert(a.len == out.len);
    const W = widthSIMD(T);
    const v: @Vector(W, T) = @splat(scale);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) storeSIMD(T, out, i, loadSIMD(T, a, i) / v);
    for (i..a.len) |j| out[j] = a[j] / scale;
}
pub fn subtractScaledSIMD(comptime T: type, a: []const T, scale: T, out: []T) void {
    assert(a.len == out.len);
    const W = widthSIMD(T);
    const v: @Vector(W, T) = @splat(scale);
    var i: usize = 0;
    while (a.len - i >= W) : (i += W) storeSIMD(T, out, i, loadSIMD(T, out, i) - v * loadSIMD(T, a, i));
    for (i..a.len) |j| out[j] -= scale * a[j];
}
pub fn subtractStridedProductsSIMD(comptime T: type, initial: T, a: []const T, stride: usize, b: []const T) T {
    const W = widthSIMD(T);
    var result = initial;
    var i: usize = 0;
    while (b.len - i >= W) : (i += W) {
        var v: @Vector(W, T) = undefined;
        inline for (0..W) |j| v[j] = a[(i + j) * stride];
        const products = v * loadSIMD(T, b, i);
        inline for (0..W) |j| result -= products[j];
    }
    for (i..b.len) |j| result -= a[j * stride] * b[j];
    return result;
}
pub const normSIMD = normL2SquareSIMD;
pub const vecLenSIMD = normL2SIMD;
