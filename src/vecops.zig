// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const assert = std.debug.assert;
const VecStack = @import("vecstack.zig").VecStack;
const VecSlice = @import("vecslice.zig").VecSlice;

/// Computes the dot product of two fixed-size vectors `lhs` and `rhs`.
///
/// Dimensions are known at compile time via `N`.
/// Vectors are passed by value and are not mutated.
/// Performs no heap allocation.
pub fn dotStack(
    comptime N: usize,
    comptime T: type,
    lhs: VecStack(N, T),
    rhs: VecStack(N, T),
) T {
    var sum: T = 0;
    inline for (0..N) |ii| {
        sum += lhs.vec[ii] * rhs.vec[ii];
    }
    return sum;
}

/// Computes the dot product of two borrowed slice vectors `lhs` and `rhs`.
///
/// Asserts that both slices have identical length.
/// Storage is borrowed; no memory is allocated or mutated.
pub fn dotSlice(
    comptime T: type,
    lhs: VecSlice(T),
    rhs: VecSlice(T),
) T {
    assert(lhs.slice.len == rhs.slice.len);
    var sum: T = 0;
    for (0..lhs.slice.len) |ii| {
        sum += lhs.slice[ii] * rhs.slice[ii];
    }
    return sum;
}

/// Computes the squared Euclidean norm (sum of squares) of fixed-size vector `vec`.
///
/// Dimensions are known at compile time via `N`.
/// Value semantics; no heap allocation.
pub fn squaredNormStack(
    comptime N: usize,
    comptime T: type,
    vec: VecStack(N, T),
) T {
    return dotStack(N, T, vec, vec);
}

/// Computes the squared Euclidean norm (sum of squares) of borrowed slice vector `vec`.
///
/// Storage is borrowed; no heap allocation.
pub fn squaredNormSlice(
    comptime T: type,
    vec: VecSlice(T),
) T {
    return dotSlice(T, vec, vec);
}

/// Computes the Euclidean L2 norm (magnitude) of fixed-size vector `vec`.
///
/// Dimensions are known at compile time via `N`.
/// Value semantics; no heap allocation.
pub fn normStack(
    comptime N: usize,
    comptime T: type,
    vec: VecStack(N, T),
) T {
    return @sqrt(squaredNormStack(N, T, vec));
}

/// Computes the Euclidean L2 norm (magnitude) of borrowed slice vector `vec`.
///
/// Storage is borrowed; no heap allocation.
pub fn normSlice(
    comptime T: type,
    vec: VecSlice(T),
) T {
    return @sqrt(squaredNormSlice(T, vec));
}

/// Computes the maximum absolute value across all elements of fixed-size vector `vec`.
///
/// Dimensions are known at compile time via `N`. Requires `N > 0`.
/// Value semantics; no heap allocation.
pub fn maxAbsStack(
    comptime N: usize,
    comptime T: type,
    vec: VecStack(N, T),
) T {
    comptime assert(N > 0);
    var max_val: T = @abs(vec.vec[0]);
    inline for (1..N) |ii| {
        const abs_val = @abs(vec.vec[ii]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

/// Computes the maximum absolute value across all elements of borrowed slice vector `vec`.
///
/// Asserts that `vec.slice.len > 0`.
/// Storage is borrowed; no heap allocation.
pub fn maxAbsSlice(
    comptime T: type,
    vec: VecSlice(T),
) T {
    assert(vec.slice.len > 0);
    var max_val: T = @abs(vec.slice[0]);
    for (1..vec.slice.len) |ii| {
        const abs_val = @abs(vec.slice[ii]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

/// Checks whether all elements of fixed-size vector `vec` are finite (neither NaN nor Inf).
///
/// Dimensions are known at compile time via `N`.
/// Value semantics; no heap allocation.
pub fn allFiniteStack(
    comptime N: usize,
    comptime T: type,
    vec: VecStack(N, T),
) bool {
    inline for (0..N) |ii| {
        if (!std.math.isFinite(vec.vec[ii])) {
            return false;
        }
    }
    return true;
}

/// Checks whether all elements of borrowed slice vector `vec` are finite.
///
/// Storage is borrowed; no heap allocation.
pub fn allFiniteSlice(
    comptime T: type,
    vec: VecSlice(T),
) bool {
    for (0..vec.slice.len) |ii| {
        if (!std.math.isFinite(vec.slice[ii])) {
            return false;
        }
    }
    return true;
}

/// Computes the AXPY operation `y = alpha * x + y` for fixed-size vectors.
///
/// Dimensions are known at compile time via `N`.
/// Returns the updated vector by value without mutating arguments.
/// Performs no heap allocation.
pub fn axpyStack(
    comptime N: usize,
    comptime T: type,
    alpha: T,
    x: VecStack(N, T),
    y: VecStack(N, T),
) VecStack(N, T) {
    var result: VecStack(N, T) = undefined;
    inline for (0..N) |ii| {
        result.vec[ii] = alpha * x.vec[ii] + y.vec[ii];
    }
    return result;
}

/// Computes the AXPY operation `y = alpha * x + y` in place for slice vector `y`.
///
/// Asserts that `x` and `y` have identical lengths.
/// Mutates `y` in place. Performs no heap allocation.
pub fn axpySlice(
    comptime T: type,
    alpha: T,
    x: VecSlice(T),
    y: VecSlice(T),
) void {
    assert(x.slice.len == y.slice.len);
    for (0..x.slice.len) |ii| {
        y.slice[ii] = alpha * x.slice[ii] + y.slice[ii];
    }
}

test "vecops.dotStack and dotSlice" {
    const v1 = VecStack(3, f64).initSlice(&.{ 1.0, 2.0, 3.0 });
    const v2 = VecStack(3, f64).initSlice(&.{ 4.0, 5.0, 6.0 });
    const expected: f64 = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;

    const stack_result = dotStack(3, f64, v1, v2);
    try std.testing.expectEqual(expected, stack_result);

    var raw1 = [_]f64{ 1.0, 2.0, 3.0 };
    var raw2 = [_]f64{ 4.0, 5.0, 6.0 };
    const s1 = VecSlice(f64).init(&raw1);
    const s2 = VecSlice(f64).init(&raw2);
    const slice_result = dotSlice(f64, s1, s2);
    try std.testing.expectEqual(expected, slice_result);
}

test "vecops.squaredNorm and norm" {
    const v = VecStack(3, f64).initSlice(&.{ 3.0, 4.0, 0.0 });
    try std.testing.expectEqual(@as(f64, 25.0), squaredNormStack(3, f64, v));
    try std.testing.expectEqual(@as(f64, 5.0), normStack(3, f64, v));

    var raw = [_]f64{ 3.0, 4.0, 0.0 };
    const s = VecSlice(f64).init(&raw);
    try std.testing.expectEqual(@as(f64, 25.0), squaredNormSlice(f64, s));
    try std.testing.expectEqual(@as(f64, 5.0), normSlice(f64, s));
}

test "vecops.maxAbs" {
    const v = VecStack(4, f64).initSlice(&.{ -10.5, 3.2, 7.1, -1.0 });
    try std.testing.expectEqual(@as(f64, 10.5), maxAbsStack(4, f64, v));

    var raw = [_]f64{ -10.5, 3.2, 7.1, -1.0 };
    const s = VecSlice(f64).init(&raw);
    try std.testing.expectEqual(@as(f64, 10.5), maxAbsSlice(f64, s));
}

test "vecops.allFinite" {
    const v_good = VecStack(3, f64).initSlice(&.{ 1.0, -2.0, 0.0 });
    try std.testing.expect(allFiniteStack(3, f64, v_good));

    const nan_val = std.math.nan(f64);
    const inf_val = std.math.inf(f64);
    const v_nan = VecStack(3, f64).initSlice(&.{ 1.0, nan_val, 0.0 });
    const v_inf = VecStack(3, f64).initSlice(&.{ 1.0, -inf_val, 0.0 });

    try std.testing.expect(!allFiniteStack(3, f64, v_nan));
    try std.testing.expect(!allFiniteStack(3, f64, v_inf));

    var raw_good = [_]f64{ 1.0, -2.0, 0.0 };
    var raw_nan = [_]f64{ 1.0, nan_val, 0.0 };
    try std.testing.expect(allFiniteSlice(f64, VecSlice(f64).init(&raw_good)));
    try std.testing.expect(!allFiniteSlice(f64, VecSlice(f64).init(&raw_nan)));
}

test "vecops.axpy" {
    const x = VecStack(3, f64).initSlice(&.{ 1.0, 2.0, 3.0 });
    const y = VecStack(3, f64).initSlice(&.{ 4.0, 5.0, 6.0 });
    const result = axpyStack(3, f64, 2.0, x, y);

    try std.testing.expectEqual(@as(f64, 6.0), result.vec[0]);
    try std.testing.expectEqual(@as(f64, 9.0), result.vec[1]);
    try std.testing.expectEqual(@as(f64, 12.0), result.vec[2]);

    var raw_x = [_]f64{ 1.0, 2.0, 3.0 };
    var raw_y = [_]f64{ 4.0, 5.0, 6.0 };
    axpySlice(f64, 2.0, VecSlice(f64).init(&raw_x), VecSlice(f64).init(&raw_y));
    try std.testing.expectEqual(@as(f64, 6.0), raw_y[0]);
    try std.testing.expectEqual(@as(f64, 9.0), raw_y[1]);
    try std.testing.expectEqual(@as(f64, 12.0), raw_y[2]);
}
