// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const assert = std.debug.assert;
const Vec = @import("vectype.zig").Vec;
const VecView = @import("vecviewtype.zig").VecView;

/// Computes the dot product of two fixed-size vectors `lhs` and `rhs`.
///
/// Dimensions are known at compile time via `N`.
/// Vectors are passed by value and are not mutated.
/// Performs no heap allocation.
pub fn dotStack(
    comptime N: usize,
    comptime T: type,
    lhs: Vec(N, T),
    rhs: Vec(N, T),
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
    lhs: VecView(T),
    rhs: VecView(T),
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
    vec: Vec(N, T),
) T {
    return dotStack(N, T, vec, vec);
}

/// Computes the squared Euclidean norm (sum of squares) of borrowed slice vector `vec`.
///
/// Storage is borrowed; no heap allocation.
pub fn squaredNormSlice(
    comptime T: type,
    vec: VecView(T),
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
    vec: Vec(N, T),
) T {
    return @sqrt(squaredNormStack(N, T, vec));
}

/// Computes the Euclidean L2 norm (magnitude) of borrowed slice vector `vec`.
///
/// Storage is borrowed; no heap allocation.
pub fn normSlice(
    comptime T: type,
    vec: VecView(T),
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
    vec: Vec(N, T),
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
    vec: VecView(T),
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
    vec: Vec(N, T),
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
    vec: VecView(T),
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
    x: Vec(N, T),
    y: Vec(N, T),
) Vec(N, T) {
    var result: Vec(N, T) = undefined;
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
    x: VecView(T),
    y: VecView(T),
) void {
    assert(x.slice.len == y.slice.len);
    for (0..x.slice.len) |ii| {
        y.slice[ii] = alpha * x.slice[ii] + y.slice[ii];
    }
}

test "vecops.dotStack and dotSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecops.vecops_dotStack_and_dotSlice");
}

test "vecops.squaredNorm and norm" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecops.vecops_squaredNorm_and_norm");
}

test "vecops.maxAbs" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecops.vecops_maxAbs");
}

test "vecops.allFinite" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecops.vecops_allFinite");
}

test "vecops.axpy" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecops.vecops_axpy");
}

pub const dotVec = dotStack;

pub const dotVecView = dotSlice;

pub const squaredNormVec = squaredNormStack;

pub const squaredNormVecView = squaredNormSlice;

pub const normVec = normStack;

pub const normVecView = normSlice;

pub const maxAbsVec = maxAbsStack;

pub const maxAbsVecView = maxAbsSlice;

pub const allFiniteVec = allFiniteStack;

pub const allFiniteVecView = allFiniteSlice;

pub const axpyVec = axpyStack;

pub const axpyVecView = axpySlice;

pub fn normL1Vec(comptime N: usize, comptime T: type, v: Vec(N, T)) T {
    return v.normL1();
}

pub fn normL2Vec(comptime N: usize, comptime T: type, v: Vec(N, T)) T {
    return v.normL2();
}

pub fn normL2SquareVec(comptime N: usize, comptime T: type, v: Vec(N, T)) T {
    return v.normL2Square();
}

pub fn normL1VecView(comptime T: type, v: VecView(T)) T {
    return v.normL1();
}

pub fn normL2VecView(comptime T: type, v: VecView(T)) T {
    return v.normL2();
}

pub fn normL2SquareVecView(comptime T: type, v: VecView(T)) T {
    return v.normL2Square();
}
