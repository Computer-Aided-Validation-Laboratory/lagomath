// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const assert = std.debug.assert;
const Mat = @import("mattype.zig").Mat;
const Vec = @import("vectype.zig").Vec;
const MatView = @import("matviewtype.zig").MatView;
const VecView = @import("vecviewtype.zig").VecView;

/// Multiplies an M x N fixed-size matrix by an N-vector, returning an M-vector.
///
/// Dimensions M and N are known at compile time.
/// Arguments are passed by value and are not mutated.
/// Performs no heap allocation.
pub fn mulVecStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: Mat(M, N, T),
    vec: Vec(N, T),
) Vec(M, T) {
    var result: Vec(M, T) = undefined;
    inline for (0..M) |rr| {
        var sum: T = 0;
        inline for (0..N) |cc| {
            sum += mat.get(rr, cc) * vec.get(cc);
        }
        result.set(rr, sum);
    }
    return result;
}

/// Multiplies a borrowed slice matrix `mat` by `vec`, writing into `out`.
///
/// Asserts that `mat.cols_num == vec.slice.len` and `mat.rows_num == out.slice.len`.
/// Storage is borrowed; no memory is allocated.
pub fn mulVecSlice(
    comptime T: type,
    mat: MatView(T),
    vec: VecView(T),
    out: VecView(T),
) void {
    assert(mat.cols_num == vec.slice.len);
    assert(mat.rows_num == out.slice.len);
    for (0..mat.rows_num) |rr| {
        var sum: T = 0;
        for (0..mat.cols_num) |cc| {
            sum += mat.get(rr, cc) * vec.get(cc);
        }
        out.set(rr, sum);
    }
}

/// Multiplies an M x N matrix by an N x P matrix, returning an M x P matrix.
///
/// General rectangular matrix multiplication with dimensions known at compile time.
/// Passed by value; performs no heap allocation.
pub fn mulMatStack(
    comptime M: usize,
    comptime N: usize,
    comptime P: usize,
    comptime T: type,
    lhs: Mat(M, N, T),
    rhs: Mat(N, P, T),
) Mat(M, P, T) {
    var result = Mat(M, P, T).initZeros();
    inline for (0..M) |rr| {
        inline for (0..P) |cc| {
            var sum: T = 0;
            inline for (0..N) |kk| {
                sum += lhs.get(rr, kk) * rhs.get(kk, cc);
            }
            result.set(rr, cc, sum);
        }
    }
    return result;
}

/// Multiplies borrowed slice matrix `lhs` by `rhs`, writing to `out`.
///
/// Asserts matching inner dimensions and output shape.
/// Storage is borrowed; no heap allocation.
pub fn mulMatSlice(
    comptime T: type,
    lhs: MatView(T),
    rhs: MatView(T),
    out: MatView(T),
) void {
    assert(lhs.cols_num == rhs.rows_num);
    assert(lhs.rows_num == out.rows_num);
    assert(rhs.cols_num == out.cols_num);

    for (0..lhs.rows_num) |rr| {
        for (0..rhs.cols_num) |cc| {
            var sum: T = 0;
            for (0..lhs.cols_num) |kk| {
                sum += lhs.get(rr, kk) * rhs.get(kk, cc);
            }
            out.set(rr, cc, sum);
        }
    }
}

/// Computes the outer product `A = lhs * rhs^T` of two fixed-size vectors.
///
/// Dimensions M and N are known at compile time.
/// Result is returned by value as an M x N matrix.
/// Performs no heap allocation.
pub fn outerStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    lhs: Vec(M, T),
    rhs: Vec(N, T),
) Mat(M, N, T) {
    var result: Mat(M, N, T) = undefined;
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            result.set(rr, cc, lhs.get(rr) * rhs.get(cc));
        }
    }
    return result;
}

/// Computes the outer product `out = lhs * rhs^T` for borrowed slices.
///
/// Asserts that `out.rows_num == lhs.slice.len` and `out.cols_num == rhs.slice.len`.
/// Writes directly to caller-provided `out`. Performs no heap allocation.
pub fn outerSlice(
    comptime T: type,
    lhs: VecView(T),
    rhs: VecView(T),
    out: MatView(T),
) void {
    assert(out.rows_num == lhs.slice.len);
    assert(out.cols_num == rhs.slice.len);
    for (0..lhs.slice.len) |rr| {
        for (0..rhs.slice.len) |cc| {
            out.set(rr, cc, lhs.get(rr) * rhs.get(cc));
        }
    }
}

/// Accumulates scaled outer product `mat += alpha * lhs * rhs^T` for fixed-size types.
///
/// Dimensions M and N are compile-time constants.
/// Does not construct a temporary matrix; accumulates directly into returned value.
/// Performs no heap allocation.
pub fn outerAddStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    alpha: T,
    lhs: Vec(M, T),
    rhs: Vec(N, T),
    mat: Mat(M, N, T),
) Mat(M, N, T) {
    var result = mat;
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            const current = result.get(rr, cc);
            result.set(rr, cc, current + alpha * lhs.get(rr) * rhs.get(cc));
        }
    }
    return result;
}

/// Accumulates scaled outer product `mat += alpha * lhs * rhs^T` in place for slices.
///
/// Asserts matching dimensions between vectors and target matrix.
/// Directly mutates `mat` without temporary storage. Performs no heap allocation.
pub fn outerAddSlice(
    comptime T: type,
    alpha: T,
    lhs: VecView(T),
    rhs: VecView(T),
    mat: MatView(T),
) void {
    assert(mat.rows_num == lhs.slice.len);
    assert(mat.cols_num == rhs.slice.len);
    for (0..lhs.slice.len) |rr| {
        for (0..rhs.slice.len) |cc| {
            const current = mat.get(rr, cc);
            mat.set(rr, cc, current + alpha * lhs.get(rr) * rhs.get(cc));
        }
    }
}

/// Computes the Frobenius norm of fixed-size matrix `mat`.
///
/// Reuses flat contiguous array elements directly.
/// Value semantics; performs no heap allocation.
pub fn frobeniusNormStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: Mat(M, N, T),
) T {
    var sum: T = 0;
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            sum += mat.mat[rr][cc] * mat.mat[rr][cc];
        }
    }
    return @sqrt(sum);
}

/// Computes the Frobenius norm of borrowed slice matrix `mat`.
///
/// Reuses flat slice storage directly. Performs no heap allocation.
pub fn frobeniusNormSlice(
    comptime T: type,
    mat: MatView(T),
) T {
    var sum: T = 0;
    for (0..mat.slice.len) |ii| {
        sum += mat.slice[ii] * mat.slice[ii];
    }
    return @sqrt(sum);
}

/// Computes the maximum absolute value across all elements of fixed-size matrix `mat`.
///
/// Dimensions are known at compile time. Requires M > 0 and N > 0.
/// Value semantics; performs no heap allocation.
pub fn maxAbsStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: Mat(M, N, T),
) T {
    comptime assert(M > 0 and N > 0);
    var max_val: T = @abs(mat.mat[0][0]);
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            const abs_val = @abs(mat.mat[rr][cc]);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
    }
    return max_val;
}

/// Computes the maximum absolute value across all elements of borrowed slice matrix `mat`.
///
/// Asserts `mat.slice.len > 0`.
/// Performs no heap allocation.
pub fn maxAbsSlice(
    comptime T: type,
    mat: MatView(T),
) T {
    assert(mat.slice.len > 0);
    var max_val: T = @abs(mat.slice[0]);
    for (1..mat.slice.len) |ii| {
        const abs_val = @abs(mat.slice[ii]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

/// Checks whether all elements of fixed-size matrix `mat` are finite.
///
/// Value semantics; performs no heap allocation.
pub fn allFiniteStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: Mat(M, N, T),
) bool {
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            if (!std.math.isFinite(mat.mat[rr][cc])) {
                return false;
            }
        }
    }
    return true;
}

/// Checks whether all elements of borrowed slice matrix `mat` are finite.
///
/// Performs no heap allocation.
pub fn allFiniteSlice(
    comptime T: type,
    mat: MatView(T),
) bool {
    for (0..mat.slice.len) |ii| {
        if (!std.math.isFinite(mat.slice[ii])) {
            return false;
        }
    }
    return true;
}

test "matops.mulVecStack and mulVecSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.matops.matops_mulVecStack_and_mulVecSlice");
}

test "matops.mulMatStack rectangular and mulMatSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.matops.matops_mulMatStack_rectangular_and_mulMatSlice");
}

test "matops.outer and outerAdd" {
    try @import("case_runner.zig").verifyLegacy("legacy.matops.matops_outer_and_outerAdd");
}

test "matops.frobeniusNorm and maxAbs" {
    try @import("case_runner.zig").verifyLegacy("legacy.matops.matops_frobeniusNorm_and_maxAbs");
}

test "matops.allFinite" {
    try @import("case_runner.zig").verifyLegacy("legacy.matops.matops_allFinite");
}

pub const mulVecMat = mulVecStack;

pub const mulVecMatView = mulVecSlice;

pub const mulMatMat = mulMatStack;

pub const mulMatMatView = mulMatSlice;

pub const outerMat = outerStack;

pub const outerMatView = outerSlice;

pub const outerAddMat = outerAddStack;

pub const outerAddMatView = outerAddSlice;

pub const frobeniusNormMat = frobeniusNormStack;

pub const frobeniusNormMatView = frobeniusNormSlice;

pub const maxAbsMat = maxAbsStack;

pub const maxAbsMatView = maxAbsSlice;

pub const allFiniteMat = allFiniteStack;

pub const allFiniteMatView = allFiniteSlice;
