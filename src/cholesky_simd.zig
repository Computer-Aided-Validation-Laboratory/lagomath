const simd = @import("sliceops_simd.zig");
pub const CholeskyStackSIMD = @import("cholesky.zig").CholeskyStack;
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
const matops = @import("matops_simd.zig");
const vecops = @import("vecops_simd.zig");
const contracts = @import("contracts.zig");
const SolveError = @import("linsolve.zig").SolveError;

/// Stack-allocated factorisation representation storing the lower triangular Cholesky factor.
/// Computes the Cholesky factorisation `A = L L^T` for a symmetric positive-definite matrix.
///
/// `a` is an N x N fixed-size matrix passed by value.
/// Returns `CholeskyStackSIMD(N, T)` containing the lower triangular factor `L`.
/// Performs no heap allocation.
///
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.NotPositiveDefinite` if `A` is not positive definite.
pub fn factorCholeskyStackSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
) SolveError!CholeskyStackSIMD(N, T) {
    comptime assert(N > 0);
    const scale = matops.maxAbsMatSIMD(N, N, T, a);
    if (scale == 0) return error.NotPositiveDefinite;
    var result = CholeskyStackSIMD(N, T){
        .lower = Mat(N, N, T).initZeros(),
    };

    for (0..N) |ii| {
        for (0..(ii + 1)) |jj| {
            var sum = a.get(ii, jj) / scale;
            sum = simd.subtractProductsSIMD(T, sum, result.lower.mat[ii][0..jj], result.lower.mat[jj][0..jj], 1);

            if (ii == jj) {
                if (sum <= 0 or !std.math.isFinite(sum)) {
                    return error.NotPositiveDefinite;
                }
                result.lower.set(ii, ii, @sqrt(sum));
            } else {
                const diag = result.lower.get(jj, jj);
                result.lower.set(ii, jj, sum / diag);
            }
        }
    }

    const root_scale = @sqrt(scale);
    for (0..N) |r| for (0..(r + 1)) |c| {
        result.lower.mat[r][c] *= root_scale;
    };
    return result;
}

/// Solves linear system `A x = b` given a pre-computed Cholesky factorisation `chol`.
///
/// Performs forward solve `L y = b` followed by back solve `L^T x = y`.
/// Returns the solution vector by value. Performs no heap allocation.
pub fn solveCholeskyFactoredStackSIMD(
    comptime N: usize,
    comptime T: type,
    chol: CholeskyStackSIMD(N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    for (0..N) |i| {
        if (chol.lower.get(i, i) <= 0) return error.NotPositiveDefinite;
    }

    // Forward substitution: L y = b
    var y_vec: Vec(N, T) = undefined;
    for (0..N) |ii| {
        var sum = b.get(ii);
        sum = simd.subtractProductsSIMD(T, sum, chol.lower.mat[ii][0..ii], y_vec.vec[0..ii], 1);
        y_vec.set(ii, sum / chol.lower.get(ii, ii));
    }

    // Backward substitution: L^T x = y
    var x_sol: Vec(N, T) = undefined;
    var ii: usize = N;
    while (ii > 0) {
        ii -= 1;
        var sum = y_vec.get(ii);
        if (ii + 1 < N) sum = simd.subtractStridedProductsSIMD(T, sum, @as([*]const T, @ptrCast(&chol.lower.mat))[(ii + 1) * N + ii .. N * N], N, x_sol.vec[ii + 1 ..]);
        x_sol.set(ii, sum / chol.lower.get(ii, ii));
    }

    return x_sol;
}

/// Solves symmetric positive-definite linear system `A x = b` using Cholesky factorisation.
///
/// Convenience wrapper performing factorisation and solve.
/// Dimensions are known at compile time; arguments use value semantics.
/// Performs no heap allocation.
///
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.NotPositiveDefinite` if `a` is not positive definite.
pub fn solveCholeskyStackSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    const chol = try factorCholeskyStackSIMD(N, T, a);
    return solveCholeskyFactoredStackSIMD(N, T, chol, b);
}

/// Computes the Cholesky factorisation `A = L L^T` for borrowed slices.
///
/// `a` is the N x N input matrix.
/// `lower` (N x N) receives the lower triangular factor `L` with upper triangle zeroed.
///
/// Performs no heap allocation.
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.NotPositiveDefinite` if `a` is not positive definite.
pub fn factorCholeskySliceSIMD(
    comptime T: type,
    a: MatView(T),
    lower: MatView(T),
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(lower.rows_num == a.rows_num);
    assert(lower.cols_num == a.cols_num);

    const n_dim = a.rows_num;
    assert(n_dim > 0);
    const scale = matops.maxAbsMatViewSIMD(T, a);
    if (scale == 0) return error.NotPositiveDefinite;
    lower.fill(0);

    for (0..n_dim) |ii| {
        for (0..(ii + 1)) |jj| {
            var sum = a.get(ii, jj) / scale;
            sum = simd.subtractProductsSIMD(T, sum, lower.slice[ii * n_dim .. ii * n_dim + jj], lower.slice[jj * n_dim .. jj * n_dim + jj], 1);

            if (ii == jj) {
                if (sum <= 0 or !std.math.isFinite(sum)) {
                    return error.NotPositiveDefinite;
                }
                lower.set(ii, ii, @sqrt(sum));
            } else {
                const diag = lower.get(jj, jj);
                lower.set(ii, jj, sum / diag);
            }
        }
    }
    const root_scale = @sqrt(scale);
    simd.mulScalSIMD(T, lower.slice, root_scale, lower.slice);
}

/// Solves linear system `A x = b` given pre-factored Cholesky slice `lower`.
///
/// `work_vec` (length N) is caller-supplied scratch storage.
/// `out` (length N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveCholeskyFactoredSliceSIMD(
    comptime T: type,
    lower: MatView(T),
    b: VecView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    for (0..lower.rows_num) |i| {
        if (lower.get(i, i) <= 0) return error.NotPositiveDefinite;
    }
    assert(lower.rows_num == lower.cols_num);
    assert(b.slice.len == lower.rows_num);
    assert(work_vec.slice.len == lower.rows_num);
    assert(out.slice.len == lower.rows_num);

    const n_dim = lower.rows_num;

    // Forward substitution: L y = b
    for (0..n_dim) |ii| {
        var sum = b.get(ii);
        sum = simd.subtractProductsSIMD(T, sum, lower.slice[ii * n_dim .. ii * n_dim + ii], work_vec.slice[0..ii], 1);
        work_vec.set(ii, sum / lower.get(ii, ii));
    }

    // Backward substitution: L^T x = y
    var ii: usize = n_dim;
    while (ii > 0) {
        ii -= 1;
        var sum = work_vec.get(ii);
        if (ii + 1 < n_dim) sum = simd.subtractStridedProductsSIMD(T, sum, lower.slice[(ii + 1) * n_dim + ii .. n_dim * n_dim], n_dim, out.slice[ii + 1 ..]);
        out.set(ii, sum / lower.get(ii, ii));
    }
}

/// Solves symmetric positive-definite linear system `A x = b` for slices.
///
/// `work_mat` (N x N) and `work_vec` (N) provide caller-supplied scratch memory.
/// `out` (N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveCholeskySliceSIMD(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try factorCholeskySliceSIMD(T, a, work_mat);
    try solveCholeskyFactoredSliceSIMD(T, work_mat, b, work_vec, out);
}

pub const factorCholeskyMatSIMD = factorCholeskyStackSIMD;

pub const solveCholeskyFactoredMatSIMD = solveCholeskyFactoredStackSIMD;

pub const solveCholeskyMatSIMD = solveCholeskyStackSIMD;

pub const factorCholeskyMatViewSIMD = factorCholeskySliceSIMD;

pub const solveCholeskyFactoredMatViewSIMD = solveCholeskyFactoredSliceSIMD;

pub const solveCholeskyMatViewSIMD = solveCholeskySliceSIMD;

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn factorCholeskyStackCheckedSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
) SolveError!CholeskyStackSIMD(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatSIMD(N, N, T, a)) return error.NonFiniteInput;
    return factorCholeskyStackSIMD(N, T, a);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskyFactoredStackCheckedSIMD(
    comptime N: usize,
    comptime T: type,
    chol: CholeskyStackSIMD(N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatSIMD(N, N, T, chol.lower)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecSIMD(N, T, b)) return error.NonFiniteInput;
    return solveCholeskyFactoredStackSIMD(N, T, chol, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskyStackCheckedSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatSIMD(N, N, T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecSIMD(N, T, b)) return error.NonFiniteInput;
    return solveCholeskyStackSIMD(N, T, a, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn factorCholeskySliceCheckedSIMD(
    comptime T: type,
    a: MatView(T),
    lower: MatView(T),
) SolveError!void {
    try contracts.matrix(T, a);
    try contracts.matrix(T, lower);
    if (contracts.overlaps(a.slice, lower.slice)) return error.InvalidAliasing;
    if (!(a.rows_num == a.cols_num)) return error.ShapeMismatch;
    if (!(lower.rows_num == a.rows_num)) return error.ShapeMismatch;
    if (!(lower.cols_num == a.cols_num)) return error.ShapeMismatch;

    if (a.rows_num == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatViewSIMD(T, a)) return error.NonFiniteInput;
    return factorCholeskySliceSIMD(T, a, lower);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskyFactoredSliceCheckedSIMD(
    comptime T: type,
    lower: MatView(T),
    b: VecView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try contracts.factored(T, lower, b, work_vec, out);
    if (!(lower.rows_num == lower.cols_num)) return error.ShapeMismatch;
    if (!(b.slice.len == lower.rows_num)) return error.ShapeMismatch;
    if (!(work_vec.slice.len == lower.rows_num)) return error.ShapeMismatch;
    if (!(out.slice.len == lower.rows_num)) return error.ShapeMismatch;
    if (!matops.allFiniteMatViewSIMD(T, lower)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecViewSIMD(T, b)) return error.NonFiniteInput;
    return solveCholeskyFactoredSliceSIMD(T, lower, b, work_vec, out);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskySliceCheckedSIMD(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try contracts.solve(T, a, b, work_mat, work_vec, out);
    if (a.rows_num == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatViewSIMD(T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecViewSIMD(T, b)) return error.NonFiniteInput;
    return solveCholeskySliceSIMD(T, a, b, work_mat, work_vec, out);
}

pub const solveCholeskyFactoredMatCheckedSIMD = solveCholeskyFactoredStackCheckedSIMD;

pub const solveCholeskyMatCheckedSIMD = solveCholeskyStackCheckedSIMD;

pub const factorCholeskyMatViewCheckedSIMD = factorCholeskySliceCheckedSIMD;

pub const solveCholeskyFactoredMatViewCheckedSIMD = solveCholeskyFactoredSliceCheckedSIMD;

pub const solveCholeskyMatViewCheckedSIMD = solveCholeskySliceCheckedSIMD;

pub const factorCholeskyMatCheckedSIMD = factorCholeskyStackCheckedSIMD;
