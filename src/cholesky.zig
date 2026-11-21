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
const matops = @import("matops.zig");
const vecops = @import("vecops.zig");
const contracts = @import("contracts.zig");
const SolveError = @import("linsolve.zig").SolveError;

/// Stack-allocated factorisation representation storing the lower triangular Cholesky factor.
pub fn CholeskyStack(comptime N: usize, comptime T: type) type {
    return struct {
        lower: Mat(N, N, T),
    };
}

/// Computes the Cholesky factorisation `A = L L^T` for a symmetric positive-definite matrix.
///
/// `a` is an N x N fixed-size matrix passed by value.
/// Returns `CholeskyStack(N, T)` containing the lower triangular factor `L`.
/// Performs no heap allocation.
///
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.NotPositiveDefinite` if `A` is not positive definite.
pub fn factorCholeskyStack(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
) SolveError!CholeskyStack(N, T) {
    comptime assert(N > 0);
    const scale = matops.maxAbsStack(N, N, T, a);
    if (scale == 0) return error.NotPositiveDefinite;
    var result = CholeskyStack(N, T){
        .lower = Mat(N, N, T).initZeros(),
    };

    for (0..N) |ii| {
        for (0..(ii + 1)) |jj| {
            var sum = a.get(ii, jj) / scale;
            for (0..jj) |kk| {
                sum -= result.lower.get(ii, kk) * result.lower.get(jj, kk);
            }

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
pub fn solveCholeskyFactoredStack(
    comptime N: usize,
    comptime T: type,
    chol: CholeskyStack(N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    for (0..N) |i| {
        if (chol.lower.get(i, i) <= 0) return error.NotPositiveDefinite;
    }

    // Forward substitution: L y = b
    var y_vec: Vec(N, T) = undefined;
    for (0..N) |ii| {
        var sum = b.get(ii);
        for (0..ii) |jj| {
            sum -= chol.lower.get(ii, jj) * y_vec.get(jj);
        }
        y_vec.set(ii, sum / chol.lower.get(ii, ii));
    }

    // Backward substitution: L^T x = y
    var x_sol: Vec(N, T) = undefined;
    var ii: usize = N;
    while (ii > 0) {
        ii -= 1;
        var sum = y_vec.get(ii);
        for ((ii + 1)..N) |jj| {
            sum -= chol.lower.get(jj, ii) * x_sol.get(jj);
        }
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
pub fn solveCholeskyStack(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    const chol = try factorCholeskyStack(N, T, a);
    return solveCholeskyFactoredStack(N, T, chol, b);
}

/// Computes the Cholesky factorisation `A = L L^T` for borrowed slices.
///
/// `a` is the N x N input matrix.
/// `lower` (N x N) receives the lower triangular factor `L` with upper triangle zeroed.
///
/// Performs no heap allocation.
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.NotPositiveDefinite` if `a` is not positive definite.
pub fn factorCholeskySlice(
    comptime T: type,
    a: MatView(T),
    lower: MatView(T),
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(lower.rows_num == a.rows_num);
    assert(lower.cols_num == a.cols_num);

    const n_dim = a.rows_num;
    assert(n_dim > 0);
    const scale = matops.maxAbsSlice(T, a);
    if (scale == 0) return error.NotPositiveDefinite;
    lower.fill(0);

    for (0..n_dim) |ii| {
        for (0..(ii + 1)) |jj| {
            var sum = a.get(ii, jj) / scale;
            for (0..jj) |kk| {
                sum -= lower.get(ii, kk) * lower.get(jj, kk);
            }

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
    for (lower.slice) |*v| v.* *= root_scale;
}

/// Solves linear system `A x = b` given pre-factored Cholesky slice `lower`.
///
/// `work_vec` (length N) is caller-supplied scratch storage.
/// `out` (length N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveCholeskyFactoredSlice(
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
        for (0..ii) |jj| {
            sum -= lower.get(ii, jj) * work_vec.get(jj);
        }
        work_vec.set(ii, sum / lower.get(ii, ii));
    }

    // Backward substitution: L^T x = y
    var ii: usize = n_dim;
    while (ii > 0) {
        ii -= 1;
        var sum = work_vec.get(ii);
        for ((ii + 1)..n_dim) |jj| {
            sum -= lower.get(jj, ii) * out.get(jj);
        }
        out.set(ii, sum / lower.get(ii, ii));
    }
}

/// Solves symmetric positive-definite linear system `A x = b` for slices.
///
/// `work_mat` (N x N) and `work_vec` (N) provide caller-supplied scratch memory.
/// `out` (N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveCholeskySlice(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try factorCholeskySlice(T, a, work_mat);
    try solveCholeskyFactoredSlice(T, work_mat, b, work_vec, out);
}

test "cholesky.factorCholeskyStack and solveCholeskyStack 2x2" {
    try @import("case_runner.zig").verifyLegacy("legacy.cholesky.cholesky_factorCholeskyStack_and_solveCholeskyStack_2x2");
}

test "cholesky.factorCholeskyStack rejects non-positive-definite" {
    try @import("case_runner.zig").verifyLegacy("legacy.cholesky.cholesky_factorCholeskyStack_rejects_non_positive_definite");
}

test "cholesky.solveCholeskySlice 2x2" {
    try @import("case_runner.zig").verifyLegacy("legacy.cholesky.cholesky_solveCholeskySlice_2x2");
}

test "cholesky factorisation reconstruction A == L * L^T" {
    try @import("case_runner.zig").verifyLegacy("legacy.cholesky.cholesky_factorisation_reconstruction_A_L_L_T");
}

pub const factorCholeskyMat = factorCholeskyStack;

pub const solveCholeskyFactoredMat = solveCholeskyFactoredStack;

pub const solveCholeskyMat = solveCholeskyStack;

pub const factorCholeskyMatView = factorCholeskySlice;

pub const solveCholeskyFactoredMatView = solveCholeskyFactoredSlice;

pub const solveCholeskyMatView = solveCholeskySlice;

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn factorCholeskyStackChecked(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
) SolveError!CholeskyStack(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteStack(N, N, T, a)) return error.NonFiniteInput;
    return factorCholeskyStack(N, T, a);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskyFactoredStackChecked(
    comptime N: usize,
    comptime T: type,
    chol: CholeskyStack(N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteStack(N, N, T, chol.lower)) return error.NonFiniteInput;
    if (!vecops.allFiniteStack(N, T, b)) return error.NonFiniteInput;
    return solveCholeskyFactoredStack(N, T, chol, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskyStackChecked(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteStack(N, N, T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteStack(N, T, b)) return error.NonFiniteInput;
    return solveCholeskyStack(N, T, a, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn factorCholeskySliceChecked(
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
    if (!matops.allFiniteSlice(T, a)) return error.NonFiniteInput;
    return factorCholeskySlice(T, a, lower);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskyFactoredSliceChecked(
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
    if (!matops.allFiniteSlice(T, lower)) return error.NonFiniteInput;
    if (!vecops.allFiniteSlice(T, b)) return error.NonFiniteInput;
    return solveCholeskyFactoredSlice(T, lower, b, work_vec, out);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveCholeskySliceChecked(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try contracts.solve(T, a, b, work_mat, work_vec, out);
    if (a.rows_num == 0) return error.ShapeMismatch;
    if (!matops.allFiniteSlice(T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteSlice(T, b)) return error.NonFiniteInput;
    return solveCholeskySlice(T, a, b, work_mat, work_vec, out);
}

pub const solveCholeskyFactoredMatChecked = solveCholeskyFactoredStackChecked;

pub const solveCholeskyMatChecked = solveCholeskyStackChecked;

pub const factorCholeskyMatViewChecked = factorCholeskySliceChecked;

pub const solveCholeskyFactoredMatViewChecked = solveCholeskyFactoredSliceChecked;

pub const solveCholeskyMatViewChecked = solveCholeskySliceChecked;

pub const factorCholeskyMatChecked = factorCholeskyStackChecked;
