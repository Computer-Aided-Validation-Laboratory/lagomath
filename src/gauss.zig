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
const rowops = @import("rowops.zig");
const matops = @import("matops.zig");
const vecops = @import("vecops.zig");
const contracts = @import("contracts.zig");
const SolveError = @import("linsolve.zig").SolveError;

/// Solves linear system `A x = b` using Gaussian elimination with partial pivoting.
///
/// `a` is an N x N fixed-size square matrix and `b` has length N.
/// Arguments are passed by value and are not mutated.
/// The implementation uses local fixed-size scratch storage and performs no heap allocation.
///
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.SingularMatrix` if the system is singular or rank-deficient.
pub fn solveStack(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    comptime assert(N > 0);
    const scale = matops.maxAbsStack(N, N, T, a);
    if (scale == 0) return error.SingularMatrix;
    var mat_work = a;
    var vec_work = b;
    for (0..N) |r| {
        vec_work.vec[r] /= scale;
        for (0..N) |c| mat_work.mat[r][c] /= scale;
    }
    const eps_tol: T = std.math.floatEps(T) * @as(T, @floatFromInt(N));

    for (0..N) |kk| {
        var pivot_row = kk;
        var max_abs_pivot = @abs(mat_work.get(kk, kk));
        for ((kk + 1)..N) |rr| {
            const abs_val = @abs(mat_work.get(rr, kk));
            if (abs_val > max_abs_pivot) {
                max_abs_pivot = abs_val;
                pivot_row = rr;
            }
        }

        if (max_abs_pivot <= eps_tol) {
            return error.SingularMatrix;
        }

        if (pivot_row != kk) {
            rowops.swapRowsStack(N, N, T, kk, pivot_row, &mat_work);
            const temp_vec = vec_work.get(kk);
            vec_work.set(kk, vec_work.get(pivot_row));
            vec_work.set(pivot_row, temp_vec);
        }

        const pivot_elem = mat_work.get(kk, kk);
        for ((kk + 1)..N) |rr| {
            const factor = mat_work.get(rr, kk) / pivot_elem;
            mat_work.set(rr, kk, 0);
            for ((kk + 1)..N) |cc| {
                const current_val = mat_work.get(rr, cc);
                mat_work.set(rr, cc, current_val - factor * mat_work.get(kk, cc));
            }
            const current_rhs = vec_work.get(rr);
            vec_work.set(rr, current_rhs - factor * vec_work.get(kk));
        }
    }

    var x_sol: Vec(N, T) = undefined;
    var ii: usize = N;
    while (ii > 0) {
        ii -= 1;
        const diag_elem = mat_work.get(ii, ii);
        if (@abs(diag_elem) <= eps_tol) {
            return error.SingularMatrix;
        }
        var rhs_sum = vec_work.get(ii);
        for ((ii + 1)..N) |jj| {
            rhs_sum -= mat_work.get(ii, jj) * x_sol.get(jj);
        }
        x_sol.set(ii, rhs_sum / diag_elem);
    }

    return x_sol;
}

/// Solves linear system `A x = b` using Gaussian elimination with
/// partial pivoting for slices.
///
/// `a` is an N x N matrix and `b` has length N.
/// `work_mat` (N x N) and `work_vec` (N) provide scratch memory owned by the caller.
/// `out` (N) receives the solution vector.
///
/// Performs no heap allocation.
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.SingularMatrix` if the system is singular or rank-deficient.
pub fn solveSlice(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(a.rows_num == b.slice.len);
    assert(work_mat.rows_num == a.rows_num);
    assert(work_mat.cols_num == a.cols_num);
    assert(work_vec.slice.len == a.rows_num);
    assert(out.slice.len == a.rows_num);

    const n_dim = a.rows_num;
    @memcpy(work_mat.slice, a.slice);
    @memcpy(work_vec.slice, b.slice);
    assert(n_dim > 0);
    const scale = matops.maxAbsSlice(T, a);
    if (scale == 0) return error.SingularMatrix;
    for (work_mat.slice) |*v| v.* /= scale;
    for (work_vec.slice) |*v| v.* /= scale;
    const eps_tol: T = std.math.floatEps(T) * @as(T, @floatFromInt(n_dim));

    for (0..n_dim) |kk| {
        var pivot_row = kk;
        var max_abs_pivot = @abs(work_mat.get(kk, kk));
        for ((kk + 1)..n_dim) |rr| {
            const abs_val = @abs(work_mat.get(rr, kk));
            if (abs_val > max_abs_pivot) {
                max_abs_pivot = abs_val;
                pivot_row = rr;
            }
        }

        if (max_abs_pivot <= eps_tol) {
            return error.SingularMatrix;
        }

        if (pivot_row != kk) {
            rowops.swapRowsSlice(T, kk, pivot_row, work_mat);
            const temp_vec = work_vec.get(kk);
            work_vec.set(kk, work_vec.get(pivot_row));
            work_vec.set(pivot_row, temp_vec);
        }

        const pivot_elem = work_mat.get(kk, kk);
        for ((kk + 1)..n_dim) |rr| {
            const factor = work_mat.get(rr, kk) / pivot_elem;
            work_mat.set(rr, kk, 0);
            for ((kk + 1)..n_dim) |cc| {
                const current_val = work_mat.get(rr, cc);
                work_mat.set(rr, cc, current_val - factor * work_mat.get(kk, cc));
            }
            const current_rhs = work_vec.get(rr);
            work_vec.set(rr, current_rhs - factor * work_vec.get(kk));
        }
    }

    var ii: usize = n_dim;
    while (ii > 0) {
        ii -= 1;
        const diag_elem = work_mat.get(ii, ii);
        if (@abs(diag_elem) <= eps_tol) {
            return error.SingularMatrix;
        }
        var rhs_sum = work_vec.get(ii);
        for ((ii + 1)..n_dim) |jj| {
            rhs_sum -= work_mat.get(ii, jj) * out.get(jj);
        }
        out.set(ii, rhs_sum / diag_elem);
    }
}

test "gauss.solveStack 2x2 known system" {
    try @import("case_runner.zig").verifyLegacy("legacy.gauss.gauss_solveStack_2x2_known_system");
}

test "gauss.solveStack 3x3 requiring pivoting" {
    try @import("case_runner.zig").verifyLegacy("legacy.gauss.gauss_solveStack_3x3_requiring_pivoting");
}

test "gauss.solveStack singular matrix" {
    try @import("case_runner.zig").verifyLegacy("legacy.gauss.gauss_solveStack_singular_matrix");
}

test "gauss.solveStack non-finite input" {
    try @import("case_runner.zig").verifyLegacy("legacy.gauss.gauss_solveStack_non_finite_input");
}

test "gauss.solveSlice 3x3 with scratch" {
    try @import("case_runner.zig").verifyLegacy("legacy.gauss.gauss_solveSlice_3x3_with_scratch");
}

pub const solveMat = solveStack;

pub const solveMatView = solveSlice;

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveStackChecked(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteStack(N, N, T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteStack(N, T, b)) return error.NonFiniteInput;
    return solveStack(N, T, a, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveSliceChecked(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try contracts.solve(T, a, b, work_mat, work_vec, out);
    if (!(a.rows_num == a.cols_num)) return error.ShapeMismatch;
    if (!(a.rows_num == b.slice.len)) return error.ShapeMismatch;
    if (!(work_mat.rows_num == a.rows_num)) return error.ShapeMismatch;
    if (!(work_mat.cols_num == a.cols_num)) return error.ShapeMismatch;
    if (!(work_vec.slice.len == a.rows_num)) return error.ShapeMismatch;
    if (!(out.slice.len == a.rows_num)) return error.ShapeMismatch;

    if (a.rows_num == 0) return error.ShapeMismatch;
    if (!matops.allFiniteSlice(T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteSlice(T, b)) return error.NonFiniteInput;
    return solveSlice(T, a, b, work_mat, work_vec, out);
}

pub const solveMatChecked = solveStackChecked;

pub const solveMatViewChecked = solveSliceChecked;
