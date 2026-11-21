const simd = @import("sliceops_simd.zig");
pub const LUStackSIMD = @import("lu.zig").LUStack;
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
const rowops = @import("rowops_simd.zig");
const matops = @import("matops_simd.zig");
const vecops = @import("vecops_simd.zig");
const contracts = @import("contracts.zig");
const SolveError = @import("linsolve.zig").SolveError;

/// Stack-allocated factorisation representation storing combined LU matrix and
/// row permutation index array.
/// Computes the in-place LU factorisation with partial pivoting `P A = L U`.
///
/// `a` is an N x N fixed-size matrix passed by value.
/// Returns `LUStackSIMD(N, T)` containing the packed LU matrix and permutation indices.
/// Performs no heap allocation.
///
/// Requires finite inputs; the Checked variant validates this precondition.
/// Returns `error.SingularMatrix` if any pivot is zero or singular.
pub fn factorLUStackSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
) SolveError!LUStackSIMD(N, T) {
    var result = LUStackSIMD(N, T){
        .lu = a,
        .permutation = undefined,
    };
    inline for (0..N) |ii| {
        result.permutation[ii] = ii;
    }

    comptime assert(N > 0);
    const scale = matops.maxAbsMatSIMD(N, N, T, a);
    if (scale == 0) return error.SingularMatrix;
    for (0..N) |r| simd.divScalSIMD(T, &result.lu.mat[r], scale, &result.lu.mat[r]);
    const eps_tol: T = std.math.floatEps(T) * @as(T, @floatFromInt(N));

    for (0..N) |kk| {
        var pivot_row = kk;
        var max_abs_pivot = @abs(result.lu.get(kk, kk));
        for ((kk + 1)..N) |rr| {
            const abs_val = @abs(result.lu.get(rr, kk));
            if (abs_val > max_abs_pivot) {
                max_abs_pivot = abs_val;
                pivot_row = rr;
            }
        }

        if (max_abs_pivot <= eps_tol) {
            return error.SingularMatrix;
        }

        if (pivot_row != kk) {
            rowops.swapRowsMatSIMD(N, N, T, kk, pivot_row, &result.lu);
            const temp_perm = result.permutation[kk];
            result.permutation[kk] = result.permutation[pivot_row];
            result.permutation[pivot_row] = temp_perm;
        }

        const pivot_elem = result.lu.get(kk, kk);
        for ((kk + 1)..N) |rr| {
            const mult = result.lu.get(rr, kk) / pivot_elem;
            result.lu.set(rr, kk, mult);
            simd.subtractScaledSIMD(T, result.lu.mat[kk][kk + 1 ..], mult, result.lu.mat[rr][kk + 1 ..]);
        }
    }

    for (0..N) |ii| {
        if (@abs(result.lu.get(ii, ii)) <= eps_tol) {
            return error.SingularMatrix;
        }
    }

    for (0..N) |r| for (r..N) |c| {
        result.lu.mat[r][c] *= scale;
    };
    return result;
}

/// Solves linear system `A x = b` given a pre-computed factorisation `lu`.
///
/// `lu` contains the packed LU matrix and permutation array.
/// `b` is the right-hand side vector.
/// Performs forward substitution `L y = P b` followed by back substitution `U x = y`.
/// Returns the solution vector by value. Performs no heap allocation.
pub fn solveLUFactoredStackSIMD(
    comptime N: usize,
    comptime T: type,
    lu: LUStackSIMD(N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    for (0..N) |i| {
        if (lu.lu.get(i, i) == 0) return error.SingularMatrix;
    }
    var scale: T = 0;
    for (0..N) |r| for (r..N) |c| {
        scale = @max(scale, @abs(lu.lu.get(r, c)));
    };
    if (scale == 0) return error.SingularMatrix;
    // Apply row permutation: y = P * b
    var y_vec: Vec(N, T) = undefined;
    inline for (0..N) |ii| {
        y_vec.set(ii, b.get(lu.permutation[ii]) / scale);
    }

    // Forward substitution: L y = P b (unit diagonal on L)
    for (0..N) |ii| {
        var sum = y_vec.get(ii);
        sum = simd.subtractProductsSIMD(T, sum, lu.lu.mat[ii][0..ii], y_vec.vec[0..ii], 1);
        y_vec.set(ii, sum);
    }

    // Backward substitution: U x = y
    var x_sol: Vec(N, T) = undefined;
    var ii: usize = N;
    while (ii > 0) {
        ii -= 1;
        var sum = y_vec.get(ii);
        sum = simd.subtractProductsSIMD(T, sum, lu.lu.mat[ii][ii + 1 ..], x_sol.vec[ii + 1 ..], scale);
        x_sol.set(ii, sum / (lu.lu.get(ii, ii) / scale));
    }

    return x_sol;
}

/// Convenience function that computes LU factorisation and solves `A x = b`.
///
/// Fixed-size value semantics; performs no heap allocation.
pub fn solveLUStackSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    const factored = try factorLUStackSIMD(N, T, a);
    return solveLUFactoredStackSIMD(N, T, factored, b);
}

/// Computes the in-place LU factorisation with partial pivoting for slices.
///
/// `a` is the N x N input matrix.
/// `lu_mat` (N x N) receives the packed L and U factors.
/// `perm` (length N) receives the pivot row permutation array.
///
/// Performs no heap allocation.
pub fn factorLUSliceSIMD(
    comptime T: type,
    a: MatView(T),
    lu_mat: MatView(T),
    perm: []usize,
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(lu_mat.rows_num == a.rows_num);
    assert(lu_mat.cols_num == a.cols_num);
    assert(perm.len == a.rows_num);

    const n_dim = a.rows_num;
    @memcpy(lu_mat.slice, a.slice);
    for (0..n_dim) |ii| {
        perm[ii] = ii;
    }

    assert(n_dim > 0);
    const scale = matops.maxAbsMatViewSIMD(T, a);
    if (scale == 0) return error.SingularMatrix;
    simd.divScalSIMD(T, lu_mat.slice, scale, lu_mat.slice);
    const eps_tol: T = std.math.floatEps(T) * @as(T, @floatFromInt(n_dim));

    for (0..n_dim) |kk| {
        var pivot_row = kk;
        var max_abs_pivot = @abs(lu_mat.get(kk, kk));
        for ((kk + 1)..n_dim) |rr| {
            const abs_val = @abs(lu_mat.get(rr, kk));
            if (abs_val > max_abs_pivot) {
                max_abs_pivot = abs_val;
                pivot_row = rr;
            }
        }

        if (max_abs_pivot <= eps_tol) {
            return error.SingularMatrix;
        }

        if (pivot_row != kk) {
            rowops.swapRowsMatViewSIMD(T, kk, pivot_row, lu_mat);
            const temp_perm = perm[kk];
            perm[kk] = perm[pivot_row];
            perm[pivot_row] = temp_perm;
        }

        const pivot_elem = lu_mat.get(kk, kk);
        for ((kk + 1)..n_dim) |rr| {
            const mult = lu_mat.get(rr, kk) / pivot_elem;
            lu_mat.set(rr, kk, mult);
            simd.subtractScaledSIMD(T, lu_mat.slice[kk * n_dim + kk + 1 .. (kk + 1) * n_dim], mult, lu_mat.slice[rr * n_dim + kk + 1 .. (rr + 1) * n_dim]);
        }
    }

    for (0..n_dim) |ii| {
        if (@abs(lu_mat.get(ii, ii)) <= eps_tol) {
            return error.SingularMatrix;
        }
    }
    for (0..n_dim) |r| for (r..n_dim) |c| {
        lu_mat.set(r, c, lu_mat.get(r, c) * scale);
    };
}

/// Solves linear system `A x = b` given pre-factored slice `lu_mat` and `perm`.
///
/// `work_vec` (length N) is caller-supplied scratch storage.
/// `out` (length N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveLUFactoredSliceSIMD(
    comptime T: type,
    lu_mat: MatView(T),
    perm: []const usize,
    b: VecView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    for (0..lu_mat.rows_num) |i| {
        if (lu_mat.get(i, i) == 0) return error.SingularMatrix;
    }
    assert(lu_mat.rows_num == lu_mat.cols_num);
    assert(perm.len == lu_mat.rows_num);
    assert(b.slice.len == lu_mat.rows_num);
    assert(work_vec.slice.len == lu_mat.rows_num);
    assert(out.slice.len == lu_mat.rows_num);

    const n_dim = lu_mat.rows_num;

    var scale: T = 0;
    for (0..n_dim) |r| for (r..n_dim) |c| {
        scale = @max(scale, @abs(lu_mat.get(r, c)));
    };
    if (scale == 0) return error.SingularMatrix;
    // Apply permutation: work_vec = P * b
    for (0..n_dim) |ii| {
        work_vec.set(ii, b.get(perm[ii]) / scale);
    }

    // Forward substitution: L y = P b
    for (0..n_dim) |ii| {
        var sum = work_vec.get(ii);
        sum = simd.subtractProductsSIMD(T, sum, lu_mat.slice[ii * n_dim .. ii * n_dim + ii], work_vec.slice[0..ii], 1);
        work_vec.set(ii, sum);
    }

    // Backward substitution: U x = y
    var ii: usize = n_dim;
    while (ii > 0) {
        ii -= 1;
        var sum = work_vec.get(ii);
        sum = simd.subtractProductsSIMD(T, sum, lu_mat.slice[ii * n_dim + ii + 1 .. (ii + 1) * n_dim], out.slice[ii + 1 ..], scale);
        out.set(ii, sum / (lu_mat.get(ii, ii) / scale));
    }
}

/// Solves linear system `A x = b` via LU factorisation with caller-supplied scratch storage.
///
/// Performs no heap allocation.
pub fn solveLUSliceSIMD(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    perm: []usize,
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try factorLUSliceSIMD(T, a, work_mat, perm);
    try solveLUFactoredSliceSIMD(T, work_mat, perm, b, work_vec, out);
}

pub const factorLUMatSIMD = factorLUStackSIMD;

pub const solveLUFactoredMatSIMD = solveLUFactoredStackSIMD;

pub const solveLUMatSIMD = solveLUStackSIMD;

pub const factorLUMatViewSIMD = factorLUSliceSIMD;

pub const solveLUFactoredMatViewSIMD = solveLUFactoredSliceSIMD;

pub const solveLUMatViewSIMD = solveLUSliceSIMD;

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn factorLUStackCheckedSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
) SolveError!LUStackSIMD(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatSIMD(N, N, T, a)) return error.NonFiniteInput;
    return factorLUStackSIMD(N, T, a);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveLUFactoredStackCheckedSIMD(
    comptime N: usize,
    comptime T: type,
    lu: LUStackSIMD(N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatSIMD(N, N, T, lu.lu)) return error.NonFiniteInput;
    for (lu.permutation, 0..) |idx, i| {
        if (idx >= N) return error.InvalidPermutation;
        for (lu.permutation[0..i]) |prev| {
            if (prev == idx) return error.InvalidPermutation;
        }
    }
    if (!vecops.allFiniteVecSIMD(N, T, b)) return error.NonFiniteInput;
    return solveLUFactoredStackSIMD(N, T, lu, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveLUStackCheckedSIMD(
    comptime N: usize,
    comptime T: type,
    a: Mat(N, N, T),
    b: Vec(N, T),
) SolveError!Vec(N, T) {
    if (N == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatSIMD(N, N, T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecSIMD(N, T, b)) return error.NonFiniteInput;
    return solveLUStackSIMD(N, T, a, b);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn factorLUSliceCheckedSIMD(
    comptime T: type,
    a: MatView(T),
    lu_mat: MatView(T),
    perm: []usize,
) SolveError!void {
    if (contracts.overlaps(perm, a.slice) or contracts.overlaps(perm, lu_mat.slice)) return error.InvalidAliasing;
    try contracts.matrix(T, a);
    try contracts.matrix(T, lu_mat);
    if (contracts.overlaps(a.slice, lu_mat.slice)) return error.InvalidAliasing;
    if (!(a.rows_num == a.cols_num)) return error.ShapeMismatch;
    if (!(lu_mat.rows_num == a.rows_num)) return error.ShapeMismatch;
    if (!(lu_mat.cols_num == a.cols_num)) return error.ShapeMismatch;
    if (!(perm.len == a.rows_num)) return error.ShapeMismatch;

    if (a.rows_num == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatViewSIMD(T, a)) return error.NonFiniteInput;

    return factorLUSliceSIMD(T, a, lu_mat, perm);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveLUFactoredSliceCheckedSIMD(
    comptime T: type,
    lu_mat: MatView(T),
    perm: []const usize,
    b: VecView(T),
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    if (contracts.overlaps(perm, lu_mat.slice) or contracts.overlaps(perm, b.slice) or contracts.overlaps(perm, work_vec.slice) or contracts.overlaps(perm, out.slice)) return error.InvalidAliasing;
    try contracts.factored(T, lu_mat, b, work_vec, out);
    if (!(lu_mat.rows_num == lu_mat.cols_num)) return error.ShapeMismatch;
    if (!(perm.len == lu_mat.rows_num)) return error.ShapeMismatch;
    if (!(b.slice.len == lu_mat.rows_num)) return error.ShapeMismatch;
    if (!(work_vec.slice.len == lu_mat.rows_num)) return error.ShapeMismatch;
    if (!(out.slice.len == lu_mat.rows_num)) return error.ShapeMismatch;
    if (!matops.allFiniteMatViewSIMD(T, lu_mat)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecViewSIMD(T, b)) return error.NonFiniteInput;
    for (perm, 0..) |idx, i| {
        if (idx >= perm.len) return error.InvalidPermutation;
        for (perm[0..i]) |prev| {
            if (prev == idx) return error.InvalidPermutation;
        }
    }
    return solveLUFactoredSliceSIMD(T, lu_mat, perm, b, work_vec, out);
}

/// Validates shapes and finite inputs before running the numerical kernel.
pub fn solveLUSliceCheckedSIMD(
    comptime T: type,
    a: MatView(T),
    b: VecView(T),
    work_mat: MatView(T),
    perm: []usize,
    work_vec: VecView(T),
    out: VecView(T),
) SolveError!void {
    try contracts.solve(T, a, b, work_mat, work_vec, out);
    if (perm.len != a.rows_num) return error.ShapeMismatch;
    if (contracts.overlaps(perm, a.slice) or contracts.overlaps(perm, b.slice) or contracts.overlaps(perm, work_mat.slice) or contracts.overlaps(perm, work_vec.slice) or contracts.overlaps(perm, out.slice)) return error.InvalidAliasing;
    if (a.rows_num == 0) return error.ShapeMismatch;
    if (!matops.allFiniteMatViewSIMD(T, a)) return error.NonFiniteInput;
    if (!vecops.allFiniteVecViewSIMD(T, b)) return error.NonFiniteInput;

    return solveLUSliceSIMD(T, a, b, work_mat, perm, work_vec, out);
}

pub const solveLUFactoredMatCheckedSIMD = solveLUFactoredStackCheckedSIMD;

pub const solveLUMatCheckedSIMD = solveLUStackCheckedSIMD;

pub const factorLUMatViewCheckedSIMD = factorLUSliceCheckedSIMD;

pub const solveLUFactoredMatViewCheckedSIMD = solveLUFactoredSliceCheckedSIMD;

pub const solveLUMatViewCheckedSIMD = solveLUSliceCheckedSIMD;

pub const factorLUMatCheckedSIMD = factorLUStackCheckedSIMD;
