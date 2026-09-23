// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const assert = std.debug.assert;
const MatStack = @import("matstack.zig").MatStack;
const VecStack = @import("vecstack.zig").VecStack;
const MatSlice = @import("matslice.zig").MatSlice;
const VecSlice = @import("vecslice.zig").VecSlice;
const rowops = @import("rowops.zig");
const matops = @import("matops.zig");
const vecops = @import("vecops.zig");
const SolveError = @import("linsolve.zig").SolveError;

/// Solves linear system `A x = b` using Gaussian elimination with partial pivoting.
///
/// `a` is an N x N fixed-size square matrix and `b` has length N.
/// Arguments are passed by value and are not mutated.
/// The implementation uses local fixed-size scratch storage and performs no heap allocation.
///
/// Returns `error.NonFiniteInput` if any input element is NaN or Inf.
/// Returns `error.SingularMatrix` if the system is singular or rank-deficient.
pub fn solveStack(
    comptime N: usize,
    comptime T: type,
    a: MatStack(N, N, T),
    b: VecStack(N, T),
) SolveError!VecStack(N, T) {
    if (!matops.allFiniteStack(N, N, T, a) or !vecops.allFiniteStack(N, T, b)) {
        return error.NonFiniteInput;
    }

    var mat_work = a;
    var vec_work = b;
    const eps_tol: T = std.math.floatEps(T) * 1e-4;

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

    var x_sol: VecStack(N, T) = undefined;
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
/// Returns `error.NonFiniteInput` if any input element is NaN or Inf.
/// Returns `error.SingularMatrix` if the system is singular or rank-deficient.
pub fn solveSlice(
    comptime T: type,
    a: MatSlice(T),
    b: VecSlice(T),
    work_mat: MatSlice(T),
    work_vec: VecSlice(T),
    out: VecSlice(T),
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(a.rows_num == b.slice.len);
    assert(work_mat.rows_num == a.rows_num);
    assert(work_mat.cols_num == a.cols_num);
    assert(work_vec.slice.len == a.rows_num);
    assert(out.slice.len == a.rows_num);

    if (!matops.allFiniteSlice(T, a) or !vecops.allFiniteSlice(T, b)) {
        return error.NonFiniteInput;
    }

    const n_dim = a.rows_num;
    @memcpy(work_mat.slice, a.slice);
    @memcpy(work_vec.slice, b.slice);
    const eps_tol: T = std.math.floatEps(T) * 1e-4;

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
    const a2 = MatStack(2, 2, f64).initRows(.{
        .{ 3.0, 1.0 },
        .{ 1.0, 2.0 },
    });
    const b2 = VecStack(2, f64).initSlice(&.{ 5.0, 5.0 });
    const x2 = try solveStack(2, f64, a2, b2);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x2.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x2.get(1), 1e-12);
}

test "gauss.solveStack 3x3 requiring pivoting" {
    // Top-left is zero, strictly requires row swap
    const a3 = MatStack(3, 3, f64).initRows(.{
        .{ 0.0, 2.0, 1.0 },
        .{ 3.0, -1.0, 2.0 },
        .{ 4.0, -1.0, 5.0 },
    });
    const b3 = VecStack(3, f64).initSlice(&.{ 5.0, 7.0, 15.0 });
    const x3 = try solveStack(3, f64, a3, b3);

    // Exact reference from Python/NumPy:
    // x = [17/13, 18/13, 29/13]
    // ≈ [1.3076923076923077, 1.3846153846153846, 2.2307692307692308]
    try std.testing.expectApproxEqAbs(@as(f64, 1.3076923076923077), x3.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3846153846153846), x3.get(1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2307692307692308), x3.get(2), 1e-12);
}

test "gauss.solveStack singular matrix" {
    const a_sing = MatStack(2, 2, f64).initRows(.{
        .{ 1.0, 2.0 },
        .{ 2.0, 4.0 },
    });
    const b = VecStack(2, f64).initSlice(&.{ 1.0, 2.0 });
    try std.testing.expectError(error.SingularMatrix, solveStack(2, f64, a_sing, b));
}

test "gauss.solveStack non-finite input" {
    var a_nan = MatStack(2, 2, f64).initFill(1.0);
    a_nan.set(0, 0, std.math.nan(f64));
    const b = VecStack(2, f64).initSlice(&.{ 1.0, 2.0 });
    try std.testing.expectError(error.NonFiniteInput, solveStack(2, f64, a_nan, b));
}

test "gauss.solveSlice 3x3 with scratch" {
    var a_raw = [_]f64{
        0.0, 2.0,  1.0,
        3.0, -1.0, 2.0,
        4.0, -1.0, 5.0,
    };
    var b_raw = [_]f64{ 5.0, 7.0, 15.0 };
    var work_mat_raw = [_]f64{0.0} ** 9;
    var work_vec_raw = [_]f64{0.0} ** 3;
    var out_raw = [_]f64{0.0} ** 3;

    try solveSlice(
        f64,
        MatSlice(f64).init(&a_raw, 3, 3),
        VecSlice(f64).init(&b_raw),
        MatSlice(f64).init(&work_mat_raw, 3, 3),
        VecSlice(f64).init(&work_vec_raw),
        VecSlice(f64).init(&out_raw),
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.3076923076923077), out_raw[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3846153846153846), out_raw[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2307692307692308), out_raw[2], 1e-12);
}
