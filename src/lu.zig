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

/// Stack-allocated factorisation representation storing combined LU matrix and
/// row permutation index array.
pub fn LUStack(comptime N: usize, comptime T: type) type {
    return struct {
        lu: MatStack(N, N, T),
        permutation: [N]usize,
    };
}

/// Computes the in-place LU factorisation with partial pivoting `P A = L U`.
///
/// `a` is an N x N fixed-size matrix passed by value.
/// Returns `LUStack(N, T)` containing the packed LU matrix and permutation indices.
/// Performs no heap allocation.
///
/// Returns `error.NonFiniteInput` if any input element is NaN or Inf.
/// Returns `error.SingularMatrix` if any pivot is zero or singular.
pub fn factorLUStack(
    comptime N: usize,
    comptime T: type,
    a: MatStack(N, N, T),
) SolveError!LUStack(N, T) {
    if (!matops.allFiniteStack(N, N, T, a)) {
        return error.NonFiniteInput;
    }

    var result = LUStack(N, T){
        .lu = a,
        .permutation = undefined,
    };
    inline for (0..N) |ii| {
        result.permutation[ii] = ii;
    }

    const eps_tol: T = std.math.floatEps(T) * 1e-4;

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
            rowops.swapRowsStack(N, N, T, kk, pivot_row, &result.lu);
            const temp_perm = result.permutation[kk];
            result.permutation[kk] = result.permutation[pivot_row];
            result.permutation[pivot_row] = temp_perm;
        }

        const pivot_elem = result.lu.get(kk, kk);
        for ((kk + 1)..N) |rr| {
            const mult = result.lu.get(rr, kk) / pivot_elem;
            result.lu.set(rr, kk, mult);
            for ((kk + 1)..N) |cc| {
                const current_val = result.lu.get(rr, cc);
                result.lu.set(rr, cc, current_val - mult * result.lu.get(kk, cc));
            }
        }
    }

    for (0..N) |ii| {
        if (@abs(result.lu.get(ii, ii)) <= eps_tol) {
            return error.SingularMatrix;
        }
    }

    return result;
}

/// Solves linear system `A x = b` given a pre-computed factorisation `lu`.
///
/// `lu` contains the packed LU matrix and permutation array.
/// `b` is the right-hand side vector.
/// Performs forward substitution `L y = P b` followed by back substitution `U x = y`.
/// Returns the solution vector by value. Performs no heap allocation.
pub fn solveLUFactoredStack(
    comptime N: usize,
    comptime T: type,
    lu: LUStack(N, T),
    b: VecStack(N, T),
) SolveError!VecStack(N, T) {
    if (!vecops.allFiniteStack(N, T, b)) {
        return error.NonFiniteInput;
    }

    // Apply row permutation: y = P * b
    var y_vec: VecStack(N, T) = undefined;
    inline for (0..N) |ii| {
        y_vec.set(ii, b.get(lu.permutation[ii]));
    }

    // Forward substitution: L y = P b (unit diagonal on L)
    for (0..N) |ii| {
        var sum = y_vec.get(ii);
        for (0..ii) |jj| {
            sum -= lu.lu.get(ii, jj) * y_vec.get(jj);
        }
        y_vec.set(ii, sum);
    }

    // Backward substitution: U x = y
    var x_sol: VecStack(N, T) = undefined;
    var ii: usize = N;
    while (ii > 0) {
        ii -= 1;
        var sum = y_vec.get(ii);
        for ((ii + 1)..N) |jj| {
            sum -= lu.lu.get(ii, jj) * x_sol.get(jj);
        }
        x_sol.set(ii, sum / lu.lu.get(ii, ii));
    }

    return x_sol;
}

/// Convenience function that computes LU factorisation and solves `A x = b`.
///
/// Fixed-size value semantics; performs no heap allocation.
pub fn solveLUStack(
    comptime N: usize,
    comptime T: type,
    a: MatStack(N, N, T),
    b: VecStack(N, T),
) SolveError!VecStack(N, T) {
    const factored = try factorLUStack(N, T, a);
    return solveLUFactoredStack(N, T, factored, b);
}

/// Computes the in-place LU factorisation with partial pivoting for slices.
///
/// `a` is the N x N input matrix.
/// `lu_mat` (N x N) receives the packed L and U factors.
/// `perm` (length N) receives the pivot row permutation array.
///
/// Performs no heap allocation.
pub fn factorLUSlice(
    comptime T: type,
    a: MatSlice(T),
    lu_mat: MatSlice(T),
    perm: []usize,
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(lu_mat.rows_num == a.rows_num);
    assert(lu_mat.cols_num == a.cols_num);
    assert(perm.len == a.rows_num);

    if (!matops.allFiniteSlice(T, a)) {
        return error.NonFiniteInput;
    }

    const n_dim = a.rows_num;
    @memcpy(lu_mat.slice, a.slice);
    for (0..n_dim) |ii| {
        perm[ii] = ii;
    }

    const eps_tol: T = std.math.floatEps(T) * 1e-4;

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
            rowops.swapRowsSlice(T, kk, pivot_row, lu_mat);
            const temp_perm = perm[kk];
            perm[kk] = perm[pivot_row];
            perm[pivot_row] = temp_perm;
        }

        const pivot_elem = lu_mat.get(kk, kk);
        for ((kk + 1)..n_dim) |rr| {
            const mult = lu_mat.get(rr, kk) / pivot_elem;
            lu_mat.set(rr, kk, mult);
            for ((kk + 1)..n_dim) |cc| {
                const current_val = lu_mat.get(rr, cc);
                lu_mat.set(rr, cc, current_val - mult * lu_mat.get(kk, cc));
            }
        }
    }

    for (0..n_dim) |ii| {
        if (@abs(lu_mat.get(ii, ii)) <= eps_tol) {
            return error.SingularMatrix;
        }
    }
}

/// Solves linear system `A x = b` given pre-factored slice `lu_mat` and `perm`.
///
/// `work_vec` (length N) is caller-supplied scratch storage.
/// `out` (length N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveLUFactoredSlice(
    comptime T: type,
    lu_mat: MatSlice(T),
    perm: []const usize,
    b: VecSlice(T),
    work_vec: VecSlice(T),
    out: VecSlice(T),
) SolveError!void {
    assert(lu_mat.rows_num == lu_mat.cols_num);
    assert(perm.len == lu_mat.rows_num);
    assert(b.slice.len == lu_mat.rows_num);
    assert(work_vec.slice.len == lu_mat.rows_num);
    assert(out.slice.len == lu_mat.rows_num);

    if (!vecops.allFiniteSlice(T, b)) {
        return error.NonFiniteInput;
    }

    const n_dim = lu_mat.rows_num;

    // Apply permutation: work_vec = P * b
    for (0..n_dim) |ii| {
        work_vec.set(ii, b.get(perm[ii]));
    }

    // Forward substitution: L y = P b
    for (0..n_dim) |ii| {
        var sum = work_vec.get(ii);
        for (0..ii) |jj| {
            sum -= lu_mat.get(ii, jj) * work_vec.get(jj);
        }
        work_vec.set(ii, sum);
    }

    // Backward substitution: U x = y
    var ii: usize = n_dim;
    while (ii > 0) {
        ii -= 1;
        var sum = work_vec.get(ii);
        for ((ii + 1)..n_dim) |jj| {
            sum -= lu_mat.get(ii, jj) * out.get(jj);
        }
        out.set(ii, sum / lu_mat.get(ii, ii));
    }
}

/// Solves linear system `A x = b` via LU factorisation with caller-supplied scratch storage.
///
/// Performs no heap allocation.
pub fn solveLUSlice(
    comptime T: type,
    a: MatSlice(T),
    b: VecSlice(T),
    work_mat: MatSlice(T),
    perm: []usize,
    work_vec: VecSlice(T),
    out: VecSlice(T),
) SolveError!void {
    try factorLUSlice(T, a, work_mat, perm);
    try solveLUFactoredSlice(T, work_mat, perm, b, work_vec, out);
}

test "lu.factorLUStack and solveLUStack 2x2" {
    const a2 = MatStack(2, 2, f64).initRows(.{
        .{ 3.0, 1.0 },
        .{ 1.0, 2.0 },
    });
    const b2 = VecStack(2, f64).initSlice(&.{ 5.0, 5.0 });

    const lu_res = try factorLUStack(2, f64, a2);
    const x2 = try solveLUFactoredStack(2, f64, lu_res, b2);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x2.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x2.get(1), 1e-12);

    const x_conv = try solveLUStack(2, f64, a2, b2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x_conv.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x_conv.get(1), 1e-12);
}

test "lu.solveLUStack 3x3 requiring pivoting" {
    const a3 = MatStack(3, 3, f64).initRows(.{
        .{ 0.0, 2.0, 1.0 },
        .{ 3.0, -1.0, 2.0 },
        .{ 4.0, -1.0, 5.0 },
    });
    const b3 = VecStack(3, f64).initSlice(&.{ 5.0, 7.0, 15.0 });
    const x3 = try solveLUStack(3, f64, a3, b3);

    try std.testing.expectApproxEqAbs(@as(f64, 1.3076923076923077), x3.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3846153846153846), x3.get(1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2307692307692308), x3.get(2), 1e-12);
}

test "lu.solveLUSlice 3x3" {
    var a_raw = [_]f64{
        0.0, 2.0,  1.0,
        3.0, -1.0, 2.0,
        4.0, -1.0, 5.0,
    };
    var b_raw = [_]f64{ 5.0, 7.0, 15.0 };
    var work_mat_raw = [_]f64{0.0} ** 9;
    var perm_raw = [_]usize{0} ** 3;
    var work_vec_raw = [_]f64{0.0} ** 3;
    var out_raw = [_]f64{0.0} ** 3;

    try solveLUSlice(
        f64,
        MatSlice(f64).init(&a_raw, 3, 3),
        VecSlice(f64).init(&b_raw),
        MatSlice(f64).init(&work_mat_raw, 3, 3),
        &perm_raw,
        VecSlice(f64).init(&work_vec_raw),
        VecSlice(f64).init(&out_raw),
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.3076923076923077), out_raw[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.3846153846153846), out_raw[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2307692307692308), out_raw[2], 1e-12);
}

test "lu factorisation reconstruction P * A == L * U" {
    const a = MatStack(3, 3, f64).initRows(.{
        .{ 2.0, 1.0, 1.0 },
        .{ 4.0, -6.0, 0.0 },
        .{ -2.0, 7.0, 2.0 },
    });
    const factored = try factorLUStack(3, f64, a);

    // Reconstruct L and U
    var mat_l = MatStack(3, 3, f64).initZeros();
    var mat_u = MatStack(3, 3, f64).initZeros();

    inline for (0..3) |rr| {
        mat_l.set(rr, rr, 1.0);
        inline for (0..rr) |cc| {
            mat_l.set(rr, cc, factored.lu.get(rr, cc));
        }
        inline for (rr..3) |cc| {
            mat_u.set(rr, cc, factored.lu.get(rr, cc));
        }
    }

    const lu_prod = matops.mulMatStack(3, 3, 3, f64, mat_l, mat_u);

    // Permuted A: (P A)[i, j] = A[perm[i], j]
    for (0..3) |rr| {
        for (0..3) |cc| {
            const expected_pa = a.get(factored.permutation[rr], cc);
            try std.testing.expectApproxEqAbs(expected_pa, lu_prod.get(rr, cc), 1e-12);
        }
    }
}
