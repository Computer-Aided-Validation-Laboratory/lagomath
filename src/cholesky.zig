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
const matops = @import("matops.zig");
const vecops = @import("vecops.zig");
const SolveError = @import("linsolve.zig").SolveError;

/// Stack-allocated factorisation representation storing the lower triangular Cholesky factor.
pub fn CholeskyStack(comptime N: usize, comptime T: type) type {
    return struct {
        lower: MatStack(N, N, T),
    };
}

/// Computes the Cholesky factorisation `A = L L^T` for a symmetric positive-definite matrix.
///
/// `a` is an N x N fixed-size matrix passed by value.
/// Returns `CholeskyStack(N, T)` containing the lower triangular factor `L`.
/// Performs no heap allocation.
///
/// Returns `error.NonFiniteInput` if any input element is NaN or Inf.
/// Returns `error.NotPositiveDefinite` if `A` is not positive definite.
pub fn factorCholeskyStack(
    comptime N: usize,
    comptime T: type,
    a: MatStack(N, N, T),
) SolveError!CholeskyStack(N, T) {
    if (!matops.allFiniteStack(N, N, T, a)) {
        return error.NonFiniteInput;
    }

    var result = CholeskyStack(N, T){
        .lower = MatStack(N, N, T).initZeros(),
    };

    for (0..N) |ii| {
        for (0..(ii + 1)) |jj| {
            var sum = a.get(ii, jj);
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
    b: VecStack(N, T),
) SolveError!VecStack(N, T) {
    if (!vecops.allFiniteStack(N, T, b)) {
        return error.NonFiniteInput;
    }

    // Forward substitution: L y = b
    var y_vec: VecStack(N, T) = undefined;
    for (0..N) |ii| {
        var sum = b.get(ii);
        for (0..ii) |jj| {
            sum -= chol.lower.get(ii, jj) * y_vec.get(jj);
        }
        y_vec.set(ii, sum / chol.lower.get(ii, ii));
    }

    // Backward substitution: L^T x = y
    var x_sol: VecStack(N, T) = undefined;
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
/// Returns `error.NonFiniteInput` if any input element is NaN or Inf.
/// Returns `error.NotPositiveDefinite` if `a` is not positive definite.
pub fn solveCholeskyStack(
    comptime N: usize,
    comptime T: type,
    a: MatStack(N, N, T),
    b: VecStack(N, T),
) SolveError!VecStack(N, T) {
    const chol = try factorCholeskyStack(N, T, a);
    return solveCholeskyFactoredStack(N, T, chol, b);
}

/// Computes the Cholesky factorisation `A = L L^T` for borrowed slices.
///
/// `a` is the N x N input matrix.
/// `lower` (N x N) receives the lower triangular factor `L` with upper triangle zeroed.
///
/// Performs no heap allocation.
/// Returns `error.NonFiniteInput` if any input element is NaN or Inf.
/// Returns `error.NotPositiveDefinite` if `a` is not positive definite.
pub fn factorCholeskySlice(
    comptime T: type,
    a: MatSlice(T),
    lower: MatSlice(T),
) SolveError!void {
    assert(a.rows_num == a.cols_num);
    assert(lower.rows_num == a.rows_num);
    assert(lower.cols_num == a.cols_num);

    if (!matops.allFiniteSlice(T, a)) {
        return error.NonFiniteInput;
    }

    const n_dim = a.rows_num;
    lower.fill(0);

    for (0..n_dim) |ii| {
        for (0..(ii + 1)) |jj| {
            var sum = a.get(ii, jj);
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
}

/// Solves linear system `A x = b` given pre-factored Cholesky slice `lower`.
///
/// `work_vec` (length N) is caller-supplied scratch storage.
/// `out` (length N) receives the solution vector.
///
/// Performs no heap allocation.
pub fn solveCholeskyFactoredSlice(
    comptime T: type,
    lower: MatSlice(T),
    b: VecSlice(T),
    work_vec: VecSlice(T),
    out: VecSlice(T),
) SolveError!void {
    assert(lower.rows_num == lower.cols_num);
    assert(b.slice.len == lower.rows_num);
    assert(work_vec.slice.len == lower.rows_num);
    assert(out.slice.len == lower.rows_num);

    if (!vecops.allFiniteSlice(T, b)) {
        return error.NonFiniteInput;
    }

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
    a: MatSlice(T),
    b: VecSlice(T),
    work_mat: MatSlice(T),
    work_vec: VecSlice(T),
    out: VecSlice(T),
) SolveError!void {
    try factorCholeskySlice(T, a, work_mat);
    try solveCholeskyFactoredSlice(T, work_mat, b, work_vec, out);
}

test "cholesky.factorCholeskyStack and solveCholeskyStack 2x2" {
    const a2 = MatStack(2, 2, f64).initRows(.{
        .{ 3.0, 1.0 },
        .{ 1.0, 2.0 },
    });
    const b2 = VecStack(2, f64).initSlice(&.{ 5.0, 5.0 });

    const chol = try factorCholeskyStack(2, f64, a2);
    // Reference L from python:
    // [1.7320508075688772, 0.0]
    // [0.5773502691896258, 1.2909944487358056]
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.7320508075688772),
        chol.lower.get(0, 0),
        1e-12,
    );
    try std.testing.expectEqual(@as(f64, 0.0), chol.lower.get(0, 1));
    try std.testing.expectApproxEqAbs(
        @as(f64, 0.5773502691896258),
        chol.lower.get(1, 0),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.2909944487358056),
        chol.lower.get(1, 1),
        1e-12,
    );

    const x2 = try solveCholeskyFactoredStack(2, f64, chol, b2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x2.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x2.get(1), 1e-12);

    const x_conv = try solveCholeskyStack(2, f64, a2, b2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x_conv.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x_conv.get(1), 1e-12);
}

test "cholesky.factorCholeskyStack rejects non-positive-definite" {
    // Indefinite matrix
    const a_indef = MatStack(2, 2, f64).initRows(.{
        .{ 1.0, 2.0 },
        .{ 2.0, 1.0 },
    });
    try std.testing.expectError(
        error.NotPositiveDefinite,
        factorCholeskyStack(2, f64, a_indef),
    );

    // Negative diagonal element
    const a_neg = MatStack(2, 2, f64).initRows(.{
        .{ -2.0, 0.0 },
        .{ 0.0, 2.0 },
    });
    try std.testing.expectError(
        error.NotPositiveDefinite,
        factorCholeskyStack(2, f64, a_neg),
    );
}

test "cholesky.solveCholeskySlice 2x2" {
    var a_raw = [_]f64{ 3.0, 1.0, 1.0, 2.0 };
    var b_raw = [_]f64{ 5.0, 5.0 };
    var work_mat_raw = [_]f64{0.0} ** 4;
    var work_vec_raw = [_]f64{0.0} ** 2;
    var out_raw = [_]f64{0.0} ** 2;

    try solveCholeskySlice(
        f64,
        MatSlice(f64).init(&a_raw, 2, 2),
        VecSlice(f64).init(&b_raw),
        MatSlice(f64).init(&work_mat_raw, 2, 2),
        VecSlice(f64).init(&work_vec_raw),
        VecSlice(f64).init(&out_raw),
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), out_raw[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), out_raw[1], 1e-12);
}

test "cholesky factorisation reconstruction A == L * L^T" {
    const a = MatStack(3, 3, f64).initRows(.{
        .{ 4.0, 12.0, -16.0 },
        .{ 12.0, 37.0, -43.0 },
        .{ -16.0, -43.0, 98.0 },
    });
    const chol = try factorCholeskyStack(3, f64, a);

    // Reconstruct L * L^T
    for (0..3) |rr| {
        for (0..3) |cc| {
            var sum: f64 = 0;
            for (0..3) |kk| {
                sum += chol.lower.get(rr, kk) * chol.lower.get(cc, kk);
            }
            try std.testing.expectApproxEqAbs(a.get(rr, cc), sum, 1e-12);
        }
    }
}
