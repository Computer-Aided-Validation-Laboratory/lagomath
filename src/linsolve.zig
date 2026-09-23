// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const MatStack = @import("matstack.zig").MatStack;
const VecStack = @import("vecstack.zig").VecStack;
const MatSlice = @import("matslice.zig").MatSlice;
const VecSlice = @import("vecslice.zig").VecSlice;

pub const SolveError = error{
    SingularMatrix,
    NotPositiveDefinite,
    NonFiniteInput,
};

pub const gauss = @import("gauss.zig");
pub const lu = @import("lu.zig");
pub const cholesky = @import("cholesky.zig");

// Public solver re-exports
pub const solveGaussStack = gauss.solveStack;
pub const solveGaussSlice = gauss.solveSlice;

pub const LUStack = lu.LUStack;
pub const factorLUStack = lu.factorLUStack;
pub const factorLUSlice = lu.factorLUSlice;
pub const solveLUFactoredStack = lu.solveLUFactoredStack;
pub const solveLUFactoredSlice = lu.solveLUFactoredSlice;
pub const solveLUStack = lu.solveLUStack;
pub const solveLUSlice = lu.solveLUSlice;

pub const CholeskyStack = cholesky.CholeskyStack;
pub const factorCholeskyStack = cholesky.factorCholeskyStack;
pub const factorCholeskySlice = cholesky.factorCholeskySlice;
pub const solveCholeskyFactoredStack = cholesky.solveCholeskyFactoredStack;
pub const solveCholeskyFactoredSlice = cholesky.solveCholeskyFactoredSlice;
pub const solveCholeskyStack = cholesky.solveCholeskyStack;
pub const solveCholeskySlice = cholesky.solveCholeskySlice;

test {
    _ = gauss;
    _ = lu;
    _ = cholesky;
}

test "linsolve cross-solver verification 2x2 SPD system" {
    const a2 = MatStack(2, 2, f64).initRows(.{
        .{ 3.0, 1.0 },
        .{ 1.0, 2.0 },
    });
    const b2 = VecStack(2, f64).initSlice(&.{ 5.0, 5.0 });

    const x_gauss = try solveGaussStack(2, f64, a2, b2);
    const x_lu = try solveLUStack(2, f64, a2, b2);
    const x_chol = try solveCholeskyStack(2, f64, a2, b2);

    inline for (0..2) |ii| {
        try std.testing.expectApproxEqAbs(x_gauss.get(ii), x_lu.get(ii), 1e-12);
        try std.testing.expectApproxEqAbs(x_gauss.get(ii), x_chol.get(ii), 1e-12);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x_chol.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x_chol.get(1), 1e-12);
}

test "linsolve representative 6x6 DIC SPD Hessian solve" {
    // 6x6 SPD system generated via NumPy / SciPy reference (A = J^T J + 0.1 I)
    const a6 = MatStack(6, 6, f64).initRows(.{
        .{
            6.83618531024444298e+00,
            1.32071154964033077e+00,
            -3.05466448304476890e+00,
            2.12284855465204636e+00,
            -2.16853351080305590e+00,
            1.49169925123400349e+00,
        },
        .{
            1.32071154964033077e+00,
            2.71515105003049406e+01,
            -1.64741096885242144e+00,
            -1.16240166056642225e+00,
            -1.07636629904745318e+00,
            -2.05957914591337543e+00,
        },
        .{
            -3.05466448304476890e+00,
            -1.64741096885242144e+00,
            2.52536093351211974e+01,
            1.30464019262404451e+00,
            -2.02259038545519576e-01,
            -7.95463221720539249e+00,
        },
        .{
            2.12284855465204636e+00,
            -1.16240166056642225e+00,
            1.30464019262404451e+00,
            9.59224787901177045e+00,
            -1.40876825391276594e+00,
            -3.71900238432621233e+00,
        },
        .{
            -2.16853351080305590e+00,
            -1.07636629904745318e+00,
            -2.02259038545519576e-01,
            -1.40876825391276594e+00,
            1.40503884159721348e+01,
            1.67600268321302015e+00,
        },
        .{
            1.49169925123400349e+00,
            -2.05957914591337543e+00,
            -7.95463221720539249e+00,
            -3.71900238432621233e+00,
            1.67600268321302015e+00,
            2.03572954874524079e+01,
        },
    });

    const b6 = VecStack(6, f64).initSlice(&.{
        1.2,
        -0.5,
        3.4,
        -2.1,
        0.8,
        1.5,
    });

    // Reference solution computed by NumPy:
    const x6_ref = [_]f64{
        3.80783555962461218e-01,
        -2.85003939561861708e-02,
        2.15729555340055057e-01,
        -2.98784875014349860e-01,
        7.87846704255997371e-02,
        6.61242515234334305e-02,
    };

    const x_gauss = try solveGaussStack(6, f64, a6, b6);
    const x_lu = try solveLUStack(6, f64, a6, b6);
    const x_chol = try solveCholeskyStack(6, f64, a6, b6);

    inline for (0..6) |ii| {
        try std.testing.expectApproxEqAbs(x6_ref[ii], x_gauss.get(ii), 1e-12);
        try std.testing.expectApproxEqAbs(x6_ref[ii], x_lu.get(ii), 1e-12);
        try std.testing.expectApproxEqAbs(x6_ref[ii], x_chol.get(ii), 1e-12);
        try std.testing.expectApproxEqAbs(x_gauss.get(ii), x_chol.get(ii), 1e-12);
    }
}

test "linsolve factorisation reuse with multiple right-hand sides" {
    const a = MatStack(2, 2, f64).initRows(.{
        .{ 4.0, 1.0 },
        .{ 1.0, 3.0 },
    });
    const b1 = VecStack(2, f64).initSlice(&.{ 1.0, 2.0 });
    const b2 = VecStack(2, f64).initSlice(&.{ 3.0, 4.0 });

    // LU reuse
    const lu_factors = try factorLUStack(2, f64, a);
    const x_lu1 = try solveLUFactoredStack(2, f64, lu_factors, b1);
    const x_lu2 = try solveLUFactoredStack(2, f64, lu_factors, b2);

    // Cholesky reuse
    const chol_factors = try factorCholeskyStack(2, f64, a);
    const x_chol1 = try solveCholeskyFactoredStack(2, f64, chol_factors, b1);
    const x_chol2 = try solveCholeskyFactoredStack(2, f64, chol_factors, b2);

    inline for (0..2) |ii| {
        try std.testing.expectApproxEqAbs(x_lu1.get(ii), x_chol1.get(ii), 1e-12);
        try std.testing.expectApproxEqAbs(x_lu2.get(ii), x_chol2.get(ii), 1e-12);
    }
}
