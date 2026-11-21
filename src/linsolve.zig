// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const Mat = @import("mattype.zig").Mat;
const Vec = @import("vectype.zig").Vec;
const MatView = @import("matviewtype.zig").MatView;
const VecView = @import("vecviewtype.zig").VecView;

pub const SolveError = error{
    ShapeMismatch,
    InvalidPermutation,
    InvalidAliasing,
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
    try @import("case_runner.zig").verifyLegacy("legacy.linsolve.linsolve_cross_solver_verification_2x2_SPD_system");
}

test "linsolve representative 6x6 DIC SPD Hessian solve" {
    try @import("case_runner.zig").verifyLegacy("legacy.linsolve.linsolve_representative_6x6_DIC_SPD_Hessian_solve");
}

test "linsolve factorisation reuse with multiple right-hand sides" {
    try @import("case_runner.zig").verifyLegacy("legacy.linsolve.linsolve_factorisation_reuse_with_multiple_right_hand_sides");
}

pub const solveGaussMat = gauss.solveMat;
pub const solveGaussMatView = gauss.solveMatView;
pub const solveGaussStackChecked = gauss.solveStackChecked;
pub const solveGaussSliceChecked = gauss.solveSliceChecked;
pub const solveGaussMatChecked = gauss.solveMatChecked;
pub const solveGaussMatViewChecked = gauss.solveMatViewChecked;
pub const factorLUMat = lu.factorLUMat;
pub const solveLUFactoredMat = lu.solveLUFactoredMat;
pub const solveLUMat = lu.solveLUMat;
pub const factorLUMatView = lu.factorLUMatView;
pub const solveLUFactoredMatView = lu.solveLUFactoredMatView;
pub const solveLUMatView = lu.solveLUMatView;
pub const solveLUFactoredStackChecked = lu.solveLUFactoredStackChecked;
pub const solveLUStackChecked = lu.solveLUStackChecked;
pub const factorLUSliceChecked = lu.factorLUSliceChecked;
pub const solveLUFactoredSliceChecked = lu.solveLUFactoredSliceChecked;
pub const solveLUSliceChecked = lu.solveLUSliceChecked;
pub const solveLUFactoredMatChecked = lu.solveLUFactoredMatChecked;
pub const solveLUMatChecked = lu.solveLUMatChecked;
pub const factorLUMatViewChecked = lu.factorLUMatViewChecked;
pub const solveLUFactoredMatViewChecked = lu.solveLUFactoredMatViewChecked;
pub const solveLUMatViewChecked = lu.solveLUMatViewChecked;
pub const factorCholeskyMat = cholesky.factorCholeskyMat;
pub const solveCholeskyFactoredMat = cholesky.solveCholeskyFactoredMat;
pub const solveCholeskyMat = cholesky.solveCholeskyMat;
pub const factorCholeskyMatView = cholesky.factorCholeskyMatView;
pub const solveCholeskyFactoredMatView = cholesky.solveCholeskyFactoredMatView;
pub const solveCholeskyMatView = cholesky.solveCholeskyMatView;
pub const solveCholeskyFactoredStackChecked = cholesky.solveCholeskyFactoredStackChecked;
pub const solveCholeskyStackChecked = cholesky.solveCholeskyStackChecked;
pub const factorCholeskySliceChecked = cholesky.factorCholeskySliceChecked;
pub const solveCholeskyFactoredSliceChecked = cholesky.solveCholeskyFactoredSliceChecked;
pub const solveCholeskySliceChecked = cholesky.solveCholeskySliceChecked;
pub const solveCholeskyFactoredMatChecked = cholesky.solveCholeskyFactoredMatChecked;
pub const solveCholeskyMatChecked = cholesky.solveCholeskyMatChecked;
pub const factorCholeskyMatViewChecked = cholesky.factorCholeskyMatViewChecked;
pub const solveCholeskyFactoredMatViewChecked = cholesky.solveCholeskyFactoredMatViewChecked;
pub const solveCholeskyMatViewChecked = cholesky.solveCholeskyMatViewChecked;
pub const factorLUMatChecked = lu.factorLUMatChecked;
pub const factorCholeskyMatChecked = cholesky.factorCholeskyMatChecked;

pub const LU = lu.LUStack;
pub const Cholesky = cholesky.CholeskyStack;
