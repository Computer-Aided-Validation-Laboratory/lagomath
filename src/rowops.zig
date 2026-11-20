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

/// Swaps row `row_a` and row `row_b` in-place in fixed-size matrix `mat`.
///
/// Dimensions M and N are known at compile time.
/// Mutates `mat` in place via pointer. Performs no heap allocation.
pub fn swapRowsStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    row_a: usize,
    row_b: usize,
    mat: *Mat(M, N, T),
) void {
    assert(row_a < M);
    assert(row_b < M);
    if (row_a == row_b) {
        return;
    }
    std.mem.swap([N]T, &mat.mat[row_a], &mat.mat[row_b]);
}

/// Swaps row `row_a` and row `row_b` in-place in borrowed slice matrix `mat`.
///
/// Asserts `row_a < mat.rows_num` and `row_b < mat.rows_num`.
/// Storage is borrowed; performs no heap allocation.
pub fn swapRowsSlice(
    comptime T: type,
    row_a: usize,
    row_b: usize,
    mat: MatView(T),
) void {
    assert(row_a < mat.rows_num);
    assert(row_b < mat.rows_num);
    if (row_a == row_b) {
        return;
    }
    for (0..mat.cols_num) |cc| {
        const temp = mat.get(row_a, cc);
        mat.set(row_a, cc, mat.get(row_b, cc));
        mat.set(row_b, cc, temp);
    }
}

/// Scales row `row` by scalar `scale` in-place in fixed-size matrix `mat`.
///
/// Dimensions M and N are known at compile time.
/// Mutates `mat` in place via pointer. Performs no heap allocation.
pub fn scaleRowStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    row: usize,
    scale: T,
    mat: *Mat(M, N, T),
) void {
    assert(row < M);
    inline for (0..N) |cc| {
        mat.set(row, cc, mat.get(row, cc) * scale);
    }
}

/// Scales row `row` by scalar `scale` in-place in borrowed slice matrix `mat`.
///
/// Asserts `row < mat.rows_num`.
/// Storage is borrowed; performs no heap allocation.
pub fn scaleRowSlice(
    comptime T: type,
    row: usize,
    scale: T,
    mat: MatView(T),
) void {
    assert(row < mat.rows_num);
    for (0..mat.cols_num) |cc| {
        mat.set(row, cc, mat.get(row, cc) * scale);
    }
}

/// Adds scaled row `src_row * scale` to `dst_row` in fixed-size matrix `mat`.
///
/// Equivalent to `row[dst] = row[dst] + scale * row[src]`.
/// Dimensions M and N are known at compile time.
/// Mutates `mat` in place via pointer. Performs no heap allocation.
pub fn addScaledRowStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    src_row: usize,
    dst_row: usize,
    scale: T,
    mat: *Mat(M, N, T),
) void {
    assert(src_row < M);
    assert(dst_row < M);
    inline for (0..N) |cc| {
        const val = mat.get(dst_row, cc) + scale * mat.get(src_row, cc);
        mat.set(dst_row, cc, val);
    }
}

/// Adds scaled row `src_row * scale` to `dst_row` in borrowed slice matrix `mat`.
///
/// Asserts `src_row < mat.rows_num` and `dst_row < mat.rows_num`.
/// Storage is borrowed; performs no heap allocation.
pub fn addScaledRowSlice(
    comptime T: type,
    src_row: usize,
    dst_row: usize,
    scale: T,
    mat: MatView(T),
) void {
    assert(src_row < mat.rows_num);
    assert(dst_row < mat.rows_num);
    for (0..mat.cols_num) |cc| {
        const val = mat.get(dst_row, cc) + scale * mat.get(src_row, cc);
        mat.set(dst_row, cc, val);
    }
}

/// Applies a permutation index array `perm` to vector `vec`, returning permuted vector.
///
/// `out.slice[ii] = vec.slice[perm[ii]]`.
/// Dimensions are known at compile time.
/// Value semantics; performs no heap allocation.
pub fn applyPermutationStack(
    comptime N: usize,
    comptime T: type,
    perm: [N]usize,
    vec: Vec(N, T),
) Vec(N, T) {
    var result: Vec(N, T) = undefined;
    inline for (0..N) |ii| {
        assert(perm[ii] < N);
        result.vec[ii] = vec.vec[perm[ii]];
    }
    return result;
}

/// Applies a permutation index array `perm` to `src`, writing permuted elements to `out`.
///
/// Asserts matching slice lengths.
/// Writes directly to caller-provided `out`. Performs no heap allocation.
pub fn applyPermutationSlice(
    comptime T: type,
    perm: []const usize,
    src: VecView(T),
    out: VecView(T),
) void {
    assert(perm.len == src.slice.len);
    assert(src.slice.len == out.slice.len);
    for (0..perm.len) |ii| {
        assert(perm[ii] < src.slice.len);
        out.slice[ii] = src.slice[perm[ii]];
    }
}

test "rowops.swapRowsStack and swapRowsSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.rowops.rowops_swapRowsStack_and_swapRowsSlice");
}

test "rowops.scaleRowStack and scaleRowSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.rowops.rowops_scaleRowStack_and_scaleRowSlice");
}

test "rowops.addScaledRowStack and addScaledRowSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.rowops.rowops_addScaledRowStack_and_addScaledRowSlice");
}

test "rowops.applyPermutation" {
    try @import("case_runner.zig").verifyLegacy("legacy.rowops.rowops_applyPermutation");
}

pub const swapRowsMat = swapRowsStack;

pub const swapRowsMatView = swapRowsSlice;

pub const scaleRowMat = scaleRowStack;

pub const scaleRowMatView = scaleRowSlice;

pub const addScaledRowMat = addScaledRowStack;

pub const addScaledRowMatView = addScaledRowSlice;

pub const applyPermutationVec = applyPermutationStack;

pub const applyPermutationVecView = applyPermutationSlice;
