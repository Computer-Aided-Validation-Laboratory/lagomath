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
    mat: *MatStack(M, N, T),
) void {
    assert(row_a < M);
    assert(row_b < M);
    if (row_a == row_b) {
        return;
    }
    inline for (0..N) |cc| {
        const temp = mat.get(row_a, cc);
        mat.set(row_a, cc, mat.get(row_b, cc));
        mat.set(row_b, cc, temp);
    }
}

/// Swaps row `row_a` and row `row_b` in-place in borrowed slice matrix `mat`.
///
/// Asserts `row_a < mat.rows_num` and `row_b < mat.rows_num`.
/// Storage is borrowed; performs no heap allocation.
pub fn swapRowsSlice(
    comptime T: type,
    row_a: usize,
    row_b: usize,
    mat: MatSlice(T),
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
    mat: *MatStack(M, N, T),
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
    mat: MatSlice(T),
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
    mat: *MatStack(M, N, T),
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
    mat: MatSlice(T),
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
    vec: VecStack(N, T),
) VecStack(N, T) {
    var result: VecStack(N, T) = undefined;
    inline for (0..N) |ii| {
        assert(perm[ii] < N);
        result.slice[ii] = vec.slice[perm[ii]];
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
    src: VecSlice(T),
    out: VecSlice(T),
) void {
    assert(perm.len == src.slice.len);
    assert(src.slice.len == out.slice.len);
    for (0..perm.len) |ii| {
        assert(perm[ii] < src.slice.len);
        out.slice[ii] = src.slice[perm[ii]];
    }
}

test "rowops.swapRowsStack and swapRowsSlice" {
    var mat = MatStack(3, 2, f64).initRows(.{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
        .{ 5.0, 6.0 },
    });
    swapRowsStack(3, 2, f64, 0, 2, &mat);
    try std.testing.expectEqual(@as(f64, 5.0), mat.get(0, 0));
    try std.testing.expectEqual(@as(f64, 6.0), mat.get(0, 1));
    try std.testing.expectEqual(@as(f64, 1.0), mat.get(2, 0));
    try std.testing.expectEqual(@as(f64, 2.0), mat.get(2, 1));

    var raw = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const s_mat = MatSlice(f64).init(&raw, 3, 2);
    swapRowsSlice(f64, 0, 2, s_mat);
    try std.testing.expectEqual(@as(f64, 5.0), s_mat.get(0, 0));
    try std.testing.expectEqual(@as(f64, 6.0), s_mat.get(0, 1));
    try std.testing.expectEqual(@as(f64, 1.0), s_mat.get(2, 0));
    try std.testing.expectEqual(@as(f64, 2.0), s_mat.get(2, 1));
}

test "rowops.scaleRowStack and scaleRowSlice" {
    var mat = MatStack(2, 2, f64).initRows(.{
        .{ 2.0, 4.0 },
        .{ 1.0, 3.0 },
    });
    scaleRowStack(2, 2, f64, 0, 0.5, &mat);
    try std.testing.expectEqual(@as(f64, 1.0), mat.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), mat.get(0, 1));

    var raw = [_]f64{ 2.0, 4.0, 1.0, 3.0 };
    const s_mat = MatSlice(f64).init(&raw, 2, 2);
    scaleRowSlice(f64, 0, 0.5, s_mat);
    try std.testing.expectEqual(@as(f64, 1.0), s_mat.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), s_mat.get(0, 1));
}

test "rowops.addScaledRowStack and addScaledRowSlice" {
    var mat = MatStack(2, 2, f64).initRows(.{
        .{ 2.0, 3.0 },
        .{ 4.0, 5.0 },
    });
    // dst row 1 += -2 * row 0 => 4 + (-2)*2 = 0; 5 + (-2)*3 = -1
    addScaledRowStack(2, 2, f64, 0, 1, -2.0, &mat);
    try std.testing.expectEqual(@as(f64, 0.0), mat.get(1, 0));
    try std.testing.expectEqual(@as(f64, -1.0), mat.get(1, 1));

    var raw = [_]f64{ 2.0, 3.0, 4.0, 5.0 };
    const s_mat = MatSlice(f64).init(&raw, 2, 2);
    addScaledRowSlice(f64, 0, 1, -2.0, s_mat);
    try std.testing.expectEqual(@as(f64, 0.0), s_mat.get(1, 0));
    try std.testing.expectEqual(@as(f64, -1.0), s_mat.get(1, 1));
}

test "rowops.applyPermutation" {
    const perm = [_]usize{ 2, 0, 1 };
    const vec = VecStack(3, f64).initSlice(&.{ 10.0, 20.0, 30.0 });
    const permuted = applyPermutationStack(3, f64, perm, vec);

    try std.testing.expectEqual(@as(f64, 30.0), permuted.get(0));
    try std.testing.expectEqual(@as(f64, 10.0), permuted.get(1));
    try std.testing.expectEqual(@as(f64, 20.0), permuted.get(2));

    var raw_src = [_]f64{ 10.0, 20.0, 30.0 };
    var raw_out = [_]f64{ 0.0, 0.0, 0.0 };
    applyPermutationSlice(
        f64,
        &perm,
        VecSlice(f64).init(&raw_src),
        VecSlice(f64).init(&raw_out),
    );
    try std.testing.expectEqual(@as(f64, 30.0), raw_out[0]);
    try std.testing.expectEqual(@as(f64, 10.0), raw_out[1]);
    try std.testing.expectEqual(@as(f64, 20.0), raw_out[2]);
}
