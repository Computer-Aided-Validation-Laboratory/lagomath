//! Explicit SIMD row-major row operations.
const std = @import("std");
const assert = std.debug.assert;
const core = @import("sliceops_simd.zig");
const Mat = @import("mattype.zig").Mat;
const MatView = @import("matviewtype.zig").MatView;
pub fn swapRowsMatViewSIMD(comptime T: type, a: usize, b: usize, m: MatView(T)) void {
    assert(a < m.rows_num and b < m.rows_num);
    core.swapSIMD(T, m.slice[a * m.cols_num ..][0..m.cols_num], m.slice[b * m.cols_num ..][0..m.cols_num]);
}
pub fn scaleRowMatViewSIMD(comptime T: type, row: usize, alpha: T, m: MatView(T)) void {
    assert(row < m.rows_num);
    const a = m.slice[row * m.cols_num ..][0..m.cols_num];
    core.mulScalSIMD(T, a, alpha, a);
}
pub fn addScaledRowMatViewSIMD(comptime T: type, src: usize, dst: usize, alpha: T, m: MatView(T)) void {
    assert(src < m.rows_num and dst < m.rows_num);
    const a = m.slice[src * m.cols_num ..][0..m.cols_num];
    const b = m.slice[dst * m.cols_num ..][0..m.cols_num];
    core.axpySIMD(T, alpha, a, b, b);
}
pub fn swapRowsMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: usize, b: usize, m: *Mat(R, C, T)) void {
    core.swapSIMD(T, &m.mat[a], &m.mat[b]);
}
pub fn scaleRowMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, row: usize, alpha: T, m: *Mat(R, C, T)) void {
    core.mulScalSIMD(T, &m.mat[row], alpha, &m.mat[row]);
}
pub fn addScaledRowMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, src: usize, dst: usize, alpha: T, m: *Mat(R, C, T)) void {
    core.axpySIMD(T, alpha, &m.mat[src], &m.mat[dst], &m.mat[dst]);
}
pub fn applyPermutationVecViewSIMD(comptime T: type, perm: []const usize, a: @import("vecviewtype.zig").VecView(T), out: @import("vecviewtype.zig").VecView(T)) void {
    assert(perm.len == a.slice.len and out.slice.len == a.slice.len);
    const W = 32 / @sizeOf(T);
    var i: usize = 0;
    while (perm.len - i >= W) : (i += W) {
        var v: @Vector(W, T) = undefined;
        inline for (0..W) |j| {
            assert(perm[i + j] < a.slice.len);
            v[j] = a.slice[perm[i + j]];
        }
        out.slice[i..][0..W].* = v;
    }
    for (i..perm.len) |j| {
        assert(perm[j] < a.slice.len);
        out.slice[j] = a.slice[perm[j]];
    }
}
pub fn applyPermutationVecSIMD(comptime N: usize, comptime T: type, perm: [N]usize, a: @import("vectype.zig").Vec(N, T)) @import("vectype.zig").Vec(N, T) {
    const W = 32 / @sizeOf(T);
    var out: @import("vectype.zig").Vec(N, T) = undefined;
    var i: usize = 0;
    while (N - i >= W) : (i += W) {
        var v: @Vector(W, T) = undefined;
        inline for (0..W) |j| {
            assert(perm[i + j] < N);
            v[j] = a.vec[perm[i + j]];
        }
        out.vec[i..][0..W].* = v;
    }
    for (i..N) |j| {
        assert(perm[j] < N);
        out.vec[j] = a.vec[perm[j]];
    }
    return out;
}
