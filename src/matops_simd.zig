//! Opt-in row-major kernels. Outputs of products/transpose must be disjoint
//! from inputs; caller supplies all storage. No hidden allocations.
const std = @import("std");
const assert = std.debug.assert;
const core = @import("sliceops_simd.zig");
const Mat = @import("mattype.zig").Mat;
const Vec = @import("vectype.zig").Vec;
const MatView = @import("matviewtype.zig").MatView;
const VecView = @import("vecviewtype.zig").VecView;
fn flatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: *const Mat(R, C, T)) []const T {
    return @as([*]const T, @ptrCast(&a.mat))[0 .. R * C];
}
fn mutableFlatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: *Mat(R, C, T)) []T {
    return @as([*]T, @ptrCast(&a.mat))[0 .. R * C];
}
pub fn mulMatSIMD(comptime T: type, r: usize, k: usize, c: usize, a: []const T, b: []const T, out: []T) void {
    assert(a.len >= r * k and b.len >= k * c and out.len >= r * c);
    const W = 32 / @sizeOf(T);
    var row: usize = 0;
    while (row < r) : (row += 4) {
        var col: usize = 0;
        while (c - col >= W) : (col += W) {
            var acc: [4]@Vector(W, T) = @splat(@as(@Vector(W, T), @splat(0)));
            for (0..k) |inner| {
                const bv: @Vector(W, T) = b[inner * c + col ..][0..W].*;
                inline for (0..4) |lane| {
                    if (row + lane < r) acc[lane] += @as(@Vector(W, T), @splat(a[(row + lane) * k + inner])) * bv;
                }
            }
            inline for (0..4) |lane| {
                if (row + lane < r) out[(row + lane) * c + col ..][0..W].* = acc[lane];
            }
        }
        while (col < c) : (col += 1) {
            for (row..@min(row + 4, r)) |rr| {
                var v: T = 0;
                for (0..k) |inner| v += a[rr * k + inner] * b[inner * c + col];
                out[rr * c + col] = v;
            }
        }
    }
}
pub fn mulMatMatViewSIMD(comptime T: type, a: MatView(T), b: MatView(T), out: MatView(T)) void {
    assert(a.cols_num == b.rows_num and out.rows_num == a.rows_num and out.cols_num == b.cols_num);
    mulMatSIMD(T, a.rows_num, a.cols_num, b.cols_num, a.slice, b.slice, out.slice);
}
pub fn mulMatMatSIMD(comptime R: usize, comptime K: usize, comptime C: usize, comptime T: type, a: Mat(R, K, T), b: Mat(K, C, T)) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    mulMatSIMD(T, R, K, C, flatSIMD(R, K, T, &a), flatSIMD(K, C, T, &b), mutableFlatSIMD(R, C, T, &out));
    return out;
}
pub fn mulVecMatViewSIMD(comptime T: type, a: MatView(T), b: VecView(T), out: VecView(T)) void {
    assert(a.cols_num == b.slice.len and a.rows_num == out.slice.len);
    for (0..a.rows_num) |r| out.slice[r] = core.dotSIMD(T, a.slice[r * a.cols_num ..][0..a.cols_num], b.slice);
}
pub fn mulVecMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T), b: Vec(C, T)) Vec(R, T) {
    var out: Vec(R, T) = undefined;
    for (0..R) |r| out.vec[r] = core.dotSIMD(T, &a.mat[r], &b.vec);
    return out;
}
pub fn transposeSIMD(comptime T: type, r: usize, c: usize, a: []const T, out: []T) void {
    assert(a.len >= r * c and out.len >= r * c);
    const W = 32 / @sizeOf(T);
    // A vector tile uses register shuffles and contiguous reads/writes.
    var rr: usize = 0;
    while (r - rr >= W) : (rr += W) {
        var cc: usize = 0;
        while (c - cc >= W) : (cc += W) {
            var tile: [W]@Vector(W, T) = undefined;
            inline for (0..W) |i| tile[i] = a[(rr + i) * c + cc ..][0..W].*;
            inline for (0..W) |j| {
                var v: @Vector(W, T) = undefined;
                inline for (0..W) |i| v[i] = tile[i][j];
                out[(cc + j) * r + rr ..][0..W].* = v;
            }
        }
        for (cc..c) |j| for (0..W) |i| {
            out[j * r + rr + i] = a[(rr + i) * c + j];
        };
    }
    for (rr..r) |i| for (0..c) |j| {
        out[j * r + i] = a[i * c + j];
    };
}
pub fn transposeMatViewSIMD(comptime T: type, a: MatView(T), out: MatView(T)) void {
    assert(a.rows_num == out.cols_num and a.cols_num == out.rows_num);
    transposeSIMD(T, a.rows_num, a.cols_num, a.slice, out.slice);
}
pub fn transposeMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T)) Mat(C, R, T) {
    var out: Mat(C, R, T) = undefined;
    transposeSIMD(T, R, C, flatSIMD(R, C, T, &a), mutableFlatSIMD(C, R, T, &out));
    return out;
}
pub fn outerMatViewSIMD(comptime T: type, a: VecView(T), b: VecView(T), out: MatView(T)) void {
    assert(out.rows_num == a.slice.len and out.cols_num == b.slice.len);
    for (0..out.rows_num) |r| core.mulScalSIMD(T, b.slice, a.slice[r], out.slice[r * out.cols_num ..][0..out.cols_num]);
}
pub fn outerAddMatViewSIMD(comptime T: type, alpha: T, a: VecView(T), b: VecView(T), out: MatView(T)) void {
    assert(out.rows_num == a.slice.len and out.cols_num == b.slice.len);
    for (0..out.rows_num) |r| {
        const row = out.slice[r * out.cols_num ..][0..out.cols_num];
        core.axpySIMD(T, alpha * a.slice[r], b.slice, row, row);
    }
}
pub fn outerMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Vec(R, T), b: Vec(C, T)) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    for (0..R) |r| core.mulScalSIMD(T, &b.vec, a.vec[r], &out.mat[r]);
    return out;
}
pub fn outerAddMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, alpha: T, a: Vec(R, T), b: Vec(C, T), m: Mat(R, C, T)) Mat(R, C, T) {
    var out = m;
    for (0..R) |r| core.axpySIMD(T, alpha * a.vec[r], &b.vec, &out.mat[r], &out.mat[r]);
    return out;
}
pub fn maxAbsMatViewSIMD(comptime T: type, a: MatView(T)) T {
    return core.maxAbsSIMD(T, a.slice);
}
pub fn maxAbsMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T)) T {
    return core.maxAbsSIMD(T, flatSIMD(R, C, T, &a));
}
pub fn allFiniteMatViewSIMD(comptime T: type, a: MatView(T)) bool {
    return core.allFiniteSIMD(T, a.slice);
}
pub fn allFiniteMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T)) bool {
    return core.allFiniteSIMD(T, flatSIMD(R, C, T, &a));
}
pub fn frobeniusNormMatViewSIMD(comptime T: type, a: MatView(T)) T {
    return core.normL2SIMD(T, a.slice);
}
pub fn frobeniusNormMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T)) T {
    return core.normL2SIMD(T, flatSIMD(R, C, T, &a));
}
pub fn addMatViewSIMD(comptime T: type, a: MatView(T), b: MatView(T), out: MatView(T)) void {
    assert(a.rows_num == b.rows_num and a.cols_num == b.cols_num and out.rows_num == a.rows_num and out.cols_num == a.cols_num);
    core.addSIMD(T, a.slice, b.slice[0..a.slice.len], out.slice[0..a.slice.len]);
}
pub fn addMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T), b: Mat(R, C, T)) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    core.addSIMD(T, flatSIMD(R, C, T, &a), flatSIMD(R, C, T, &b), mutableFlatSIMD(R, C, T, &out));
    return out;
}
pub fn subMatViewSIMD(comptime T: type, a: MatView(T), b: MatView(T), out: MatView(T)) void {
    assert(a.rows_num == b.rows_num and a.cols_num == b.cols_num and out.rows_num == a.rows_num and out.cols_num == a.cols_num);
    core.subSIMD(T, a.slice, b.slice[0..a.slice.len], out.slice[0..a.slice.len]);
}
pub fn subMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T), b: Mat(R, C, T)) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    core.subSIMD(T, flatSIMD(R, C, T, &a), flatSIMD(R, C, T, &b), mutableFlatSIMD(R, C, T, &out));
    return out;
}
pub fn mulElemWiseMatViewSIMD(comptime T: type, a: MatView(T), b: MatView(T), out: MatView(T)) void {
    assert(a.rows_num == b.rows_num and a.cols_num == b.cols_num and out.rows_num == a.rows_num and out.cols_num == a.cols_num);
    core.mulSIMD(T, a.slice, b.slice[0..a.slice.len], out.slice[0..a.slice.len]);
}
pub fn mulElemWiseMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T), b: Mat(R, C, T)) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    core.mulSIMD(T, flatSIMD(R, C, T, &a), flatSIMD(R, C, T, &b), mutableFlatSIMD(R, C, T, &out));
    return out;
}
pub fn divElemWiseMatViewSIMD(comptime T: type, a: MatView(T), b: MatView(T), out: MatView(T)) void {
    assert(a.rows_num == b.rows_num and a.cols_num == b.cols_num and out.rows_num == a.rows_num and out.cols_num == a.cols_num);
    core.divSIMD(T, a.slice, b.slice[0..a.slice.len], out.slice[0..a.slice.len]);
}
pub fn divElemWiseMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T), b: Mat(R, C, T)) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    core.divSIMD(T, flatSIMD(R, C, T, &a), flatSIMD(R, C, T, &b), mutableFlatSIMD(R, C, T, &out));
    return out;
}
pub fn mulScalMatViewSIMD(comptime T: type, a: MatView(T), alpha: T, out: MatView(T)) void {
    assert(a.rows_num == out.rows_num and a.cols_num == out.cols_num);
    core.mulScalSIMD(T, a.slice, alpha, out.slice[0..a.slice.len]);
}
pub fn mulScalMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T), alpha: T) Mat(R, C, T) {
    var out: Mat(R, C, T) = undefined;
    core.mulScalSIMD(T, flatSIMD(R, C, T, &a), alpha, mutableFlatSIMD(R, C, T, &out));
    return out;
}
pub fn fillMatViewSIMD(comptime T: type, a: MatView(T), value: T) void {
    core.fillSIMD(T, a.slice, value);
}
pub fn traceMatViewSIMD(comptime T: type, a: MatView(T)) T {
    const W = 32 / @sizeOf(T);
    const n = @min(a.rows_num, a.cols_num);
    var acc: @Vector(W, T) = @splat(0);
    var i: usize = 0;
    while (n - i >= W) : (i += W) {
        var v: @Vector(W, T) = undefined;
        inline for (0..W) |j| v[j] = a.slice[(i + j) * (a.cols_num + 1)];
        acc += v;
    }
    var result: T = @reduce(.Add, acc);
    for (i..n) |j| result += a.slice[j * (a.cols_num + 1)];
    return result;
}
pub fn traceMatSIMD(comptime R: usize, comptime C: usize, comptime T: type, a: Mat(R, C, T)) T {
    const W = 32 / @sizeOf(T);
    const n = @min(R, C);
    var acc: @Vector(W, T) = @splat(0);
    var i: usize = 0;
    while (n - i >= W) : (i += W) {
        var v: @Vector(W, T) = undefined;
        inline for (0..W) |j| v[j] = a.mat[i + j][i + j];
        acc += v;
    }
    var result: T = @reduce(.Add, acc);
    for (i..n) |j| result += a.mat[j][j];
    return result;
}
