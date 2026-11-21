//! Explicit SIMD vector operations; fixed values retain inline storage.
const std = @import("std");
const core = @import("sliceops_simd.zig");
const Vec = @import("vectype.zig").Vec;
const VecView = @import("vecviewtype.zig").VecView;
pub fn dotVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T), b: Vec(N, T)) T {
    return core.dotSIMD(T, &a.vec, &b.vec);
}
pub fn dotVecViewSIMD(comptime T: type, a: VecView(T), b: VecView(T)) T {
    return core.dotSIMD(T, a.slice, b.slice);
}
pub fn normL1VecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) T {
    return core.normL1SIMD(T, &a.vec);
}
pub fn normL1VecViewSIMD(comptime T: type, a: VecView(T)) T {
    return core.normL1SIMD(T, a.slice);
}
pub fn normL2VecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) T {
    return core.normL2SIMD(T, &a.vec);
}
pub fn normL2VecViewSIMD(comptime T: type, a: VecView(T)) T {
    return core.normL2SIMD(T, a.slice);
}
pub fn normL2SquareVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) T {
    return core.normL2SquareSIMD(T, &a.vec);
}
pub fn normL2SquareVecViewSIMD(comptime T: type, a: VecView(T)) T {
    return core.normL2SquareSIMD(T, a.slice);
}
pub fn maxAbsVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) T {
    return core.maxAbsSIMD(T, &a.vec);
}
pub fn maxAbsVecViewSIMD(comptime T: type, a: VecView(T)) T {
    return core.maxAbsSIMD(T, a.slice);
}
pub fn allFiniteVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) bool {
    return core.allFiniteSIMD(T, &a.vec);
}
pub fn allFiniteVecViewSIMD(comptime T: type, a: VecView(T)) bool {
    return core.allFiniteSIMD(T, a.slice);
}
pub fn sumVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) T {
    return core.sumSIMD(T, &a.vec);
}
pub fn sumVecViewSIMD(comptime T: type, a: VecView(T)) T {
    return core.sumSIMD(T, a.slice);
}
pub fn meanVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) T {
    return core.meanSIMD(T, &a.vec);
}
pub fn meanVecViewSIMD(comptime T: type, a: VecView(T)) T {
    return core.meanSIMD(T, a.slice);
}
pub fn minVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) @import("sliceops.zig").ValIdx(T) {
    return core.minSIMD(T, &a.vec);
}
pub fn minVecViewSIMD(comptime T: type, a: VecView(T)) @import("sliceops.zig").ValIdx(T) {
    return core.minSIMD(T, a.slice);
}
pub fn maxVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T)) @import("sliceops.zig").ValIdx(T) {
    return core.maxSIMD(T, &a.vec);
}
pub fn maxVecViewSIMD(comptime T: type, a: VecView(T)) @import("sliceops.zig").ValIdx(T) {
    return core.maxSIMD(T, a.slice);
}
pub fn addVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T), b: Vec(N, T)) Vec(N, T) {
    var out: Vec(N, T) = undefined;
    core.addSIMD(T, &a.vec, &b.vec, &out.vec);
    return out;
}
pub fn addVecViewSIMD(comptime T: type, a: VecView(T), b: VecView(T), out: VecView(T)) void {
    core.addSIMD(T, a.slice, b.slice, out.slice);
}
pub fn subVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T), b: Vec(N, T)) Vec(N, T) {
    var out: Vec(N, T) = undefined;
    core.subSIMD(T, &a.vec, &b.vec, &out.vec);
    return out;
}
pub fn subVecViewSIMD(comptime T: type, a: VecView(T), b: VecView(T), out: VecView(T)) void {
    core.subSIMD(T, a.slice, b.slice, out.slice);
}
pub fn mulVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T), b: Vec(N, T)) Vec(N, T) {
    var out: Vec(N, T) = undefined;
    core.mulSIMD(T, &a.vec, &b.vec, &out.vec);
    return out;
}
pub fn mulVecViewSIMD(comptime T: type, a: VecView(T), b: VecView(T), out: VecView(T)) void {
    core.mulSIMD(T, a.slice, b.slice, out.slice);
}
pub fn divVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T), b: Vec(N, T)) Vec(N, T) {
    var out: Vec(N, T) = undefined;
    core.divSIMD(T, &a.vec, &b.vec, &out.vec);
    return out;
}
pub fn divVecViewSIMD(comptime T: type, a: VecView(T), b: VecView(T), out: VecView(T)) void {
    core.divSIMD(T, a.slice, b.slice, out.slice);
}
pub fn axpyVecSIMD(comptime N: usize, comptime T: type, alpha: T, a: Vec(N, T), b: Vec(N, T)) Vec(N, T) {
    var out: Vec(N, T) = undefined;
    core.axpySIMD(T, alpha, &a.vec, &b.vec, &out.vec);
    return out;
}
pub fn axpyVecViewSIMD(comptime T: type, alpha: T, a: VecView(T), b: VecView(T)) void {
    core.axpySIMD(T, alpha, a.slice, b.slice, b.slice);
}
pub fn mulScalVecSIMD(comptime N: usize, comptime T: type, a: Vec(N, T), alpha: T) Vec(N, T) {
    var out: Vec(N, T) = undefined;
    core.mulScalSIMD(T, &a.vec, alpha, &out.vec);
    return out;
}
pub fn mulScalVecViewSIMD(comptime T: type, a: VecView(T), alpha: T, out: VecView(T)) void {
    core.mulScalSIMD(T, a.slice, alpha, out.slice);
}
// SIMD across independent short vectors, for callers that vectorise outer work.
pub fn dotBatchSIMD(comptime N: usize, comptime T: type, a: []const T, b: []const T, out: []T) void {
    std.debug.assert(a.len == out.len * N and b.len == a.len);
    const W = 32 / @sizeOf(T);
    var i: usize = 0;
    while (out.len - i >= W) : (i += W) {
        var acc: @Vector(W, T) = @splat(0);
        inline for (0..N) |k| {
            var av: @Vector(W, T) = undefined;
            var bv: @Vector(W, T) = undefined;
            inline for (0..W) |j| {
                av[j] = a[(i + j) * N + k];
                bv[j] = b[(i + j) * N + k];
            }
            acc += av * bv;
        }
        out[i..][0..W].* = acc;
    }
    for (i..out.len) |j| {
        var sum: T = 0;
        inline for (0..N) |k| sum += a[j * N + k] * b[j * N + k];
        out[j] = sum;
    }
}
pub const normVecSIMD = normL2VecSIMD;
pub const normVecViewSIMD = normL2VecViewSIMD;
pub const squaredNormVecSIMD = normL2SquareVecSIMD;
pub const squaredNormVecViewSIMD = normL2SquareVecViewSIMD;
