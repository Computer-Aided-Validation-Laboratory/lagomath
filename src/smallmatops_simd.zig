//! Opt-in cofactor SIMD across independent minors of 2x2/3x3/4x4 matrices.
const std = @import("std");
const Mat = @import("mattype.zig").Mat;
fn cofactorsSIMD(comptime N: usize, comptime T: type, a: Mat(N, N, T), start: usize, stride: usize) @Vector(32 / @sizeOf(T), T) {
    comptime std.debug.assert(N >= 2 and N <= 4);
    const W = 32 / @sizeOf(T);
    var minor: [N - 1][N - 1]@Vector(W, T) = @splat(@splat(@as(@Vector(W, T), @splat(0))));
    var signs: @Vector(W, T) = @splat(1);
    inline for (0..W) |lane| {
        const p = start + lane * stride;
        if (p < N * N) {
            const remove_r = p % N;
            const remove_c = p / N;
            signs[lane] = if ((remove_r + remove_c) % 2 == 0) 1 else -1;
            inline for (0..N - 1) |r| inline for (0..N - 1) |c| {
                minor[r][c][lane] = a.mat[r + @intFromBool(r >= remove_r)][c + @intFromBool(c >= remove_c)];
            };
        }
    }
    if (N == 2) return signs * minor[0][0];
    if (N == 3) return signs * (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]);
    return signs * (minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1]) - minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0]) + minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]));
}
pub fn detMatSIMD(comptime N: usize, comptime T: type, a: Mat(N, N, T)) T {
    const W = 32 / @sizeOf(T);
    var first: @Vector(W, T) = @splat(0);
    inline for (0..N) |i| first[i] = a.mat[0][i];
    const terms = cofactorsSIMD(N, T, a, 0, N) * first;
    var result: T = terms[0];
    inline for (1..N) |i| result += terms[i];
    return result;
}
fn inverseSIMD(comptime N: usize, comptime T: type, a: Mat(N, N, T), det: T) Mat(N, N, T) {
    const W = 32 / @sizeOf(T);
    const scale: @Vector(W, T) = @splat(1 / det);
    var out: Mat(N, N, T) = undefined;
    var start: usize = 0;
    while (start < N * N) : (start += W) {
        const v = cofactorsSIMD(N, T, a, start, 1) * scale;
        inline for (0..W) |lane| {
            if (start + lane < N * N) out.mat[(start + lane) / N][(start + lane) % N] = v[lane];
        }
    }
    return out;
}
pub fn invMatSIMD(comptime N: usize, comptime T: type, a: Mat(N, N, T)) Mat(N, N, T) {
    const det = detMatSIMD(N, T, a);
    std.debug.assert(std.math.isFinite(det) and det != 0);
    return inverseSIMD(N, T, a, det);
}
pub fn invMatCheckedSIMD(comptime N: usize, comptime T: type, a: Mat(N, N, T), min_abs_det: T) error{SingularMatrix}!Mat(N, N, T) {
    const det = detMatSIMD(N, T, a);
    if (!std.math.isFinite(det) or @abs(det) <= min_abs_det) return error.SingularMatrix;
    return inverseSIMD(N, T, a, det);
}
