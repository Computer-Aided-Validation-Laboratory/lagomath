//! Validation shared by opt-in checked adapters. Standard kernels keep assertions.
const std = @import("std");
const MatView = @import("matviewtype.zig").MatView;
const VecView = @import("vecviewtype.zig").VecView;
const Error = @import("linsolve.zig").SolveError;
pub fn overlaps(a: anytype, b: anytype) bool {
    if (a.len == 0 or b.len == 0) return false;
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return x < y + @sizeOf(@TypeOf(b[0])) * b.len and y < x + @sizeOf(@TypeOf(a[0])) * a.len;
}
pub fn matrix(comptime T: type, a: MatView(T)) Error!void {
    const count = std.math.mul(usize, a.rows_num, a.cols_num) catch return error.ShapeMismatch;
    if (a.rows_num == 0 or a.cols_num == 0 or a.slice.len != count) return error.ShapeMismatch;
}
pub fn solve(comptime T: type, a: MatView(T), b: VecView(T), work: MatView(T), rhs: VecView(T), out: VecView(T)) Error!void {
    try matrix(T, a);
    try matrix(T, work);
    const n = a.rows_num;
    if (a.cols_num != n or work.rows_num != n or work.cols_num != n or b.slice.len != n or rhs.slice.len != n or out.slice.len != n) return error.ShapeMismatch;
    if (overlaps(a.slice, work.slice) or overlaps(a.slice, rhs.slice) or overlaps(a.slice, out.slice) or overlaps(b.slice, work.slice) or overlaps(b.slice, rhs.slice) or overlaps(b.slice, out.slice) or overlaps(work.slice, rhs.slice) or overlaps(work.slice, out.slice) or overlaps(rhs.slice, out.slice)) return error.InvalidAliasing;
}
pub fn factored(comptime T: type, a: MatView(T), b: VecView(T), rhs: VecView(T), out: VecView(T)) Error!void {
    try matrix(T, a);
    const n = a.rows_num;
    if (a.cols_num != n or b.slice.len != n or rhs.slice.len != n or out.slice.len != n) return error.ShapeMismatch;
    if (overlaps(a.slice, rhs.slice) or overlaps(a.slice, out.slice) or overlaps(b.slice, rhs.slice) or overlaps(b.slice, out.slice) or overlaps(rhs.slice, out.slice)) return error.InvalidAliasing;
}
