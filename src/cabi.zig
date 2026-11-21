const std = @import("std");
const lm = @import("root.zig");
export fn lm_abi_version() u32 {
    return 1;
}
fn status(err: anyerror) i32 {
    return switch (err) {
        error.SingularMatrix => 3,
        error.NotPositiveDefinite => 4,
        error.NonFiniteInput => 5,
        error.InvalidAliasing => 6,
        else => 2,
    };
}
fn overlap(comptime T: type, a: []const T, b: []const T) bool {
    if (a.len == 0 or b.len == 0) return false;
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return x < y + b.len * @sizeOf(T) and y < x + a.len * @sizeOf(T);
}
fn size(n: i64) ?usize {
    if (n < 0 or n > 1000000000) return null;
    return @intCast(n);
}
fn dot(comptime T: type, ap: ?[*]const T, bp: ?[*]const T, n: i64, rp: ?*T) i32 {
    const len = size(n) orelse return 1;
    const r = rp orelse return 1;
    if (len == 0) {
        r.* = 0;
        return 0;
    }
    const a = (ap orelse return 1)[0..len];
    const b = (bp orelse return 1)[0..len];
    if (!sliceFinite(T, a) or !sliceFinite(T, b)) return 5;
    r.* = lm.sliceops.dot(T, a, b);
    return 0;
}
fn matmul(comptime T: type, ap: ?[*]const T, bp: ?[*]const T, op: ?[*]T, m: i64, n: i64, p: i64) i32 {
    const rows = size(m) orelse return 1;
    const inner = size(n) orelse return 1;
    const cols = size(p) orelse return 1;
    if (rows == 0 or inner == 0 or cols == 0) return 1;
    const a = (ap orelse return 1)[0 .. rows * inner];
    const b = (bp orelse return 1)[0 .. inner * cols];
    const out = (op orelse return 1)[0 .. rows * cols];
    if (overlap(T, a, out) or overlap(T, b, out)) return 6;
    if (!sliceFinite(T, a) or !sliceFinite(T, b)) return 5;
    lm.matops.mulMatMatView(T, lm.MatView(T).init(@constCast(a), rows, inner), lm.MatView(T).init(@constCast(b), inner, cols), lm.MatView(T).init(out, rows, cols));
    return 0;
}
fn axpy(comptime T: type, alpha: T, ap: ?[*]const T, op: ?[*]T, n: i64) i32 {
    const len = size(n) orelse return 1;
    if (len == 0) return 0;
    const a = (ap orelse return 1)[0..len];
    const out = (op orelse return 1)[0..len];
    if (overlap(T, a, out) and a.ptr != out.ptr) return 6;
    if (!std.math.isFinite(alpha) or !sliceFinite(T, a) or !sliceFinite(T, out)) return 5;
    lm.vecops.axpyVecView(T, alpha, lm.VecView(T).init(@constCast(a)), lm.VecView(T).init(out));
    return 0;
}
fn solve(comptime T: type, comptime kind: u8, ap: ?[*]const T, bp: ?[*]const T, op: ?[*]T, n: i64, sp: ?[*]T, count: i64) i32 {
    const len = size(n) orelse return 1;
    const sc = size(count) orelse return 1;
    if (len == 0) return 1;
    if (sc < len * len + len) return 2;
    const a = (ap orelse return 1)[0 .. len * len];
    const b = (bp orelse return 1)[0..len];
    const out = (op orelse return 1)[0..len];
    const scratch = (sp orelse return 1)[0 .. len * len + len];
    if (overlap(T, a, out) or overlap(T, b, out) or overlap(T, a, scratch) or overlap(T, b, scratch) or overlap(T, out, scratch)) return 6;
    const matrix = lm.MatView(T).init(@constCast(a), len, len);
    const rhs = lm.VecView(T).init(@constCast(b));
    const dest = lm.VecView(T).init(out);
    const wm = lm.MatView(T).init(scratch[0 .. len * len], len, len);
    const wv = lm.VecView(T).init(scratch[len * len ..]);
    if (kind == 0) {
        lm.linsolve.solveGaussMatViewChecked(T, matrix, rhs, wm, wv, dest) catch |err| return status(err);
    } else {
        lm.linsolve.solveCholeskyMatViewChecked(T, matrix, rhs, wm, wv, dest) catch |err| return status(err);
    }
    return 0;
}
export fn lm_dot_f32(ap: ?[*]const f32, bp: ?[*]const f32, n: i64, rp: ?*f32) i32 {
    return dot(f32, ap, bp, n, rp);
}
export fn lm_matmul_f32(ap: ?[*]const f32, bp: ?[*]const f32, op: ?[*]f32, m: i64, n: i64, p: i64) i32 {
    return matmul(f32, ap, bp, op, m, n, p);
}
export fn lm_matvec_f32(ap: ?[*]const f32, m: i64, n: i64, bp: ?[*]const f32, op: ?[*]f32) i32 {
    return matmul(f32, ap, bp, op, m, n, 1);
}
export fn lm_axpy_f32(alpha: f32, ap: ?[*]const f32, op: ?[*]f32, n: i64) i32 {
    return axpy(f32, alpha, ap, op, n);
}
export fn lm_solve_gauss_f32(ap: ?[*]const f32, bp: ?[*]const f32, op: ?[*]f32, n: i64, sp: ?[*]f32, count: i64) i32 {
    return solve(f32, 0, ap, bp, op, n, sp, count);
}
export fn lm_solve_cholesky_f32(ap: ?[*]const f32, bp: ?[*]const f32, op: ?[*]f32, n: i64, sp: ?[*]f32, count: i64) i32 {
    return solve(f32, 1, ap, bp, op, n, sp, count);
}
export fn lm_dot_f64(ap: ?[*]const f64, bp: ?[*]const f64, n: i64, rp: ?*f64) i32 {
    return dot(f64, ap, bp, n, rp);
}
export fn lm_matmul_f64(ap: ?[*]const f64, bp: ?[*]const f64, op: ?[*]f64, m: i64, n: i64, p: i64) i32 {
    return matmul(f64, ap, bp, op, m, n, p);
}
export fn lm_matvec_f64(ap: ?[*]const f64, m: i64, n: i64, bp: ?[*]const f64, op: ?[*]f64) i32 {
    return matmul(f64, ap, bp, op, m, n, 1);
}
export fn lm_axpy_f64(alpha: f64, ap: ?[*]const f64, op: ?[*]f64, n: i64) i32 {
    return axpy(f64, alpha, ap, op, n);
}
export fn lm_solve_gauss_f64(ap: ?[*]const f64, bp: ?[*]const f64, op: ?[*]f64, n: i64, sp: ?[*]f64, count: i64) i32 {
    return solve(f64, 0, ap, bp, op, n, sp, count);
}
export fn lm_solve_cholesky_f64(ap: ?[*]const f64, bp: ?[*]const f64, op: ?[*]f64, n: i64, sp: ?[*]f64, count: i64) i32 {
    return solve(f64, 1, ap, bp, op, n, sp, count);
}

const Mat22ABI = extern struct { data: [4]f64 };
export fn lm_mat22_transpose_f64(ap: ?*const Mat22ABI, op: ?*Mat22ABI) i32 {
    const a = ap orelse return 1;
    const out = op orelse return 1;
    if (!sliceFinite(f64, &a.data)) return 5;
    const copy = a.*;
    out.* = .{ .data = .{ copy.data[0], copy.data[2], copy.data[1], copy.data[3] } };
    return 0;
}
export fn lm_mat22_size_f64() u32 {
    return @sizeOf(Mat22ABI);
}
export fn lm_mat22_align_f64() u32 {
    return @alignOf(Mat22ABI);
}

fn sliceFinite(comptime T: type, values: []const T) bool {
    for (values) |v| {
        if (!std.math.isFinite(v)) return false;
    }
    return true;
}
