// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const print = std.debug.print;
const F = f64;

const vecstack = @import("vecstack.zig");
const VecStack = vecstack.VecStack;
const Vec2T = vecstack.Vec2T;
const Vec3T = vecstack.Vec3T;
const Vec2f = vecstack.Vec2f;
const Vec3f = vecstack.Vec3f;

const TestType = F;
pub const Mat22f = Mat22T(F);
pub const Mat33f = Mat33T(F);
pub const Mat44f = Mat44T(F);
pub const MatrixInversionError = error{SingularMatrix};

pub fn MatStack(
    comptime rows_n: comptime_int,
    comptime cols_n: comptime_int,
    comptime T: type,
) type {
    return struct {
        mat: [rows_n][cols_n]T,

        pub const elem_n: usize = rows_n * cols_n;

        const Self: type = @This();

        pub fn initFill(fill_val: T) Self {
            return .{ .mat = [_][cols_n]T{[_]T{fill_val} ** cols_n} ** rows_n };
        }

        pub fn initZeros() Self {
            return initFill(0);
        }

        pub fn initOnes() Self {
            return initFill(1);
        }

        pub fn initDiag(diag_val: T) Self {
            var ident: Self = initZeros();

            var diag_n: usize = cols_n;
            if (cols_n > rows_n) {
                diag_n = rows_n;
            }

            for (0..diag_n) |ii| {
                ident.mat[ii][ii] = diag_val;
            }

            return ident;
        }

        pub fn initIdentity() Self {
            return initDiag(1);
        }

        pub fn initSlice(slice_in: []const T) Self {
            std.debug.assert(slice_in.len >= elem_n);
            var mat_out: Self = undefined;
            inline for (0..rows_n) |rr| {
                const start_idx: usize = rr * cols_n;
                @memcpy(&mat_out.mat[rr], slice_in[start_idx .. start_idx + cols_n]);
            }
            return mat_out;
        }

        /// Builds a row-major matrix without requiring callers to flatten rows.
        pub fn initRows(rows: [rows_n][cols_n]T) Self {
            return .{ .mat = rows };
        }

        pub fn get(self: *const Self, row: usize, col: usize) T {
            std.debug.assert(row < rows_n);
            std.debug.assert(col < cols_n);
            return self.mat[row][col];
        }

        pub fn set(self: *Self, row: usize, col: usize, val: T) void {
            std.debug.assert(row < rows_n);
            std.debug.assert(col < cols_n);
            self.mat[row][col] = val;
        }

        pub fn getRowVec(self: *const Self, row: usize) VecStack(cols_n, T) {
            std.debug.assert(row < rows_n);
            return VecStack(cols_n, T).initSlice(&self.mat[row]);
        }

        pub fn getColVec(self: *const Self, col: usize) VecStack(rows_n, T) {
            std.debug.assert(col < cols_n);
            var col_vec: [rows_n]T = undefined;
            inline for (0..rows_n) |rr| {
                col_vec[rr] = self.mat[rr][col];
            }
            const vec = VecStack(rows_n, T).initSlice(&col_vec);
            return vec;
        }

        pub fn getSubMat(
            self: *const Self,
            row_start: usize,
            col_start: usize,
            comptime rows: usize,
            comptime cols: usize,
        ) MatStack(rows, cols, T) {
            var sub_mat = MatStack(rows, cols, T).initZeros();

            const row_end: usize = row_start + rows;
            const col_end: usize = col_start + cols;
            for (row_start..row_end) |rr| {
                for (col_start..col_end) |cc| {
                    sub_mat.mat[rr - row_start][cc - col_start] = self.mat[rr][cc];
                }
            }

            return sub_mat;
        }

        pub fn insertRowVec(
            self: *Self,
            row: usize,
            col_start: usize,
            comptime vec_len: usize,
            vec: VecStack(vec_len, T),
        ) void {
            inline for (0..vec_len) |cc| {
                self.mat[row][cc + col_start] = vec.get(cc);
            }
        }

        pub fn insertColVec(
            self: *Self,
            col: usize,
            row_start: usize,
            comptime vec_len: usize,
            vec: VecStack(vec_len, T),
        ) void {
            inline for (0..vec_len) |rr| {
                self.mat[rr + row_start][col] = vec.get(rr);
            }
        }

        pub fn insertSubMat(
            self: *Self,
            row_start: usize,
            col_start: usize,
            comptime mat_rows: usize,
            comptime mat_cols: usize,
            sub_mat: MatStack(mat_rows, mat_cols, T),
        ) void {
            inline for (0..mat_rows) |rr| {
                inline for (0..mat_cols) |cc| {
                    self.mat[rr + row_start][cc + col_start] = sub_mat.mat[rr][cc];
                }
            }
        }

        pub fn transpose(self: *const Self) MatStack(cols_n, rows_n, T) {
            var mat_out: MatStack(cols_n, rows_n, T) = undefined;

            inline for (0..rows_n) |rr| {
                inline for (0..cols_n) |cc| {
                    mat_out.mat[cc][rr] = self.mat[rr][cc];
                }
            }

            return mat_out;
        }

        pub fn trace(self: *const Self) T {
            var trace_out: T = 0;

            if (rows_n <= cols_n) {
                for (0..rows_n) |ii| {
                    trace_out += self.mat[ii][ii];
                }
            } else {
                for (0..cols_n) |ii| {
                    trace_out += self.mat[ii][ii];
                }
            }

            return trace_out;
        }

        pub fn add(self: *const Self, to_add: Self) Self {
            var mat_out: Self = undefined;

            inline for (0..rows_n) |rr| {
                inline for (0..cols_n) |cc| {
                    mat_out.mat[rr][cc] = self.mat[rr][cc] + to_add.mat[rr][cc];
                }
            }

            return mat_out;
        }

        pub fn sub(self: *const Self, to_sub: Self) Self {
            var mat_out: Self = undefined;

            inline for (0..rows_n) |rr| {
                inline for (0..cols_n) |cc| {
                    mat_out.mat[rr][cc] = self.mat[rr][cc] - to_sub.mat[rr][cc];
                }
            }

            return mat_out;
        }

        pub fn mulScal(self: *const Self, scal: T) Self {
            var mat_out: Self = undefined;

            inline for (0..rows_n) |rr| {
                inline for (0..cols_n) |cc| {
                    mat_out.mat[rr][cc] = scal * self.mat[rr][cc];
                }
            }

            return mat_out;
        }

        pub fn mulVec(self: *const Self, vec: VecStack(cols_n, T)) VecStack(rows_n, T) {
            var vec_out: VecStack(rows_n, T) = undefined;
            var sum: T = 0;

            inline for (0..rows_n) |rr| {
                sum = 0;
                inline for (0..cols_n) |cc| {
                    sum += self.mat[rr][cc] * vec.get(cc);
                }
                vec_out.set(rr, sum);
            }

            return vec_out;
        }

        pub fn mulMat(self: *const Self, to_mult: Self) Self {
            return self.mulMatRect(cols_n, to_mult);
        }

        pub fn mulMatRect(
            self: *const Self,
            comptime other_cols: usize,
            to_mult: MatStack(cols_n, other_cols, T),
        ) MatStack(rows_n, other_cols, T) {
            var mat_out = MatStack(rows_n, other_cols, T).initZeros();
            var sum: T = 0;

            inline for (0..rows_n) |rr| {
                inline for (0..other_cols) |cc| {
                    sum = 0;
                    inline for (0..cols_n) |mm| {
                        sum += self.mat[rr][mm] * to_mult.mat[mm][cc];
                    }
                    mat_out.mat[rr][cc] = sum;
                }
            }
            return mat_out;
        }

        pub fn matPrint(self: *const Self) void {
            for (0..rows_n) |rr| {
                print("[", .{});
                for (0..cols_n) |cc| {
                    print("{e:.3},", .{self.mat[rr][cc]});
                }
                print("]\n", .{});
            }
            print("\n", .{});
        }
    };
}

pub fn Mat22T(comptime T: type) type {
    return MatStack(2, 2, T);
}

pub fn Mat33T(comptime T: type) type {
    return MatStack(3, 3, T);
}

pub fn Mat44T(comptime T: type) type {
    return MatStack(4, 4, T);
}

pub const Mat22Ops = struct {
    pub fn adj(comptime T: type, mat22: Mat22T(T)) Mat22T(T) {
        var adjoint: Mat22T(T) = undefined;
        adjoint.set(0, 0, mat22.get(1, 1));
        adjoint.set(1, 1, mat22.get(0, 0));
        adjoint.set(0, 1, -1 * mat22.get(0, 1));
        adjoint.set(1, 0, -1 * mat22.get(1, 0));
        return adjoint;
    }

    pub fn det(comptime T: type, mat22: Mat22T(T)) T {
        return (mat22.get(0, 0) * mat22.get(1, 1)) - (mat22.get(0, 1) * mat22.get(1, 0));
    }

    pub fn inv(comptime T: type, mat22: Mat22T(T)) Mat22T(T) {
        var inv_mat: Mat22T(T) = adj(T, mat22);
        const mat_det: T = det(T, mat22);
        std.debug.assert(std.math.isFinite(mat_det) and mat_det != 0);
        inv_mat = inv_mat.mulScal(1 / mat_det);
        return inv_mat;
    }

    pub fn invChecked(
        comptime T: type,
        mat22: Mat22T(T),
        min_abs_det: T,
    ) MatrixInversionError!Mat22T(T) {
        const mat_det = det(T, mat22);
        if (!std.math.isFinite(mat_det) or @abs(mat_det) <= min_abs_det) {
            return error.SingularMatrix;
        }
        return inv(T, mat22);
    }
};

pub const Mat33Ops = struct {
    pub fn det(comptime T: type, mat33: Mat33T(T)) T {
        var sub_dets: [3]T = undefined;

        sub_dets[0] = mat33.get(0, 0) * ((mat33.get(1, 1) * mat33.get(2, 2)) - //
            (mat33.get(1, 2) * mat33.get(2, 1)));
        sub_dets[1] = mat33.get(0, 1) * ((mat33.get(1, 0) * mat33.get(2, 2)) - //
            (mat33.get(1, 2) * mat33.get(2, 0)));
        sub_dets[2] = mat33.get(0, 2) * ((mat33.get(1, 0) * mat33.get(2, 1)) - //
            (mat33.get(1, 1) * mat33.get(2, 0)));

        return sub_dets[0] - sub_dets[1] + sub_dets[2];
    }

    pub fn inv(
        comptime T: type,
        mat33: Mat33T(T),
    ) Mat33T(T) {
        var inv33: Mat33T(T) = undefined;

        const mat_det = det(T, mat33);
        std.debug.assert(std.math.isFinite(mat_det) and mat_det != 0);
        const detm = 1 / mat_det;

        // Calculate the cofactors and transpose in one step
        inv33.mat[0][0] = detm * (mat33.get(1, 1) * mat33.get(2, 2) - //
            mat33.get(1, 2) * mat33.get(2, 1));
        inv33.mat[0][1] = -detm * (mat33.get(0, 1) * mat33.get(2, 2) - //
            mat33.get(0, 2) * mat33.get(2, 1));
        inv33.mat[0][2] = detm * (mat33.get(0, 1) * mat33.get(1, 2) - //
            mat33.get(0, 2) * mat33.get(1, 1));
        inv33.mat[1][0] = -detm * (mat33.get(1, 0) * mat33.get(2, 2) - //
            mat33.get(1, 2) * mat33.get(2, 0));
        inv33.mat[1][1] = detm * (mat33.get(0, 0) * mat33.get(2, 2) - //
            mat33.get(0, 2) * mat33.get(2, 0));
        inv33.mat[1][2] = -detm * (mat33.get(0, 0) * mat33.get(1, 2) - //
            mat33.get(0, 2) * mat33.get(1, 0));
        inv33.mat[2][0] = detm * (mat33.get(1, 0) * mat33.get(2, 1) - //
            mat33.get(1, 1) * mat33.get(2, 0));
        inv33.mat[2][1] = -detm * (mat33.get(0, 0) * mat33.get(2, 1) - //
            mat33.get(0, 1) * mat33.get(2, 0));
        inv33.mat[2][2] = detm * (mat33.get(0, 0) * mat33.get(1, 1) - //
            mat33.get(0, 1) * mat33.get(1, 0));

        return inv33;
    }

    pub fn invChecked(
        comptime T: type,
        mat33: Mat33T(T),
        min_abs_det: T,
    ) MatrixInversionError!Mat33T(T) {
        const mat_det = det(T, mat33);
        if (!std.math.isFinite(mat_det) or @abs(mat_det) <= min_abs_det) {
            return error.SingularMatrix;
        }
        return inv(T, mat33);
    }
};

pub const Mat44Ops = struct {
    pub fn det(comptime T: type, mat: Mat44T(T)) T {
        const mat_a = mat.getSubMat(0, 0, 2, 2);
        const mat_b = mat.getSubMat(0, 2, 2, 2);
        const mat_c = mat.getSubMat(2, 0, 2, 2);
        const mat_d = mat.getSubMat(2, 2, 2, 2);

        const det_a = Mat22Ops.det(T, mat_a);
        const det_b = Mat22Ops.det(T, mat_b);
        const det_c = Mat22Ops.det(T, mat_c);
        const det_d = Mat22Ops.det(T, mat_d);

        const adj_a = Mat22Ops.adj(T, mat_a);
        const adj_d = Mat22Ops.adj(T, mat_d);

        const adj_ab = adj_a.mulMat(mat_b);
        const adj_dc = adj_d.mulMat(mat_c);
        const adj_ab_dc = adj_ab.mulMat(adj_dc);

        return det_a * det_d + det_b * det_c - adj_ab_dc.trace();
    }

    pub fn insertMat22(
        comptime T: type,
        mat44: *Mat44T(T),
        mat22: Mat22T(T),
        row_start: usize,
        col_start: usize,
    ) void {
        mat44.set(0 + row_start, 0 + col_start, mat22.get(0, 0));
        mat44.set(0 + row_start, 1 + col_start, mat22.get(0, 1));
        mat44.set(1 + row_start, 0 + col_start, mat22.get(1, 0));
        mat44.set(1 + row_start, 1 + col_start, mat22.get(1, 1));
    }

    pub fn inv(comptime T: type, mat: Mat44T(T)) Mat44T(T) {
        const mat_a = mat.getSubMat(0, 0, 2, 2);
        const mat_b = mat.getSubMat(0, 2, 2, 2);
        const mat_c = mat.getSubMat(2, 0, 2, 2);
        const mat_d = mat.getSubMat(2, 2, 2, 2);

        const det_a = Mat22Ops.det(T, mat_a);
        const det_b = Mat22Ops.det(T, mat_b);
        const det_c = Mat22Ops.det(T, mat_c);
        const det_d = Mat22Ops.det(T, mat_d);

        const adj_a = Mat22Ops.adj(T, mat_a);
        const adj_d = Mat22Ops.adj(T, mat_d);

        const adj_ab = adj_a.mulMat(mat_b);
        const adj_dc = adj_d.mulMat(mat_c);
        const adj_ab_dc = adj_ab.mulMat(adj_dc);

        const det_m: T = det_a * det_d + det_b * det_c - adj_ab_dc.trace();
        std.debug.assert(std.math.isFinite(det_m) and det_m != 0);

        var inv_a = mat_a.mulScal(det_d);
        const b_adj_dc = mat_b.mulMat(adj_dc);
        inv_a = inv_a.sub(b_adj_dc);
        inv_a = Mat22Ops.adj(T, inv_a);

        var inv_b = mat_c.mulScal(det_b);
        const adj_ab_adj = Mat22Ops.adj(T, adj_ab);
        const d_adj_ab_adj = mat_d.mulMat(adj_ab_adj);
        inv_b = inv_b.sub(d_adj_ab_adj);
        inv_b = Mat22Ops.adj(T, inv_b);

        var inv_c = mat_b.mulScal(det_c);
        const adj_dc_adj = Mat22Ops.adj(T, adj_dc);
        const a_adj_dc_adj = mat_a.mulMat(adj_dc_adj);
        inv_c = inv_c.sub(a_adj_dc_adj);
        inv_c = Mat22Ops.adj(T, inv_c);

        var inv_d = mat_d.mulScal(det_a);
        const c_adj_ab = mat_c.mulMat(adj_ab);
        inv_d = inv_d.sub(c_adj_ab);
        inv_d = Mat22Ops.adj(T, inv_d);

        var mat_inv: Mat44T(T) = Mat44T(T).initIdentity();

        insertMat22(T, &mat_inv, inv_a, 0, 0);
        insertMat22(T, &mat_inv, inv_b, 0, 2);
        insertMat22(T, &mat_inv, inv_c, 2, 0);
        insertMat22(T, &mat_inv, inv_d, 2, 2);

        mat_inv = mat_inv.mulScal(1 / det_m);
        return mat_inv;
    }

    pub fn invChecked(
        comptime T: type,
        mat: Mat44T(T),
        min_abs_det: T,
    ) MatrixInversionError!Mat44T(T) {
        const mat_det = det(T, mat);
        if (!std.math.isFinite(mat_det) or @abs(mat_det) <= min_abs_det) {
            return error.SingularMatrix;
        }
        return inv(T, mat);
    }

    pub fn mulVec3(comptime T: type, mat: Mat44T(T), vec: Vec3T(T)) Vec3T(T) {
        var vec_out: Vec3T(T) = undefined;
        var sum: T = 0;

        for (0..3) |ii| {
            sum = 0;
            for (0..3) |jj| {
                sum += mat.get(ii, jj) * vec.get(jj);
            }
            sum += mat.get(ii, 3);
            vec_out.set(ii, sum);
        }

        return vec_out;
    }
};

const expectEqual = std.testing.expectEqual;

test "Mat22f.getRowVec" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const vec_exp = Vec2f.initSlice(&.{ 3, 4 });

    try expectEqual(vec_exp, mat0.getRowVec(1));
}

test "Mat22f.getColVec" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const vec_exp = Vec2f.initSlice(&.{ 1, 3 });

    try expectEqual(vec_exp, mat0.getColVec(0));
}

test "Mat22f.add" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const mat1 = Mat22f.initRows(.{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const mat_exp = Mat22f.initRows(.{
        .{ 6, 8 },
        .{ 10, 12 },
    });

    try expectEqual(mat0.add(mat1), mat_exp);
}

test "Mat22f.sub" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const mat1 = Mat22f.initRows(.{
        .{ 5, 6 },
        .{ 7, 8 },
    });
    const mat_exp = Mat22f.initRows(.{
        .{ -4, -4 },
        .{ -4, -4 },
    });

    try expectEqual(mat_exp, mat0.sub(mat1));
}

test "Mat22f.trace" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const trace_exp: TestType = 5;

    try expectEqual(trace_exp, mat0.trace());
}

test "Mat22f.transpose" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const mat_exp = Mat22f.initRows(.{
        .{ 1, 3 },
        .{ 2, 4 },
    });

    try expectEqual(mat_exp, mat0.transpose());
}

test "MatStack rectangular transpose and row construction" {
    const Mat23 = MatStack(2, 3, TestType);
    const Mat32 = MatStack(3, 2, TestType);
    const mat = Mat23.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const expected = Mat32.initRows(.{
        .{ 1, 4 },
        .{ 2, 5 },
        .{ 3, 6 },
    });

    try expectEqual(expected, mat.transpose());
}

test "Mat22f.mulScal" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const scal: TestType = 2;
    const mat_exp = Mat22f.initRows(.{
        .{ 2, 4 },
        .{ 6, 8 },
    });

    try expectEqual(mat_exp, mat0.mulScal(scal));
}

test "Mat22f.mulVec" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const vec0 = Vec2f.initSlice(&.{ 1, 2 });
    const vec_exp = Vec2f.initSlice(&.{ 5, 11 });

    try expectEqual(vec_exp, mat0.mulVec(vec0));
}

test "Mat22f.mulMat" {
    const mat0 = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const mat1 = Mat22f.initRows(.{
        .{ 4, 3 },
        .{ 2, 1 },
    });
    const mat_exp = Mat22f.initRows(.{
        .{ 8, 5 },
        .{ 20, 13 },
    });

    try expectEqual(mat_exp, mat0.mulMat(mat1));
}

test "Mat22f.mulMat.identity_and_negative" {
    const mat_ident = Mat22f.initIdentity();

    const mat0 = Mat22f.initRows(.{
        .{ 2, -1 },
        .{ 3, 4 },
    });
    try expectEqual(mat0, mat0.mulMat(mat_ident));

    const mat1 = Mat22f.initRows(.{
        .{ -1, 5 },
        .{ 2, -3 },
    });

    const mat_exp = Mat22f.initRows(.{
        .{ -4, 13 },
        .{ 5, 3 },
    });
    try expectEqual(mat_exp, mat0.mulMat(mat1));
}

test "Mat22Ops.det" {
    const mat_ident = Mat22f.initIdentity();
    try expectEqual(1, Mat22Ops.det(TestType, mat_ident));

    const mat_pos = Mat22f.initRows(.{
        .{ 3, 2 },
        .{ 1, 4 },
    });
    try expectEqual(10, Mat22Ops.det(TestType, mat_pos));

    const mat_zero = Mat22f.initRows(.{
        .{ 2, 4 },
        .{ 1, 2 },
    });
    try expectEqual(0, Mat22Ops.det(TestType, mat_zero));

    const mat_neg = Mat22f.initRows(.{
        .{ 1, 5 },
        .{ 3, 2 },
    });
    try expectEqual(-13, Mat22Ops.det(TestType, mat_neg));
}

test "Mat22Ops.inv" {
    // Inversion with positive determinant: det = 2
    const mat_pos = Mat22f.initRows(.{
        .{ 4, 2 },
        .{ 3, 2 },
    });
    const mat_pos_exp = Mat22f.initRows(.{
        .{ 1.0, -1.0 },
        .{ -1.5, 2.0 },
    });
    try expectEqual(mat_pos_exp, Mat22Ops.inv(TestType, mat_pos));

    // Inversion with negative determinant: det = -2
    const mat_neg = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    const mat_neg_exp = Mat22f.initRows(.{
        .{ -2.0, 1.0 },
        .{ 1.5, -0.5 },
    });
    try expectEqual(mat_neg_exp, Mat22Ops.inv(TestType, mat_neg));
}

test "Mat22Ops.invChecked rejects singular and near-singular matrices" {
    const valid = Mat22f.initRows(.{
        .{ 4, 2 },
        .{ 3, 2 },
    });
    try expectEqual(
        Mat22Ops.inv(TestType, valid),
        try Mat22Ops.invChecked(TestType, valid, 1.0e-8),
    );

    const singular = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 2, 4 },
    });
    try std.testing.expectError(
        error.SingularMatrix,
        Mat22Ops.invChecked(TestType, singular, 0),
    );

    const near_singular = Mat22f.initRows(.{
        .{ 1, 0 },
        .{ 0, 1.0e-10 },
    });
    try std.testing.expectError(
        error.SingularMatrix,
        Mat22Ops.invChecked(TestType, near_singular, 1.0e-8),
    );
}

test "Mat33f.add" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });
    const mat1 = mat0;

    const mat_exp = Mat33f.initRows(.{
        .{ 2, 4, 6 },
        .{ 8, 10, 12 },
        .{ 14, 16, 18 },
    });

    try expectEqual(mat_exp, mat0.add(mat1));
}

test "Mat33f.sub" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });
    const mat1 = mat0;

    const mat_exp = Mat33f.initZeros();

    try expectEqual(mat_exp, mat0.sub(mat1));
}

test "Mat33f.getRowVec" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    const vec_exp = Vec3f.initSlice(&.{ 4, 5, 6 });

    try expectEqual(vec_exp, mat0.getRowVec(1));
}

test "Mat33f.getColVec" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    const vec_exp = Vec3f.initSlice(&.{ 2, 5, 8 });

    try expectEqual(vec_exp, mat0.getColVec(1));
}

test "Mat33f.getSubMat" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    var mat_exp = Mat22f.initRows(.{
        .{ 5, 6 },
        .{ 8, 9 },
    });

    try expectEqual(mat_exp, mat0.getSubMat(1, 1, 2, 2));

    mat_exp = Mat22f.initRows(.{
        .{ 2, 3 },
        .{ 5, 6 },
    });

    try expectEqual(mat_exp, mat0.getSubMat(0, 1, 2, 2));

    mat_exp = Mat22f.initRows(.{
        .{ 1, 2 },
        .{ 4, 5 },
    });

    try expectEqual(mat_exp, mat0.getSubMat(0, 0, 2, 2));

    const mat_exp33 = mat0;

    try expectEqual(mat_exp33, mat0.getSubMat(0, 0, 3, 3));
}

test "Mat33f.transpose" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    const mat_exp = Mat33f.initRows(.{
        .{ 1, 4, 7 },
        .{ 2, 5, 8 },
        .{ 3, 6, 9 },
    });

    try expectEqual(mat_exp, mat0.transpose());
}

test "Mat33f.mulScal" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    const scal: TestType = 2;

    const mat_exp = Mat33f.initRows(.{
        .{ 2, 4, 6 },
        .{ 8, 10, 12 },
        .{ 14, 16, 18 },
    });

    try expectEqual(mat_exp, mat0.mulScal(scal));
}

test "Mat33f.mulVec" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    const vec0 = Vec3f.initSlice(&.{ 3, 2, 1 });
    const vec_exp = Vec3f.initSlice(&.{ 10, 28, 46 });

    try expectEqual(vec_exp, mat0.mulVec(vec0));
}

test "Mat33f.mulMat" {
    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });

    const mat1 = Mat33f.initRows(.{
        .{ 3, 1, 1 },
        .{ 1, 3, 1 },
        .{ 1, 1, 3 },
    });

    const mat_exp = Mat33f.initRows(.{
        .{ 8, 10, 12 },
        .{ 23, 25, 27 },
        .{ 38, 40, 42 },
    });

    try expectEqual(mat_exp, mat0.mulMat(mat1));
}

test "Mat33f.mulMat.identity_and_negative" {
    const mat_ident = Mat33f.initIdentity();

    const mat0 = Mat33f.initRows(.{
        .{ 1, -2, 3 },
        .{ 0, 4, -1 },
        .{ -1, 2, 1 },
    });
    try expectEqual(mat0, mat0.mulMat(mat_ident));

    const mat1 = Mat33f.initRows(.{
        .{ 2, 1, 0 },
        .{ -1, 3, 2 },
        .{ 4, 0, -2 },
    });

    const mat_exp = Mat33f.initRows(.{
        .{ 16, -5, -10 },
        .{ -8, 12, 10 },
        .{ 0, 5, 2 },
    });
    try expectEqual(mat_exp, mat0.mulMat(mat1));
}

test "Mat33Ops.det" {
    const mat_ident = Mat33f.initIdentity();
    try expectEqual(1, Mat33Ops.det(TestType, mat_ident));

    const mat0 = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });
    const det0_exp: TestType = 0;

    const mat1 = Mat33f.initRows(.{
        .{ 3, 1, 1 },
        .{ 1, 3, 1 },
        .{ 1, 1, 3 },
    });
    const det1_exp: TestType = 20;

    try expectEqual(det0_exp, Mat33Ops.det(TestType, mat0));
    try expectEqual(det1_exp, Mat33Ops.det(TestType, mat1));

    // Zero determinant (dependent rows: row 1 = 2 * row 0)
    const mat_zero = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 2, 4, 6 },
        .{ 5, 1, 0 },
    });
    try expectEqual(0, Mat33Ops.det(TestType, mat_zero));

    // Negative determinant: det = -21
    const mat_neg = Mat33f.initRows(.{
        .{ 2, -1, 3 },
        .{ 1, 0, 4 },
        .{ 3, 2, 1 },
    });
    try expectEqual(-21, Mat33Ops.det(TestType, mat_neg));

    // Permutation / reflection matrix: det = -1
    const mat_perm = Mat33f.initRows(.{
        .{ 0, 1, 0 },
        .{ 1, 0, 0 },
        .{ 0, 0, 1 },
    });
    try expectEqual(-1, Mat33Ops.det(TestType, mat_perm));
}

test "Mat33Ops.inv" {
    const mat1 = Mat33f.initRows(.{
        .{ 3, 1, 1 },
        .{ 1, 3, 1 },
        .{ 1, 1, 3 },
    });

    const mat_exp = Mat33f.initRows(.{
        .{ 0.4, -0.1, -0.1 },
        .{ -0.1, 0.4, -0.1 },
        .{ -0.1, -0.1, 0.4 },
    });

    try expectEqual(mat_exp, Mat33Ops.inv(TestType, mat1));

    // Inversion with negative determinant: det = -5
    const mat_neg = Mat33f.initRows(.{
        .{ 0, 1, 1 },
        .{ 1, 2, 0 },
        .{ 2, 0, 1 },
    });

    const mat_neg_exp = Mat33f.initRows(.{
        .{ -0.4, 0.2, 0.4 },
        .{ 0.2, 0.4, -0.2 },
        .{ 0.8, -0.4, 0.2 },
    });
    try expectEqual(mat_neg_exp, Mat33Ops.inv(TestType, mat_neg));
}

test "Mat33Ops.invChecked rejects singular matrices" {
    const singular = Mat33f.initRows(.{
        .{ 1, 2, 3 },
        .{ 2, 4, 6 },
        .{ 0, 1, 0 },
    });
    try std.testing.expectError(
        error.SingularMatrix,
        Mat33Ops.invChecked(TestType, singular, 0),
    );
}

test "Mat44f.insertRowVec" {
    var mat0 = Mat44f.initZeros();
    const vec0 = Vec2f.initOnes();
    const vec1 = Vec3f.initOnes();

    const mat_exp1 = Mat44f.initRows(.{
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 1, 1, 0 },
        .{ 0, 0, 0, 0 },
    });

    const mat_exp2 = Mat44f.initRows(.{
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 1, 1, 0 },
        .{ 0, 1, 1, 1 },
    });

    const mat_exp3 = Mat44f.initRows(.{
        .{ 1, 1, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 1, 1, 0 },
        .{ 0, 1, 1, 1 },
    });

    mat0.insertRowVec(2, 1, 2, vec0);
    try expectEqual(mat_exp1, mat0);

    mat0.insertRowVec(3, 1, 3, vec1);
    try expectEqual(mat_exp2, mat0);

    mat0.insertRowVec(0, 0, 2, vec0);
    try expectEqual(mat_exp3, mat0);
}

test "Mat44f.insertColVec" {
    var mat0 = Mat44f.initZeros();
    const vec0 = Vec2f.initOnes();
    const vec1 = Vec3f.initOnes();

    const mat_exp1 = Mat44f.initRows(.{
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 1 },
        .{ 0, 0, 0, 1 },
    });

    const mat_exp2 = Mat44f.initRows(.{
        .{ 1, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 1, 0, 0, 1 },
        .{ 0, 0, 0, 1 },
    });

    const mat_exp3 = Mat44f.initRows(.{
        .{ 1, 0, 1, 0 },
        .{ 1, 0, 1, 0 },
        .{ 1, 0, 0, 1 },
        .{ 0, 0, 0, 1 },
    });

    mat0.insertColVec(3, 2, 2, vec0);
    try expectEqual(mat_exp1, mat0);

    mat0.insertColVec(0, 0, 3, vec1);
    try expectEqual(mat_exp2, mat0);

    mat0.insertColVec(2, 0, 2, vec0);
    try expectEqual(mat_exp3, mat0);
}

test "Mat44f.insertSubMat" {
    var mat0 = Mat44f.initZeros();
    const mat1 = Mat22f.initOnes();
    const mat2 = Mat33f.initOnes();

    const mat_exp1 = Mat44f.initRows(.{
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 1, 1 },
        .{ 0, 0, 1, 1 },
    });

    const mat_exp2 = Mat44f.initRows(.{
        .{ 1, 1, 1, 0 },
        .{ 1, 1, 1, 0 },
        .{ 1, 1, 1, 1 },
        .{ 0, 0, 1, 1 },
    });

    mat0.insertSubMat(2, 2, 2, 2, mat1);
    try expectEqual(mat_exp1, mat0);

    mat0.insertSubMat(0, 0, 3, 3, mat2);
    try expectEqual(mat_exp2, mat0);
}

test "MatStack insertSubMat accepts rectangular matrices" {
    const Mat23 = MatStack(2, 3, TestType);
    var destination = Mat44f.initZeros();
    const source = Mat23.initRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    destination.insertSubMat(1, 0, 2, 3, source);

    const expected = Mat44f.initRows(.{
        .{ 0, 0, 0, 0 },
        .{ 1, 2, 3, 0 },
        .{ 4, 5, 6, 0 },
        .{ 0, 0, 0, 0 },
    });
    try expectEqual(expected, destination);
}

test "Mat44f.mulMat" {
    const mat_ident = Mat44f.initIdentity();

    const mat0 = Mat44f.initRows(.{
        .{ 1, 0, 2, 0 },
        .{ 0, 1, 0, 2 },
        .{ 2, 0, 1, 0 },
        .{ 0, 2, 0, 1 },
    });
    try expectEqual(mat0, mat0.mulMat(mat_ident));

    const mat1 = Mat44f.initRows(.{
        .{ 2, 1, 0, 0 },
        .{ 1, 2, 0, 0 },
        .{ 0, 0, 2, 1 },
        .{ 0, 0, 1, 2 },
    });

    const mat_exp = Mat44f.initRows(.{
        .{ 2, 1, 4, 2 },
        .{ 1, 2, 2, 4 },
        .{ 4, 2, 2, 1 },
        .{ 2, 4, 1, 2 },
    });
    try expectEqual(mat_exp, mat0.mulMat(mat1));
}

test "Mat44Ops.det" {
    const mat_ident = Mat44f.initIdentity();
    try expectEqual(1, Mat44Ops.det(TestType, mat_ident));

    const mat0 = Mat44f.initRows(.{
        .{ 1, 2, 3, 4 },
        .{ 5, 6, 7, 8 },
        .{ 9, 10, 11, 12 },
        .{ 13, 14, 15, 16 },
    });
    var det_exp: TestType = 0;
    try expectEqual(det_exp, Mat44Ops.det(TestType, mat0));

    const mat1 = Mat44f.initRows(.{
        .{ 1, 2, 1, 2 },
        .{ 3, 1, 1, 3 },
        .{ 3, 1, 2, 3 },
        .{ 2, 1, 2, 1 },
    });
    det_exp = 6;
    try expectEqual(det_exp, Mat44Ops.det(TestType, mat1));

    // Zero determinant (dependent rows: row 1 = 2 * row 0)
    const mat_zero = Mat44f.initRows(.{
        .{ 1, 2, 3, 4 },
        .{ 2, 4, 6, 8 },
        .{ 1, 0, 1, 0 },
        .{ 0, 1, 0, 1 },
    });
    try expectEqual(0, Mat44Ops.det(TestType, mat_zero));

    // Negative determinant: det = -2
    const mat_neg = Mat44f.initRows(.{
        .{ 1, 2, 0, 0 },
        .{ 3, 4, 0, 0 },
        .{ 0, 0, 1, 1 },
        .{ 0, 0, 1, 2 },
    });
    try expectEqual(-2, Mat44Ops.det(TestType, mat_neg));
}

test "Mat44Ops.insertMat22" {
    var mat0 = Mat44f.initZeros();
    const mat1 = Mat22f.initOnes();

    const mat_exp = Mat44f.initRows(.{
        .{ 0, 0, 0, 0 },
        .{ 0, 1, 1, 0 },
        .{ 0, 1, 1, 0 },
        .{ 0, 0, 0, 0 },
    });

    Mat44Ops.insertMat22(TestType, &mat0, mat1, 1, 1);

    try expectEqual(mat_exp, mat0);
}

test "Mat44Ops.inv" {
    const mat0 = Mat44f.initRows(.{
        .{ 0, 2, 0, 2 },
        .{ 2, 1, 1, 2 },
        .{ 2, 1, 2, 2 },
        .{ 2, 1, 2, 1 },
    });

    const mat_exp = Mat44f.initRows(.{
        .{ -0.25, 1.0, -1.0, 0.5 },
        .{ 0.5, 0.0, -1.0, 1.0 },
        .{ 0.0, -1.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0, -1.0 },
    });

    try expectEqual(mat_exp, Mat44Ops.inv(TestType, mat0));
}

test "Mat44Ops.inv.negative_det" {
    const mat0 = Mat44f.initRows(.{
        .{ 1, 2, 0, 0 },
        .{ 3, 4, 0, 0 },
        .{ 0, 0, 1, 1 },
        .{ 0, 0, 1, 2 },
    });

    const mat_exp = Mat44f.initRows(.{
        .{ -2.0, 1.0, 0.0, 0.0 },
        .{ 1.5, -0.5, 0.0, 0.0 },
        .{ 0.0, 0.0, 2.0, -1.0 },
        .{ 0.0, 0.0, -1.0, 1.0 },
    });
    try expectEqual(mat_exp, Mat44Ops.inv(TestType, mat0));
}

test "Mat44Ops.invChecked rejects singular matrices" {
    const singular = Mat44f.initRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 0 },
    });
    try std.testing.expectError(
        error.SingularMatrix,
        Mat44Ops.invChecked(TestType, singular, 0),
    );
}

test "MatStack constructors and rectangular matrix-vector multiplication" {
    const Mat23 = MatStack(2, 3, i32);
    const Vec3 = VecStack(3, i32);
    const matrix = Mat23.initRows(.{ .{ 1, 2, 3 }, .{ 4, 5, 6 } });
    const product = matrix.mulVec(Vec3.initSlice(&.{ 1, 0, -1 }));

    try std.testing.expectEqualSlices(i32, &.{ -2, -2 }, &product.vec);
    try expectEqual(Mat23.initOnes(), Mat23.initFill(1));
    try expectEqual(Mat23.initZeros(), Mat23.initFill(0));
    try expectEqual(@as(i32, 2), matrix.getRowVec(0).get(1));
    try expectEqual(@as(i32, 6), matrix.getColVec(2).get(1));
}

test "Mat22 adjugate and Mat44 affine vector multiplication" {
    const matrix = Mat22f.initRows(.{ .{ 1, 2 }, .{ 3, 4 } });
    const expected_adjugate = Mat22f.initRows(.{ .{ 4, -2 }, .{ -3, 1 } });
    try expectEqual(expected_adjugate, Mat22Ops.adj(f64, matrix));

    const transform = Mat44f.initRows(.{
        .{ 1, 0, 0, 10 },
        .{ 0, 1, 0, 20 },
        .{ 0, 0, 1, 30 },
        .{ 0, 0, 0, 1 },
    });
    const transformed = Mat44Ops.mulVec3(f64, transform, vecstack.initVec3(f64, 1, 2, 3));
    try expectEqual(Vec3f.initSlice(&.{ 11, 22, 33 }), transformed);
}

test "MatStack.mulMatRect rectangular multiplication" {
    const Mat23 = MatStack(2, 3, f64);
    const Mat32 = MatStack(3, 2, f64);
    const mat_a = Mat23.initRows(.{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    const mat_b = Mat32.initRows(.{
        .{ 7.0, 8.0 },
        .{ 9.0, 1.0 },
        .{ 2.0, 3.0 },
    });
    const prod = mat_a.mulMatRect(2, mat_b);
    try std.testing.expectEqual(@as(f64, 31.0), prod.get(0, 0));
    try std.testing.expectEqual(@as(f64, 19.0), prod.get(0, 1));
    try std.testing.expectEqual(@as(f64, 85.0), prod.get(1, 0));
    try std.testing.expectEqual(@as(f64, 55.0), prod.get(1, 1));
}

test "MatStack direct mat field access and get set compatibility" {
    var matrix = Mat22f{
        .mat = .{
            .{ 1.0, 2.0 },
            .{ 3.0, 4.0 },
        },
    };

    try expectEqual(@as(f64, 1.0), matrix.mat[0][0]);
    try expectEqual(@as(f64, 2.0), matrix.mat[0][1]);
    try expectEqual(@as(f64, 3.0), matrix.mat[1][0]);
    try expectEqual(@as(f64, 4.0), matrix.mat[1][1]);

    try expectEqual(matrix.mat[0][0], matrix.get(0, 0));
    try expectEqual(matrix.mat[0][1], matrix.get(0, 1));
    try expectEqual(matrix.mat[1][0], matrix.get(1, 0));
    try expectEqual(matrix.mat[1][1], matrix.get(1, 1));

    matrix.mat[1][0] = 42.0;
    try expectEqual(@as(f64, 42.0), matrix.get(1, 0));

    matrix.set(1, 0, 99.0);
    try expectEqual(@as(f64, 99.0), matrix.mat[1][0]);
}

test "MatStack row slice access and whole row copy" {
    var matrix = MatStack(2, 3, f64).initRows(.{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });

    const row_zero_slice: []const f64 = &matrix.mat[0];
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0 }, row_zero_slice);

    const replacement_row = [_]f64{ 7.0, 8.0, 9.0 };
    @memcpy(&matrix.mat[1], &replacement_row);
    try expectEqual(@as(f64, 7.0), matrix.get(1, 0));
    try expectEqual(@as(f64, 8.0), matrix.get(1, 1));
    try expectEqual(@as(f64, 9.0), matrix.get(1, 2));
}

test "MatStack initSlice flat array compatibility" {
    const flat_source = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const matrix = MatStack(2, 3, f64).initSlice(&flat_source);

    try expectEqual(@as(f64, 1.0), matrix.mat[0][0]);
    try expectEqual(@as(f64, 2.0), matrix.mat[0][1]);
    try expectEqual(@as(f64, 3.0), matrix.mat[0][2]);
    try expectEqual(@as(f64, 4.0), matrix.mat[1][0]);
    try expectEqual(@as(f64, 5.0), matrix.mat[1][1]);
    try expectEqual(@as(f64, 6.0), matrix.mat[1][2]);
}
