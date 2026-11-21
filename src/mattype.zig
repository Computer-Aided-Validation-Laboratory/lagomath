// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const print = std.debug.print;
const F = f64;

const vecstack = @import("vectype.zig");
const Vec = vecstack.Vec;
const Vec2T = vecstack.Vec2T;
const Vec3T = vecstack.Vec3T;
const Vec2f = vecstack.Vec2f;
const Vec3f = vecstack.Vec3f;

const TestType = F;
pub const Mat22f = Mat22T(F);
pub const Mat33f = Mat33T(F);
pub const Mat44f = Mat44T(F);
pub const MatrixInversionError = error{SingularMatrix};

pub fn Mat(
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

        pub fn getRowVec(self: *const Self, row: usize) Vec(cols_n, T) {
            std.debug.assert(row < rows_n);
            return Vec(cols_n, T).initSlice(&self.mat[row]);
        }

        pub fn getColVec(self: *const Self, col: usize) Vec(rows_n, T) {
            std.debug.assert(col < cols_n);
            var col_vec: [rows_n]T = undefined;
            inline for (0..rows_n) |rr| {
                col_vec[rr] = self.mat[rr][col];
            }
            const vec = Vec(rows_n, T).initSlice(&col_vec);
            return vec;
        }

        pub fn getSubMat(
            self: *const Self,
            row_start: usize,
            col_start: usize,
            comptime rows: usize,
            comptime cols: usize,
        ) Mat(rows, cols, T) {
            var sub_mat = Mat(rows, cols, T).initZeros();

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
            vec: Vec(vec_len, T),
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
            vec: Vec(vec_len, T),
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
            sub_mat: Mat(mat_rows, mat_cols, T),
        ) void {
            inline for (0..mat_rows) |rr| {
                inline for (0..mat_cols) |cc| {
                    self.mat[rr + row_start][cc + col_start] = sub_mat.mat[rr][cc];
                }
            }
        }

        pub fn transpose(self: *const Self) Mat(cols_n, rows_n, T) {
            var mat_out: Mat(cols_n, rows_n, T) = undefined;

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

        pub fn mulVec(self: *const Self, vec: Vec(cols_n, T)) Vec(rows_n, T) {
            var vec_out: Vec(rows_n, T) = undefined;
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
            to_mult: Mat(cols_n, other_cols, T),
        ) Mat(rows_n, other_cols, T) {
            var mat_out = Mat(rows_n, other_cols, T).initZeros();
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
    return Mat(2, 2, T);
}

pub fn Mat33T(comptime T: type) type {
    return Mat(3, 3, T);
}

pub fn Mat44T(comptime T: type) type {
    return Mat(4, 4, T);
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
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_getRowVec");
}

test "Mat22f.getColVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_getColVec");
}

test "Mat22f.add" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_add");
}

test "Mat22f.sub" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_sub");
}

test "Mat22f.trace" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_trace");
}

test "Mat22f.transpose" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_transpose");
}

test "Mat rectangular transpose and row construction" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_rectangular_transpose_and_row_construction");
}

test "Mat22f.mulScal" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_mulScal");
}

test "Mat22f.mulVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_mulVec");
}

test "Mat22f.mulMat" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_mulMat");
}

test "Mat22f.mulMat.identity_and_negative" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22f_mulMat_identity_and_negative");
}

test "Mat22Ops.det" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22Ops_det");
}

test "Mat22Ops.inv" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22Ops_inv");
}

test "Mat22Ops.invChecked rejects singular and near-singular matrices" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22Ops_invChecked_rejects_singular_and_near_singular_matrices");
}

test "Mat33f.add" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_add");
}

test "Mat33f.sub" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_sub");
}

test "Mat33f.getRowVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_getRowVec");
}

test "Mat33f.getColVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_getColVec");
}

test "Mat33f.getSubMat" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_getSubMat");
}

test "Mat33f.transpose" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_transpose");
}

test "Mat33f.mulScal" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_mulScal");
}

test "Mat33f.mulVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_mulVec");
}

test "Mat33f.mulMat" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_mulMat");
}

test "Mat33f.mulMat.identity_and_negative" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33f_mulMat_identity_and_negative");
}

test "Mat33Ops.det" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33Ops_det");
}

test "Mat33Ops.inv" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33Ops_inv");
}

test "Mat33Ops.invChecked rejects singular matrices" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat33Ops_invChecked_rejects_singular_matrices");
}

test "Mat44f.insertRowVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44f_insertRowVec");
}

test "Mat44f.insertColVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44f_insertColVec");
}

test "Mat44f.insertSubMat" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44f_insertSubMat");
}

test "Mat insertSubMat accepts rectangular matrices" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_insertSubMat_accepts_rectangular_matrices");
}

test "Mat44f.mulMat" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44f_mulMat");
}

test "Mat44Ops.det" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44Ops_det");
}

test "Mat44Ops.insertMat22" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44Ops_insertMat22");
}

test "Mat44Ops.inv" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44Ops_inv");
}

test "Mat44Ops.inv.negative_det" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44Ops_inv_negative_det");
}

test "Mat44Ops.invChecked rejects singular matrices" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat44Ops_invChecked_rejects_singular_matrices");
}

test "Mat constructors and rectangular matrix-vector multiplication" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_constructors_and_rectangular_matrix_vector_multiplication");
}

test "Mat22 adjugate and Mat44 affine vector multiplication" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat22_adjugate_and_Mat44_affine_vector_multiplication");
}

test "Mat.mulMatRect rectangular multiplication" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_mulMatRect_rectangular_multiplication");
}

test "Mat direct mat field access and get set compatibility" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_direct_mat_field_access_and_get_set_compatibility");
}

test "Mat row slice access and whole row copy" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_row_slice_access_and_whole_row_copy");
}

test "Mat initSlice flat array compatibility" {
    try @import("case_runner.zig").verifyLegacy("legacy.mattype.Mat_initSlice_flat_array_compatibility");
}
