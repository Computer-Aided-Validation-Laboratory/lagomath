// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const F = f64;
const print = std.debug.print;
const assert = std.debug.assert;

const VecView = @import("vecviewtype.zig").VecView;
const sliceops = @import("sliceops.zig");

pub fn MatView(comptime T: type) type {
    return struct {
        slice: []T,
        rows_num: usize,
        cols_num: usize,

        const Self: type = @This();

        pub fn init(slice: []T, rows_num: usize, cols_num: usize) Self {
            assert(slice.len == (rows_num * cols_num));

            return .{
                .slice = slice,
                .rows_num = rows_num,
                .cols_num = cols_num,
            };
        }

        pub fn initAlloc(
            outer_alloc: std.mem.Allocator,
            rows_num: usize,
            cols_num: usize,
        ) !Self {
            const slice = try outer_alloc.alloc(T, rows_num * cols_num);

            return init(slice, rows_num, cols_num);
        }

        pub fn fill(self: *const Self, fill_val: T) void {
            @memset(self.slice[0..], fill_val);
        }

        pub fn fillDiag(self: *Self, fill_val: T, diag_val: T) void {
            for (0..self.rows_num) |ii| {
                for (0..self.cols_num) |jj| {
                    if (ii == jj) {
                        self.set(ii, jj, diag_val);
                    } else {
                        self.set(ii, jj, fill_val);
                    }
                }
            }
        }

        pub fn identity(self: *Self) void {
            self.fillDiag(0, 1);
        }

        pub fn get(self: *const Self, row: usize, col: usize) T {
            assert(row < self.rows_num);
            assert(col < self.cols_num);
            return self.slice[(row * self.cols_num) + col];
        }

        pub fn set(self: *const Self, row: usize, col: usize, val: T) void {
            assert(row < self.rows_num);
            assert(col < self.cols_num);
            self.slice[(row * self.cols_num) + col] = val;
        }

        pub fn flatIndex(self: *const Self, row: usize, col: usize) usize {
            assert(row < self.rows_num);
            assert(col < self.cols_num);
            return row * self.cols_num + col;
        }

        pub fn rowBase(self: *const Self, row: usize) usize {
            assert(row < self.rows_num);
            return row * self.cols_num;
        }

        pub fn getFlat(self: *const Self, flat_idx: usize) T {
            assert(flat_idx < self.slice.len);
            return self.slice[flat_idx];
        }

        pub fn setFlat(self: *const Self, flat_idx: usize, val: T) void {
            assert(flat_idx < self.slice.len);
            self.slice[flat_idx] = val;
        }

        pub fn transposeSquare(self: *Self, buff: *Self) void {
            assert(self.rows_num == self.cols_num);
            assert(self.cols_num == buff.cols_num);
            assert(self.rows_num == buff.rows_num);

            @memcpy(buff.slice, self.slice);

            for (0..self.rows_num) |ii| {
                for (ii..self.cols_num) |jj| {
                    self.set(ii, jj, buff.get(jj, ii));
                    self.set(jj, ii, buff.get(ii, jj));
                }
            }
        }

        pub fn transpose(self: *const Self, output: *Self) void {
            assert(output.rows_num == self.cols_num);
            assert(output.cols_num == self.rows_num);

            for (0..self.rows_num) |rr| {
                for (0..self.cols_num) |cc| {
                    output.set(cc, rr, self.get(rr, cc));
                }
            }
        }

        pub fn trace(self: *const Self) T {
            var trace_out: T = 0;

            if (self.rows_num <= self.cols_num) {
                for (0..self.rows_num) |ii| {
                    trace_out += self.get(ii, ii);
                }
            } else {
                for (0..self.cols_num) |ii| {
                    trace_out += self.get(ii, ii);
                }
            }

            return trace_out;
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            assert(self.rows_num == to_add.rows_num);
            assert(self.cols_num == to_add.cols_num);
            for (0..self.slice.len) |ee| {
                self.slice[ee] += to_add.slice[ee];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            assert(self.rows_num == to_sub.rows_num);
            assert(self.cols_num == to_sub.cols_num);
            for (0..self.slice.len) |ee| {
                self.slice[ee] -= to_sub.slice[ee];
            }
        }

        pub fn mulInPlace(self: *const Self, to_sub: *const Self) void {
            assert(self.rows_num == to_sub.rows_num);
            assert(self.cols_num == to_sub.cols_num);
            for (0..self.slice.len) |ee| {
                self.slice[ee] *= to_sub.slice[ee];
            }
        }

        pub fn divInPlace(self: *const Self, to_sub: *const Self) void {
            assert(self.rows_num == to_sub.rows_num);
            assert(self.cols_num == to_sub.cols_num);
            for (0..self.slice.len) |ee| {
                self.slice[ee] /= to_sub.slice[ee];
            }
        }

        pub fn mulScalInPlace(self: *const Self, scal: T) void {
            for (0..self.slice.len) |ee| {
                self.slice[ee] = scal * self.slice[ee];
            }
        }

        pub fn getSlice(self: *const Self, row_to_slice: usize) []T {
            assert(row_to_slice < self.rows_num);

            const start_idx: usize = row_to_slice * self.cols_num;
            const end_idx: usize = start_idx + self.cols_num;
            return self.slice[start_idx..end_idx];
        }

        pub fn matPrint(self: *const Self) void {
            var ind: usize = 0;

            for (0..self.rows_num) |ii| {
                print("[", .{});
                for (0..self.cols_num) |jj| {
                    ind = (ii * self.cols_num) + jj;
                    print("{e:.3},", .{self.slice[ind]});
                }
                print("]\n", .{});
            }
            print("\n", .{});
        }

        pub fn minByRow(self: *const Self, fixed_col: usize) T {
            assert(fixed_col < self.cols_num);

            var val: T = self.get(0, fixed_col);

            for (1..self.rows_num) |ii| {
                const check = self.get(ii, fixed_col);
                if (check < val) {
                    val = check;
                }
            }

            return val;
        }

        pub fn maxByRow(self: *const Self, fixed_col: usize) T {
            assert(fixed_col < self.cols_num);

            var val: T = self.get(0, fixed_col);

            for (1..self.rows_num) |ii| {
                const check = self.get(ii, fixed_col);
                if (check > val) {
                    val = check;
                }
            }

            return val;
        }
    };
}

pub fn MatViewOps(comptime T: type) type {
    return struct {
        pub fn add(
            mat0: *const MatView(T),
            mat1: *const MatView(T),
            mat_out: *MatView(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);
            assert(mat_out.rows_num == mat0.rows_num);
            assert(mat_out.cols_num == mat0.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] + mat1.slice[ii];
            }
        }

        pub fn sub(
            mat0: *const MatView(T),
            mat1: *const MatView(T),
            mat_out: *MatView(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);
            assert(mat_out.rows_num == mat0.rows_num);
            assert(mat_out.cols_num == mat0.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] - mat1.slice[ii];
            }
        }

        pub fn mulElemWise(
            mat0: *const MatView(T),
            mat1: *const MatView(T),
            mat_out: *MatView(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);
            assert(mat_out.rows_num == mat0.rows_num);
            assert(mat_out.cols_num == mat0.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] * mat1.slice[ii];
            }
        }

        pub fn divElemWise(
            mat0: *const MatView(T),
            mat1: *const MatView(T),
            mat_out: *MatView(T),
        ) void {
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.cols_num == mat1.cols_num);
            assert(mat_out.rows_num == mat0.rows_num);
            assert(mat_out.cols_num == mat0.cols_num);

            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = mat0.slice[ii] / mat1.slice[ii];
            }
        }

        pub fn mulScal(
            mat0: *const MatView(T),
            scal: T,
            mat_out: *MatView(T),
        ) void {
            assert(mat_out.rows_num == mat0.rows_num);
            assert(mat_out.cols_num == mat0.cols_num);
            for (0..mat0.slice.len) |ii| {
                mat_out.slice[ii] = scal * mat0.slice[ii];
            }
        }

        pub fn mulVec(
            mat: *const MatView(T),
            vec_mul: *const VecView(T),
            vec_out: *VecView(T),
        ) void {
            assert(mat.cols_num == vec_mul.slice.len);
            assert(mat.rows_num == vec_out.slice.len);

            var sum: T = 0;

            for (0..mat.rows_num) |rr| {
                sum = 0;
                for (0..mat.cols_num) |cc| {
                    sum += mat.get(rr, cc) * vec_mul.get(cc);
                }
                vec_out.set(rr, sum);
            }
        }

        pub fn mulSquare(
            mat0: *const MatView(T),
            mat1: *const MatView(T),
            mat_out: *MatView(T),
        ) void {
            assert(mat0.rows_num == mat0.cols_num);
            assert(mat1.rows_num == mat1.cols_num);
            assert(mat_out.rows_num == mat_out.cols_num);
            assert(mat0.rows_num == mat1.rows_num);
            assert(mat0.rows_num == mat_out.rows_num);

            mul(mat0, mat1, mat_out);
        }

        pub fn mul(
            mat0: *const MatView(T),
            mat1: *const MatView(T),
            mat_out: *MatView(T),
        ) void {
            assert(mat0.cols_num == mat1.rows_num);
            assert(mat_out.rows_num == mat0.rows_num);
            assert(mat_out.cols_num == mat1.cols_num);

            var sum: T = 0;

            for (0..mat0.rows_num) |rr| {
                for (0..mat1.cols_num) |cc| {
                    sum = 0;

                    for (0..mat0.cols_num) |mm| {
                        sum += mat0.get(rr, mm) * mat1.get(mm, cc);
                    }

                    mat_out.set(rr, cc, sum);
                }
            }
        }
    };
}

const testing = std.testing;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;
const TestType = F;
const talloc = testing.allocator;

test "MatView.getSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_getSlice");
}

test "MatView flat helpers" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_flat_helpers");
}

test "MatViewOps.add" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_add");
}

test "MatViewOps.sub" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_sub");
}

test "MatViewOps.mulElemWise" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_mulElemWise");
}

test "MatViewOps.mulScal" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_mulScal");
}

test "MatViewOps.divElemWise" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_divElemWise");
}

test "MatView.addInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_addInPlace");
}

test "MatView.subInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_subInPlace");
}

test "MatView.trace" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_trace");
}

test "MatView.transpose" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_transpose");
}

test "MatView.mulScal" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_mulScal");
}

test "MatViewOps.mulVec" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_mulVec");
}

test "MatViewOps.mulSquare" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_mulSquare");
}

test "MatView allocation, rectangular trace, and column extrema" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_allocation_rectangular_trace_and_column_extrema");
}

test "MatView in-place multiply and divide" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_in_place_multiply_and_divide");
}

test "MatViewOps multiplies rectangular matrices" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatViewOps_multiplies_rectangular_matrices");
}

test "MatView identity clears dirty storage and transpose supports rectangles" {
    try @import("case_runner.zig").verifyLegacy("legacy.matviewtype.MatView_identity_clears_dirty_storage_and_transpose_supports_rectangles");
}
