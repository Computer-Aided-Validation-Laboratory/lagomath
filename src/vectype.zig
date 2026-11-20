// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const F = f64;

const SliceOps = @import("sliceops.zig");
const ValIdx = SliceOps.ValIdx;

const EType = F;
pub const Vec2f = Vec2T(EType);
pub const Vec3f = Vec3T(EType);

pub fn Vec(comptime elem_n: comptime_int, comptime T: type) type {
    return struct {
        vec: [elem_n]T,

        const Self: type = @This();

        pub fn initFill(fill_val: T) Self {
            return .{ .vec = [_]T{fill_val} ** elem_n };
        }

        pub fn initOnes() Self {
            return initFill(1);
        }

        pub fn initZeros() Self {
            return initFill(0);
        }

        pub fn initSlice(slice_in: []const T) Self {
            return .{ .vec = slice_in[0..elem_n].* };
        }

        pub fn get(self: *const Self, ind: usize) T {
            return self.vec[ind];
        }

        pub fn set(self: *Self, ind: usize, val: T) void {
            self.vec[ind] = val;
        }

        pub fn x(self: Self) T {
            return self.vec[0];
        }

        pub fn y(self: Self) T {
            return self.vec[1];
        }

        pub fn z(self: Self) T {
            return self.vec[2];
        }

        pub fn w(self: Self) T {
            return self.vec[3];
        }

        pub fn add(self: *const Self, to_add: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.vec[ii] = self.vec[ii] + to_add.vec[ii];
            }
            return vec_out;
        }

        pub fn sub(self: *const Self, to_sub: Self) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.vec[ii] = self.vec[ii] - to_sub.vec[ii];
            }
            return vec_out;
        }

        pub fn mulScal(self: *const Self, scal: T) Self {
            var vec_out: Self = undefined;
            for (0..elem_n) |ii| {
                vec_out.vec[ii] = scal * self.vec[ii];
            }
            return vec_out;
        }

        pub fn dot(self: *const Self, to_dot: Self) T {
            var dot_prod: T = 0;
            for (0..elem_n) |ii| {
                dot_prod += self.vec[ii] * to_dot.vec[ii];
            }
            return dot_prod;
        }

        pub fn normL1(self: *const Self) T {
            var value: T = 0;
            for (self.vec) |xval| value += @abs(xval);
            return value;
        }
        pub fn normL2Square(self: *const Self) T {
            return self.dot(self.*);
        }
        pub fn normL2(self: *const Self) T {
            return @sqrt(self.normL2Square());
        }

        pub fn norm(self: *const Self) T {
            var norm_out: T = 0;
            for (0..elem_n) |ii| {
                norm_out += self.vec[ii] * self.vec[ii];
            }
            return norm_out;
        }

        pub fn squaredNorm(self: *const Self) T {
            return self.norm();
        }

        pub fn vecLen(self: *const Self) T {
            return @sqrt(self.norm());
        }

        pub fn max(self: *const Self) ValIdx(T) {
            return SliceOps.max(T, &self.vec);
        }

        pub fn min(self: *const Self) ValIdx(T) {
            return SliceOps.min(T, &self.vec);
        }

        pub fn sum(self: *const Self) T {
            return SliceOps.sum(T, &self.vec);
        }

        pub fn mean(self: *const Self) T {
            return SliceOps.mean(T, &self.vec);
        }

        pub fn apply(
            self: *const Self,
            comptime func: anytype,
        ) Self {
            var applied: Self = undefined;
            for (self.vec, 0..) |elem, ii| {
                applied.vec[ii] = func(elem);
            }
            return applied;
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..elem_n) |ii| {
                print("{e:.6},", .{self.vec[ii]});
            }
            print("]\n", .{});
        }
    };
}

pub fn Vec2T(comptime T: type) type {
    return Vec(2, T);
}

pub fn Vec3T(comptime T: type) type {
    return Vec(3, T);
}

pub fn initVec2(comptime T: type, x_in: T, y_in: T) Vec2T(T) {
    return Vec2T(T){
        .vec = [2]T{ x_in, y_in },
    };
}

pub fn initVec3(comptime T: type, x_in: T, y_in: T, z_in: T) Vec3T(T) {
    return Vec3T(T){
        .vec = [3]T{ x_in, y_in, z_in },
    };
}

pub const Vec3Ops = struct {
    pub fn cross(comptime T: type, vec0: Vec3T(T), vec1: Vec3T(T)) Vec3T(T) {
        var vec_out: Vec3T(T) = undefined;
        vec_out.vec[0] = vec0.vec[1] * vec1.vec[2] - vec0.vec[2] * vec1.vec[1];
        vec_out.vec[1] = vec0.vec[0] * vec1.vec[2] - vec0.vec[2] * vec1.vec[0];
        vec_out.vec[2] = vec0.vec[0] * vec1.vec[1] - vec0.vec[1] * vec1.vec[0];
        return vec_out;
    }
};

pub const Vec3SliceOps = struct {
    pub fn max(comptime T: type, vec: []Vec3T(T), ind: usize) T {
        assert(vec.len > 0);
        assert(ind < 3);

        var val: T = vec[0].get(ind);
        for (vec[1..]) |vv| {
            if (vv.get(ind) > val) {
                val = vv.get(ind);
            }
        }

        return val;
    }

    pub fn min(comptime T: type, vec: []Vec3T(T), ind: usize) T {
        assert(vec.len > 0);
        assert(ind < 3);

        var val: T = vec[0].get(ind);
        for (vec[1..]) |vv| {
            if (vv.get(ind) < val) {
                val = vv.get(ind);
            }
        }

        return val;
    }
};

const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;

test "VecSliceOps.max" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.VecSliceOps_max");
}

test "Vec.apply" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_apply");
}

test "Vec.max" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_max");
}

test "Vec.min" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_min");
}

test "Vec.sum" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_sum");
}

test "Vec.mean" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_mean");
}

test "Vec3f.add" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3f_add");
}

test "Vec3f.sub" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3f_sub");
}

test "Vec3f.mulScal" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3f_mulScal");
}

test "Vec3f.dot" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3f_dot");
}

test "Vec3f.norm" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3f_norm");
}

test "Vec3f.length" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3f_length");
}

test "Vec3Ops.cross" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec3Ops_cross");
}

test "Vec constructors, accessors, and integer arithmetic" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_constructors_accessors_and_integer_arithmetic");
}

test "Vec constructors and Vec3SliceOps minimum" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_constructors_and_Vec3SliceOps_minimum");
}

test "Vec direct vec field access and get set compatibility" {
    try @import("case_runner.zig").verifyLegacy("legacy.vectype.Vec_direct_vec_field_access_and_get_set_compatibility");
}
