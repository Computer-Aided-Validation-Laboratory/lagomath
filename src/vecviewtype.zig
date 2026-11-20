// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const F = f64;
const print = std.debug.print;
const assert = std.debug.assert;

const SliceOps = @import("sliceops.zig");
const ValIdx = SliceOps.ValIdx;

const TestType = F;

pub fn VecView(comptime T: type) type {
    return struct {
        slice: []T,

        const Self: type = @This();

        pub fn init(slice: []T) Self {
            return .{
                .slice = slice,
            };
        }

        pub fn fill(self: *const Self, fill_val: T) void {
            @memset(self.slice, fill_val);
        }

        pub fn get(self: *const Self, ind: usize) T {
            return self.slice[ind];
        }

        pub fn set(self: *const Self, ind: usize, val: T) void {
            self.slice[ind] = val;
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            assert(self.slice.len == to_add.slice.len);
            for (0..self.slice.len) |ii| {
                self.slice[ii] += to_add.slice[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            assert(self.slice.len == to_sub.slice.len);
            for (0..self.slice.len) |ii| {
                self.slice[ii] -= to_sub.slice[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            assert(self.slice.len == to_mul.slice.len);
            for (0..self.slice.len) |ii| {
                self.slice[ii] *= to_mul.slice[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            assert(self.slice.len == to_div.slice.len);
            for (0..self.slice.len) |ii| {
                self.slice[ii] /= to_div.slice[ii];
            }
        }

        pub fn mulScalInPlace(self: *const Self, scal: T) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] = scal * self.slice[ii];
            }
        }

        pub fn applyInPlace(
            self: *const Self,
            comptime func: anytype,
        ) void {
            for (self.slice, 0..) |elem, ii| {
                self.slice[ii] = func(elem);
            }
        }

        pub fn dot(self: *const Self, to_dot: Self) T {
            assert(self.slice.len == to_dot.slice.len);
            var dot_prod: T = 0;
            for (0..self.slice.len) |ii| {
                dot_prod += self.slice[ii] * to_dot.slice[ii];
            }
            return dot_prod;
        }

        pub fn normL1(self: *const Self) T {
            var value: T = 0;
            for (self.slice) |xval| value += @abs(xval);
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
            for (0..self.slice.len) |ii| {
                norm_out += self.slice[ii] * self.slice[ii];
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
            return SliceOps.max(T, self.slice);
        }

        pub fn min(self: *const Self) ValIdx(T) {
            return SliceOps.min(T, self.slice);
        }

        pub fn sum(self: *const Self) T {
            return SliceOps.sum(T, self.slice);
        }

        pub fn mean(self: *const Self) T {
            return SliceOps.mean(T, self.slice);
        }

        pub fn vecPrint(self: *const Self) void {
            print("[", .{});
            for (0..self.slice.len) |ii| {
                print("{e:.3},", .{self.slice[ii]});
            }
            print("]\n", .{});
        }
    };
}

const testing = std.testing;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

test "VecView.addInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_addInPlace");
}

test "VecView.subInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_subInPlace");
}

test "VecView.mulInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_mulInPlace");
}

test "VecView.divInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_divInPlace");
}

test "VecView.mulScalInPlace" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_mulScalInPlace");
}

test "VecView.max" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_max");
}

test "VecView.min" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_min");
}

test "VecView.sum" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_sum");
}

test "VecView.mean" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_mean");
}

test "VecView.apply" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_apply");
}

test "VecView access and vector reductions" {
    try @import("case_runner.zig").verifyLegacy("legacy.vecviewtype.VecView_access_and_vector_reductions");
}
