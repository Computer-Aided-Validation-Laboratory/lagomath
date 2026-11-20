// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const F = f64;
const print = std.debug.print;
const assert = std.debug.assert;

pub const SliceError = error{LengthMismatch};

pub fn ValIdx(ValType: type) type {
    return struct {
        val: ValType,
        idx: usize,
    };
}

pub fn max(comptime T: type, slice: []const T) ValIdx(T) {
    assert(slice.len > 0);

    var val_idx = ValIdx(T){
        .val = slice[0],
        .idx = 0,
    };

    for (slice[1..], 1..) |elem, ii| {
        if (elem > val_idx.val) {
            val_idx.idx = ii;
            val_idx.val = elem;
        }
    }

    return val_idx;
}

pub fn min(comptime T: type, slice: []const T) ValIdx(T) {
    assert(slice.len > 0);

    var val_idx = ValIdx(T){
        .val = slice[0],
        .idx = 0,
    };

    for (slice[1..], 1..) |elem, ii| {
        if (elem < val_idx.val) {
            val_idx.idx = ii;
            val_idx.val = elem;
        }
    }

    return val_idx;
}

pub fn sum(comptime T: type, slice: []const T) T {
    assert(slice.len > 0);

    var sum_out: T = 0;
    for (slice[0..]) |elem| {
        sum_out += elem;
    }
    return sum_out;
}

pub fn mean(comptime T: type, slice: []const T) T {
    return sum(T, slice) / @as(T, @floatFromInt(slice.len));
}

pub fn exp(value: anytype) @TypeOf(value) {
    return @exp(value);
}

pub fn apply(
    comptime T: type,
    dest: []T,
    source: []const T,
    comptime func: anytype,
) void {
    for (dest[0..source.len], source) |*dd, ss| {
        dd.* = func(ss);
    }
}

pub fn rangeLen(start: F, stop: F, step: F) usize {
    const range: F = @ceil((stop - start) / step);
    const range_length: usize = @as(usize, @intFromFloat(range));
    return range_length;
}

pub fn dot(comptime T: type, slice0: []const T, slice1: []const T) T {
    assert(slice0.len == slice1.len);

    var dot_prod: T = 0;
    for (0..slice0.len) |ii| {
        dot_prod += slice0[ii] * slice1[ii];
    }
    return dot_prod;
}

pub fn normL1(comptime T: type, vec: []const T) T {
    var value: T = 0;
    for (vec) |v| value += @abs(v);
    return value;
}
pub fn normL2Square(comptime T: type, vec: []const T) T {
    return dot(T, vec, vec);
}
pub fn normL2(comptime T: type, vec: []const T) T {
    return @sqrt(normL2Square(T, vec));
}

pub fn norm(comptime T: type, vec: []const T) T {
    var norm_out: T = 0;

    for (0..vec.len) |ii| {
        norm_out += vec[ii] * vec[ii];
    }

    return norm_out;
}

pub fn vecLen(comptime T: type, vec: []const T) T {
    return @sqrt(norm(T, vec));
}

pub fn add(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    addUnchecked(T, vec0, vec1, vec_out);
}

pub fn addChecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) SliceError!void {
    if (vec0.len != vec1.len or vec0.len != vec_out.len) return error.LengthMismatch;

    addUnchecked(T, vec0, vec1, vec_out);
}

fn addUnchecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] + vec1[ii];
    }
}

pub fn sub(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    subUnchecked(T, vec0, vec1, vec_out);
}

pub fn subChecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) SliceError!void {
    if (vec0.len != vec1.len or vec0.len != vec_out.len) return error.LengthMismatch;
    subUnchecked(T, vec0, vec1, vec_out);
}

fn subUnchecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] - vec1[ii];
    }
}

pub fn mul(
    comptime T: type,
    vec0: []const T,
    vec1: []const T,
    vec_out: []T,
) void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    mulUnchecked(T, vec0, vec1, vec_out);
}

pub fn mulChecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) SliceError!void {
    if (vec0.len != vec1.len or vec0.len != vec_out.len) return error.LengthMismatch;
    mulUnchecked(T, vec0, vec1, vec_out);
}

fn mulUnchecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] * vec1[ii];
    }
}

pub fn div(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    assert(vec0.len == vec1.len);
    assert(vec0.len == vec_out.len);

    divUnchecked(T, vec0, vec1, vec_out);
}

pub fn divChecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) SliceError!void {
    if (vec0.len != vec1.len or vec0.len != vec_out.len) return error.LengthMismatch;
    divUnchecked(T, vec0, vec1, vec_out);
}

fn divUnchecked(comptime T: type, vec0: []const T, vec1: []const T, vec_out: []T) void {
    for (0..vec0.len) |ii| {
        vec_out[ii] = vec0[ii] / vec1[ii];
    }
}

pub fn mulScal(comptime T: type, vec0: []const T, scal: T, vec_out: []T) void {
    assert(vec0.len == vec_out.len);

    mulScalUnchecked(T, vec0, scal, vec_out);
}

pub fn mulScalChecked(comptime T: type, vec0: []const T, scal: T, vec_out: []T) SliceError!void {
    if (vec0.len != vec_out.len) return error.LengthMismatch;
    mulScalUnchecked(T, vec0, scal, vec_out);
}

fn mulScalUnchecked(comptime T: type, vec0: []const T, scal: T, vec_out: []T) void {
    for (0..vec0.len) |ii| {
        vec_out[ii] = scal * vec0[ii];
    }
}

pub fn slicePrint(comptime T: type, slice: []const T) void {
    print("[", .{});
    for (0..slice.len) |ii| {
        print("{},", .{slice[ii]});
    }
    print("]\n", .{});
}

const TestType = F;

const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;

test "slice.add" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_add");
}

test "slice.sub" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_sub");
}

test "slice.mul" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_mul");
}

test "slice.div" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_div");
}

test "slice.mulScal" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_mulScal");
}

test "slice.apply" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_apply");
}

test "slice.max" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_max");
}

test "slice.min" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_min");
}

test "slice.sum" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_sum");
}

test "slice.mean" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_mean");
}

test "slice reductions and range length" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_reductions_and_range_length");
}

test "slice operations support f32 and partial apply output" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.slice_operations_support_f32_and_partial_apply_output");
}

test "checked slice operations reject mismatched lengths" {
    try @import("case_runner.zig").verifyLegacy("legacy.sliceops.checked_slice_operations_reject_mismatched_lengths");
}
