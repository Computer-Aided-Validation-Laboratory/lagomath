// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const F = f64;
const print = std.debug.print;
const assert = std.debug.assert;

const MatView = @import("matviewtype.zig").MatView;
const sliceops = @import("sliceops.zig");

pub fn NDArray(comptime T: type) type {
    return struct {
        slice: []T,
        dims: []usize,
        strides: []usize,

        const Self: type = @This();

        pub fn init(allocator: std.mem.Allocator, slice: []T, dims: []const usize) !Self {
            assert(dims.len > 0);
            for (dims) |dim| assert(dim > 0);
            var dim_prod: usize = dims[0];
            for (1..dims.len) |dd| {
                dim_prod *= dims[dd];
            }

            assert(slice.len >= dim_prod);

            const dims_heap = try allocator.dupe(usize, dims);
            errdefer allocator.free(dims_heap);
            const strides_heap = try allocator.alloc(usize, dims.len);

            var ndarray = NDArray(T){
                .slice = slice,
                .dims = dims_heap,
                .strides = strides_heap,
            };

            for (0..dims.len) |dd| {
                ndarray.strides[dd] = ndarray.calcFlatStride(dd);
            }

            return ndarray;
        }

        pub fn initFlat(allocator: std.mem.Allocator, dims: []const usize) !Self {
            assert(dims.len > 0);
            for (dims) |dim| assert(dim > 0);
            var dim_prod: usize = 1;
            for (dims) |dd| {
                dim_prod *= dd;
            }

            const slice = try allocator.alloc(T, dim_prod);
            errdefer allocator.free(slice);

            return try Self.init(allocator, slice, dims);
        }

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            allocator.free(self.dims);
            allocator.free(self.strides);
        }

        pub fn set(self: *Self, indices: []const usize, in_val: T) void {
            const idx: usize = self.getFlatIdx(indices);
            self.slice[idx] = in_val;
        }

        pub fn get(self: *const Self, indices: []const usize) T {
            const idx: usize = self.getFlatIdx(indices);
            return self.slice[idx];
        }

        pub fn getFlat(self: *const Self, flat_idx: usize) T {
            assert(flat_idx < self.slice.len);
            return self.slice[flat_idx];
        }

        pub fn setFlat(self: *Self, flat_idx: usize, in_val: T) void {
            assert(flat_idx < self.slice.len);
            self.slice[flat_idx] = in_val;
        }

        pub fn getFlatIdx(self: *const Self, indices: []const usize) usize {
            assert(indices.len == self.dims.len);

            var flat: usize = 0;
            for (indices, 0..) |idx, dim| {
                assert(idx < self.dims[dim]);

                flat += idx * self.strides[dim];
            }

            return flat;
        }

        pub fn planeBase(self: *const Self, idx0: usize) usize {
            assert(self.dims.len >= 1);
            assert(idx0 < self.dims[0]);
            return idx0 * self.strides[0];
        }

        pub fn subBase2(self: *const Self, idx0: usize, idx1: usize) usize {
            assert(self.dims.len >= 2);
            assert(idx0 < self.dims[0]);
            assert(idx1 < self.dims[1]);
            return idx0 * self.strides[0] + idx1 * self.strides[1];
        }

        pub fn offset2(
            self: *const Self,
            idx0: usize,
            idx1: usize,
        ) usize {
            assert(self.dims.len == 2);
            return self.subBase2(idx0, idx1);
        }

        pub fn offset3(
            self: *const Self,
            idx0: usize,
            idx1: usize,
            idx2: usize,
        ) usize {
            assert(self.dims.len == 3);
            assert(idx2 < self.dims[2]);
            return self.subBase2(idx0, idx1) + idx2;
        }

        pub fn offset4(
            self: *const Self,
            idx0: usize,
            idx1: usize,
            idx2: usize,
            idx3: usize,
        ) usize {
            assert(self.dims.len == 4);
            assert(idx0 < self.dims[0]);
            assert(idx1 < self.dims[1]);
            assert(idx2 < self.dims[2]);
            assert(idx3 < self.dims[3]);
            return idx0 * self.strides[0] +
                idx1 * self.strides[1] +
                idx2 * self.strides[2] +
                idx3;
        }

        pub fn calcFlatStride(self: *const Self, dim_idx: usize) usize {
            assert(dim_idx < self.dims.len);

            var stride: usize = 1;

            var ii: usize = dim_idx + 1;
            while (ii < self.dims.len) : (ii += 1) {
                stride *= self.dims[ii];
            }

            return stride;
        }

        pub fn fill(self: *const Self, fill_val: T) void {
            @memset(self.slice[0..], fill_val);
        }

        pub fn addInPlace(self: *const Self, to_add: *const Self) void {
            assert(matchArrayDims(T, self, to_add));

            for (0..self.slice.len) |ii| {
                self.slice[ii] += to_add.slice[ii];
            }
        }

        pub fn subInPlace(self: *const Self, to_sub: *const Self) void {
            assert(matchArrayDims(T, self, to_sub));

            for (0..self.slice.len) |ii| {
                self.slice[ii] -= to_sub.slice[ii];
            }
        }

        pub fn mulInPlace(self: *const Self, to_mul: *const Self) void {
            assert(matchArrayDims(T, self, to_mul));

            for (0..self.slice.len) |ii| {
                self.slice[ii] *= to_mul.slice[ii];
            }
        }

        pub fn divInPlace(self: *const Self, to_div: *const Self) void {
            assert(matchArrayDims(T, self, to_div));

            for (0..self.slice.len) |ii| {
                self.slice[ii] /= to_div.slice[ii];
            }
        }

        pub fn mulScalInPlace(self: *const Self, scal: T) void {
            for (0..self.slice.len) |ii| {
                self.slice[ii] *= scal;
            }
        }

        pub fn getSlice(
            self: *const Self,
            fixed_idx: []const usize,
            fixed_dim: usize,
        ) []T {
            assert(fixed_idx.len == self.dims.len);
            assert((fixed_dim + 1) < self.dims.len);

            const start = self.getFlatIdx(fixed_idx);
            const stride = self.strides[fixed_dim];
            return self.slice[start .. start + stride];
        }

        pub fn getPlaneSlice(self: *const Self, chan: usize) []T {
            const start = chan * self.strides[0];
            return self.slice[start..];
        }

        pub fn fixedPrefixView(
            self: *const Self,
            allocator: std.mem.Allocator,
            fixed_prefix: []const usize,
        ) !Self {
            assert(fixed_prefix.len < self.dims.len);

            if (fixed_prefix.len == 0) {
                return try Self.init(allocator, self.slice, self.dims);
            }

            var start: usize = 0;
            for (fixed_prefix, 0..) |idx, dim| {
                assert(idx < self.dims[dim]);
                start += idx * self.strides[dim];
            }

            const prefix_last_dim = fixed_prefix.len - 1;
            const view_len = self.strides[prefix_last_dim];
            return try Self.init(
                allocator,
                self.slice[start .. start + view_len],
                self.dims[fixed_prefix.len..],
            );
        }

        pub fn dupe(self: *const Self, allocator: std.mem.Allocator) !Self {
            const new_slice = try allocator.dupe(T, self.slice);
            errdefer allocator.free(new_slice);
            return try Self.init(allocator, new_slice, self.dims);
        }
    };
}

pub fn MappedNDArray(comptime T: type) type {
    return struct {
        array: NDArray(T),
        map: []usize,

        const Self: type = @This();

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            self.array.deinit(allocator);
            allocator.free(self.map);
        }
    };
}

pub fn NDArrayOps(comptime T: type) type {
    return struct {
        pub fn add(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.add(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn sub(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.sub(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn mulElemWise(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.mul(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn divElemWise(
            arr0: *const NDArray(T),
            arr1: *const NDArray(T),
            arr_out: *NDArray(T),
        ) void {
            assert(matchArrayDims(T, arr0, arr1));
            assert(matchArrayDims(T, arr0, arr_out));

            sliceops.div(T, arr0.slice, arr1.slice, arr_out.slice);
        }

        pub fn extractMat(
            allocator: std.mem.Allocator,
            arr: *const NDArray(T),
            fixed_idxs: []const usize,
            row_ext: usize,
            col_ext: usize,
            mat: *MatView(T),
        ) !void {
            assert(fixed_idxs.len == arr.dims.len);
            assert(row_ext < arr.dims.len);
            assert(col_ext < arr.dims.len);
            assert(row_ext != col_ext);
            const num_elems: usize = arr.dims[row_ext] * arr.dims[col_ext];
            assert(num_elems == mat.slice.len);

            var get_dims_slice: []usize = try allocator.dupe(usize, fixed_idxs);
            defer allocator.free(get_dims_slice);

            for (0..arr.dims[row_ext]) |rr| {
                for (0..arr.dims[col_ext]) |cc| {
                    get_dims_slice[row_ext] = rr;
                    get_dims_slice[col_ext] = cc;

                    mat.set(rr, cc, arr.get(get_dims_slice));
                }
            }
        }
    };
}

pub fn matchArrayDims(
    comptime T: type,
    arr0: *const NDArray(T),
    arr1: *const NDArray(T),
) bool {
    if (arr0.dims.len != arr1.dims.len) {
        return false;
    }

    for (0..arr0.dims.len) |ii| {
        if (arr0.dims[ii] != arr1.dims[ii]) {
            return false;
        }
    }

    return true;
}

const testing = std.testing;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;
const expectEqualSlices = testing.expectEqualSlices;
const talloc = testing.allocator;

test "matchArrayDims" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.matchArrayDims");
}

test "getFlatIdx" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.getFlatIdx");
}

test "calcFlatStride" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.calcFlatStride");
}

test "getSlice" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.getSlice");
}

test "fixedPrefixView" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.fixedPrefixView");
}

test "offset helpers" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.offset_helpers");
}

test "NDArray flat allocation, plane slicing, and duplication ownership" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.NDArray_flat_allocation_plane_slicing_and_duplication_ownership");
}

test "NDArray shape mismatch and mapped metadata cleanup" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.NDArray_shape_mismatch_and_mapped_metadata_cleanup");
}

test "NDArrayOps arithmetic, extraction, and empty prefix view" {
    try @import("case_runner.zig").verifyLegacy("legacy.ndarraytype.NDArrayOps_arithmetic_extraction_and_empty_prefix_view");
}

test "mechanical NDArray metadata and element ownership cleanup" {
    const allocator = std.testing.allocator;
    var owned = try NDArray(f64).initFlat(allocator, &.{ 2, 3, 4 });
    defer allocator.free(owned.slice);
    defer owned.deinit(allocator);
    owned.fill(0);
    var duplicate = try owned.dupe(allocator);
    defer allocator.free(duplicate.slice);
    defer duplicate.deinit(allocator);
    var prefix = try owned.fixedPrefixView(allocator, &.{1});
    defer prefix.deinit(allocator);
    const map = try allocator.alloc(usize, 2);
    const borrowed = try NDArray(f64).init(allocator, owned.slice, &.{ 2, 3, 4 });
    const mapped = MappedNDArray(f64){ .array = borrowed, .map = map };
    mapped.deinit(allocator);
}
