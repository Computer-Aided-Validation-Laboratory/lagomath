// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");

pub const ndarray = @import("ndarray.zig");
pub const matslice = @import("matslice.zig");
pub const matstack = @import("matstack.zig");
pub const vecslice = @import("vecslice.zig");
pub const vecstack = @import("vecstack.zig");
pub const sliceops = @import("sliceops.zig");

pub const matops = @import("matops.zig");
pub const vecops = @import("vecops.zig");
pub const rowops = @import("rowops.zig");
pub const linsolve = @import("linsolve.zig");

pub const NDArray = ndarray.NDArray;
pub const MatView = matviewtype.MatView;
pub const MatViewOps = matviewtype.MatViewOps;
pub const Mat = mattype.Mat;
pub const VecView = vecviewtype.VecView;
pub const Vec = vectype.Vec;

test {
    _ = ndarray;
    _ = matslice;
    _ = matstack;
    _ = vecslice;
    _ = vecstack;
    _ = sliceops;
    _ = matops;
    _ = vecops;
    _ = rowops;
    _ = linsolve;
}

// Compatibility aliases retained during migration.
pub const VecStack = Vec;
pub const MatStack = Mat;
pub const VecSlice = VecView;
pub const MatSlice = MatView;
pub const MatSliceOps = MatViewOps;
pub const vectype = @import("vectype.zig");
pub const mattype = @import("mattype.zig");
pub const vecviewtype = @import("vecviewtype.zig");
pub const matviewtype = @import("matviewtype.zig");
pub const ndarraytype = @import("ndarraytype.zig");

pub const Mat22Ops = mattype.Mat22Ops;
pub const Mat33Ops = mattype.Mat33Ops;
pub const Mat44Ops = mattype.Mat44Ops;
pub const Vec3Ops = vectype.Vec3Ops;

test "compatibility modules retain canonical type identity" {
    try std.testing.expect(Vec(3, f64) == vecstack.VecStack(3, f64));
    try std.testing.expect(Mat(2, 3, f64) == matstack.MatStack(2, 3, f64));
    try std.testing.expect(VecView(f64) == vecslice.VecSlice(f64));
    try std.testing.expect(MatView(f64) == matslice.MatSlice(f64));
    try std.testing.expect(NDArray(f64) == ndarray.NDArray(f64));
}

pub const Vec3SliceOps = vectype.Vec3SliceOps;
pub const NDArrayOps = ndarraytype.NDArrayOps;
pub const MappedNDArray = ndarraytype.MappedNDArray;
pub const matchArrayDims = ndarraytype.matchArrayDims;
pub const Vec2T = vectype.Vec2T;
pub const Vec3T = vectype.Vec3T;
pub const Vec2f = vectype.Vec2f;
pub const Vec3f = vectype.Vec3f;
pub const initVec2 = vectype.initVec2;
pub const initVec3 = vectype.initVec3;
pub const Mat22T = mattype.Mat22T;
pub const Mat33T = mattype.Mat33T;
pub const Mat44T = mattype.Mat44T;
pub const Mat22f = mattype.Mat22f;
pub const Mat33f = mattype.Mat33f;
pub const Mat44f = mattype.Mat44f;

pub const sliceops_simd = @import("sliceops_simd.zig");
pub const vecops_simd = @import("vecops_simd.zig");
pub const matops_simd = @import("matops_simd.zig");
pub const rowops_simd = @import("rowops_simd.zig");
pub const linsolve_simd = @import("linsolve_simd.zig");
pub const smallmatops_simd = @import("smallmatops_simd.zig");

test {
    _ = @import("simd_tests.zig");
}

pub const ndarrayops_simd = @import("ndarrayops_simd.zig");
