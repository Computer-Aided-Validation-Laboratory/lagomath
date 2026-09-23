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
pub const MatSlice = matslice.MatSlice;
pub const MatSliceOps = matslice.MatSliceOps;
pub const MatStack = matstack.MatStack;
pub const VecSlice = vecslice.VecSlice;
pub const VecStack = vecstack.VecStack;

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
