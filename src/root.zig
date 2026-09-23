pub const ndarray = @import("ndarray.zig");
pub const matslice = @import("matslice.zig");
pub const matstack = @import("matstack.zig");
pub const vecslice = @import("vecslice.zig");
pub const vecstack = @import("vecstack.zig");
pub const sliceops = @import("sliceops.zig");

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
}
const std = @import("std");
