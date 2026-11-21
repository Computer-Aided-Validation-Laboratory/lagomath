//! SIMD over contiguous row-major NDArray backing storage. Metadata is unchanged.
const std = @import("std");
const arrays = @import("ndarraytype.zig");
const NDArray = arrays.NDArray;
const core = @import("sliceops_simd.zig");
pub fn addNDArraySIMD(comptime T: type, a: *const NDArray(T), b: *const NDArray(T), out: *NDArray(T)) void {
    std.debug.assert(arrays.matchArrayDims(T, a, b) and arrays.matchArrayDims(T, a, out));
    core.addSIMD(T, a.slice, b.slice, out.slice);
}
pub fn subNDArraySIMD(comptime T: type, a: *const NDArray(T), b: *const NDArray(T), out: *NDArray(T)) void {
    std.debug.assert(arrays.matchArrayDims(T, a, b) and arrays.matchArrayDims(T, a, out));
    core.subSIMD(T, a.slice, b.slice, out.slice);
}
pub fn mulElemWiseNDArraySIMD(comptime T: type, a: *const NDArray(T), b: *const NDArray(T), out: *NDArray(T)) void {
    std.debug.assert(arrays.matchArrayDims(T, a, b) and arrays.matchArrayDims(T, a, out));
    core.mulSIMD(T, a.slice, b.slice, out.slice);
}
pub fn divElemWiseNDArraySIMD(comptime T: type, a: *const NDArray(T), b: *const NDArray(T), out: *NDArray(T)) void {
    std.debug.assert(arrays.matchArrayDims(T, a, b) and arrays.matchArrayDims(T, a, out));
    core.divSIMD(T, a.slice, b.slice, out.slice);
}
pub fn fillNDArraySIMD(comptime T: type, a: *const NDArray(T), value: T) void {
    core.fillSIMD(T, a.slice, value);
}
pub fn mulScalNDArraySIMD(comptime T: type, a: *const NDArray(T), alpha: T) void {
    core.mulScalSIMD(T, a.slice, alpha, a.slice);
}
