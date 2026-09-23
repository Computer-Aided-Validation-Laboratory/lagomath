const std = @import("std");
const lagomath = @import("lagomath");

pub fn main(init: std.process.Init) !void {
    try exercise(init.gpa);
}

fn exercise(alloc: std.mem.Allocator) !void {
    const Vec3 = lagomath.VecStack(3, f64);
    const vector = Vec3.initSlice(&.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(f64, 14), vector.dot(vector));

    const Mat22 = lagomath.MatStack(2, 2, f64);
    const matrix = Mat22.initRows(.{ .{ 1, 2 }, .{ 3, 4 } });
    const multiplied = matrix.mulVec(lagomath.VecStack(2, f64).initSlice(&.{ 5, 6 }));
    try std.testing.expectEqualSlices(f64, &.{ 17, 39 }, &multiplied.slice);

    var slice_values = [_]f64{ 1, 2, 3 };
    const vector_slice = lagomath.VecSlice(f64).init(&slice_values);
    vector_slice.mulScalInPlace(2);
    try std.testing.expectEqualSlices(f64, &.{ 2, 4, 6 }, vector_slice.slice);

    var matrix_values = [_]f64{ 1, 2, 3, 4 };
    const matrix_slice = lagomath.MatSlice(f64).init(&matrix_values, 2, 2);
    try std.testing.expectEqual(@as(f64, 5), matrix_slice.trace());

    var array = try lagomath.NDArray(f64).initFlat(alloc, &.{ 2, 2 });
    defer alloc.free(array.slice);
    defer array.deinit(alloc);
    array.set(&.{ 1, 0 }, 7);
    try std.testing.expectEqual(@as(f64, 7), array.get(&.{ 1, 0 }));

    var output = [_]f64{ 0, 0, 0 };
    lagomath.sliceops.add(f64, &.{ 1, 2, 3 }, &.{ 4, 5, 6 }, &output);
    try std.testing.expectEqualSlices(f64, &.{ 5, 7, 9 }, &output);
}
