const std = @import("std");
const lagomath = @import("lagomath");

pub fn main(init: std.process.Init) !void {
    const Mat22 = lagomath.MatStack(2, 2, f64);
    const Vec2 = lagomath.VecStack(2, f64);

    const matrix = Mat22.initRows(.{ .{ 1, 2 }, .{ 3, 4 } });
    const vector = Vec2.initSlice(&.{ 5, 6 });
    const product = matrix.mulVec(vector);

    try std.testing.expectEqualSlices(f64, &.{ 17, 39 }, &product.slice);

    var array = try lagomath.NDArray(f64).initFlat(init.gpa, &.{ 2, 2 });
    defer init.gpa.free(array.slice);
    defer array.deinit(init.gpa);

    array.set(&.{ 1, 0 }, 7);
    try std.testing.expectEqual(@as(f64, 7), array.get(&.{ 1, 0 }));

    std.debug.print(
        "Lagomath example passed: matrix-vector product = {{{d}, {d}}}, NDArray value = {d}.\n",
        .{ product.get(0), product.get(1), array.get(&.{ 1, 0 }) },
    );
}
