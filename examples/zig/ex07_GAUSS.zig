const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    const a = lm.Mat(3, 3, f64).initSlice(&.{ 0.0, 2.0, 1.0, 3.0, -1.0, 2.0, 4.0, -1.0, 5.0 });
    const b = lm.Vec(3, f64).initSlice(&.{ 5.0, 7.0, 15.0 });
    const result = try lm.linsolve.solveGaussMat(3, f64, a, b);
    std.debug.print("pivot-required solution={any}\n", .{result.vec});
}
