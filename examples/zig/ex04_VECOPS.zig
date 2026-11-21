// Inputs sourced from shared numerical fixtures.
const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    const v = lm.Vec(3, f64).initSlice(&.{ -0.75, -0.625, -0.5 });
    std.debug.print("L1={d}, L2={d}, squared L2={d}\n", .{ v.normL1(), v.normL2(), v.normL2Square() });
    const b = lm.Vec(3, f64).initSlice(&.{ 0.25, 0.5, 0.75 });
    const result = lm.vecops.axpyVec(3, f64, -0.5, v, b);
    std.debug.print("dot={d}, sum={d}, axpy={any}\n", .{ lm.vecops.dotVec(3, f64, v, b), v.sum(), result.vec });
}
