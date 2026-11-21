// Inputs sourced from shared numerical fixtures.
const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    const a = lm.Mat(2, 2, f64).initSlice(&.{ 12.419444258027395, -0.18022820979080997, -0.18022820979080997, 3.769247759293302 });
    const b = lm.Vec(2, f64).initSlice(&.{ 1.0, 2.0 });
    const factor = try lm.linsolve.factorCholeskyMat(2, f64, a);
    const result = try lm.linsolve.solveCholeskyFactoredMat(2, f64, factor, b);
    std.debug.print("solution={any}\n", .{result.vec});
    const second = try lm.linsolve.solveCholeskyFactoredMat(2, f64, factor, b.mulScal(2));
    std.debug.print("reused factor, second RHS={any}\n", .{second.vec});
}
