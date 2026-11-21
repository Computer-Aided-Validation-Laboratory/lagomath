// Inputs sourced from shared numerical fixtures.
const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    const a = lm.Mat(2, 2, f64).initSlice(&.{ -0.625, -0.5, -0.375, -0.25 });
    const b = lm.Vec(2, f64).initSlice(&.{ 0.0, 0.25 });
    const result = lm.matops.mulVecMat(2, 2, f64, a, b);
    std.debug.print("product={any}\n", .{result.vec});
    const rectangular = lm.Mat(2, 3, f64).initSlice(&.{ -0.625, -0.5, -0.375, -0.25, -0.125, 0.0 });
    const rhs = lm.Mat(3, 4, f64).initSlice(&.{ -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, -0.75, -0.5, -0.25, 0.0, 0.25 });
    const product_rect = lm.matops.mulMatMat(2, 3, 4, f64, rectangular, rhs);
    const six = lm.Vec(6, f64).initSlice(&.{ -0.75, -0.625, -0.5, -0.375, -0.25, -0.125 });
    const hessian = lm.matops.outerAddMat(6, 6, f64, 1, six, six, lm.Mat(6, 6, f64).initZeros());
    const small = lm.Mat(2, 2, f64).initSlice(&.{ 3.0267459962185654, -2.302533430200883, 0.8434609434185725, 1.9640895896025747 });
    std.debug.print("rectangular={any}, Hessian00={d}, det={d}\n", .{ product_rect.mat, hessian.get(0, 0), lm.Mat22Ops.det(f64, small) });
}
