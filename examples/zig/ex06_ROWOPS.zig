// Inputs sourced from shared numerical fixtures.
const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    var data = [_]f64{ 1, 2, 3, 4 };
    const matrix = lm.MatView(f64).init(&data, 2, 2);
    lm.rowops.swapRowsMatView(f64, 0, 1, matrix);
    std.debug.print("swapped={any}\n", .{data});
}
