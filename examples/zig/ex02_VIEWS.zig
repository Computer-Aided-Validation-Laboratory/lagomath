// Inputs sourced from shared numerical fixtures.
const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    var data = [_]f64{ 1, 2, 3 };
    const view = lm.VecView(f64).init(&data);
    view.set(1, 7);
    std.debug.print("borrowed mutation={d}\n", .{data[1]});
}
