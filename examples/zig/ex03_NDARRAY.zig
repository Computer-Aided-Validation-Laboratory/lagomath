// Inputs sourced from shared numerical fixtures.
const std = @import("std");
const lm = @import("lagomath");
pub fn main() !void {
    const alloc = std.heap.page_allocator;
    var array = try lm.NDArray(f64).initFlat(alloc, &.{ 2, 3, 4 });
    defer alloc.free(array.slice);
    defer array.deinit(alloc);
    array.set(&.{ 1, 2, 3 }, 7);
    std.debug.print("offset={d}, value={d}\n", .{ array.offset3(1, 2, 3), array.get(&.{ 1, 2, 3 }) });
}
