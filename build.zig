const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lagomath = b.addModule("lagomath", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_module = lagomath,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);

    const verbose_tests = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "test",
        "src/root.zig",
    });
    const verbose_test_step = b.step("test-verbose", "Run tests with per-test output");
    verbose_test_step.dependOn(&verbose_tests.step);

    const example = b.addExecutable(.{
        .name = "lagomath-basic",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "lagomath", .module = lagomath },
            },
        }),
    });
    const run_example = b.addRunArtifact(example);
    const example_step = b.step("run-example", "Run the basic example");
    example_step.dependOn(&run_example.step);
}
