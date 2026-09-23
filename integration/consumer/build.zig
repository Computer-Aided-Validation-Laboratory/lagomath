const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lagomath_dependency = b.dependency("lagomath", .{
        .target = target,
        .optimize = optimize,
    });

    const executable = b.addExecutable(.{
        .name = "lagomath-consumer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{
                    .name = "lagomath",
                    .module = lagomath_dependency.module("lagomath"),
                },
            },
        }),
    });
    const run_executable = b.addRunArtifact(executable);
    const run_step = b.step("run", "Run the package consumer");
    run_step.dependOn(&run_executable.step);
}
