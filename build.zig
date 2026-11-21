const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lagomath = b.addModule("lagomath", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    lagomath.addImport("shared_cases", b.createModule(.{ .root_source_file = b.path("tests/cases/data.zig") }));

    const tests = b.addTest(.{
        .root_module = lagomath,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);

    const verbose_tests = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "test",
        "--dep",
        "shared_cases",
        "-Mroot=src/root.zig",
        "-Mshared_cases=tests/cases/data.zig",
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
    const benchmark = b.addExecutable(.{ .name = "lagomath-benchmark", .root_module = b.createModule(.{
        .root_source_file = b.path(b.option([]const u8, "bench-source", "Benchmark adapter source") orelse "src/benchmark.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "bench_cases", .module = b.createModule(.{ .root_source_file = b.path(b.option([]const u8, "bench-cases", "Benchmark fixture module") orelse "benchmarks/cases/data.zig") }) }},
        .link_libc = true,
    }) });
    benchmark.root_module.addImport("lagomath", lagomath);
    const run_benchmark = b.addRunArtifact(benchmark);
    if (b.args) |args| run_benchmark.addArgs(args);
    b.step("bench", "Run shared JSON scalar benchmark cases").dependOn(&run_benchmark.step);
    const runner = b.addExecutable(.{ .name = "lagomath-cases", .root_module = b.createModule(.{
        .root_source_file = b.path(b.option([]const u8, "cases-source", "Numerical adapter source") orelse "src/case_runner.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "shared_cases", .module = b.createModule(.{ .root_source_file = b.path(b.option([]const u8, "cases-data", "Numerical fixture module") orelse "tests/cases/data.zig") }) }},
    }) });
    runner.root_module.addImport("lagomath", lagomath);
    const run_cases = b.addRunArtifact(runner);
    if (b.args) |args| run_cases.addArgs(args);
    b.step("cases", "Run shared numerical fixtures").dependOn(&run_cases.step);
    const verify_cases = b.addRunArtifact(runner);
    verify_cases.addArg("--quiet");
    test_step.dependOn(&verify_cases.step);
    verbose_test_step.dependOn(&verify_cases.step);
    const cabi = b.addLibrary(.{ .name = "lagomath_zig", .linkage = .dynamic, .root_module = b.createModule(.{
        .root_source_file = b.path("src/cabi.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.step("cabi", "Build the optional C ABI library").dependOn(&b.addInstallArtifact(cabi, .{}).step);
    const examples_step = b.step("examples", "Run all numbered Zig examples");
    inline for (.{ "ex01_TYPES", "ex02_VIEWS", "ex03_NDARRAY", "ex04_VECOPS", "ex05_MATOPS", "ex06_ROWOPS", "ex07_GAUSS", "ex08_LU", "ex09_CHOLESKY" }) |name| {
        const exe = b.addExecutable(.{ .name = name, .root_module = b.createModule(.{
            .root_source_file = b.path("examples/zig/" ++ name ++ ".zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "lagomath", .module = lagomath }},
        }) });
        const run = b.addRunArtifact(exe);
        examples_step.dependOn(&run.step);
        b.step("run-" ++ name, "Run " ++ name).dependOn(&run.step);
    }
}
