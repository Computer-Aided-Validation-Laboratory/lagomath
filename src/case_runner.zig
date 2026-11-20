const std = @import("std");
const lm = @import("root.zig");
const Inputs = struct { a: []std.json.Value, b: []std.json.Value = &.{}, x: []std.json.Value = &.{}, shape: []usize, index: []usize = &.{}, alpha: f64 = 0, mutate_index: usize = 0, mutate_value: f64 = 0, view_start: usize = 0, row_a: usize = 0, row_b: usize = 0 };
const Case = struct { id: []const u8, suite: []const u8, operation: []const u8, dtype: []const u8, representation: []const u8, inputs: Inputs, expected: ?[]std.json.Value = null, expected_error: ?[]const u8 = null, tolerance: struct { atol: f64 = 0, rtol: f64 = 0 } = .{} };
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn values(comptime T: type, raw: anytype) ![]T {
    const out = try std.heap.page_allocator.alloc(T, raw.len);
    for (raw, out) |a, *b| {
        if (comptime @TypeOf(a) == std.json.Value) {
            b.* = switch (a) {
                .float => |v| @floatCast(v),
                .integer => |v| @floatFromInt(v),
                .object => |v| if (eq(v.get("special").?.string, "nan")) std.math.nan(T) else if (eq(v.get("special").?.string, "-inf")) -std.math.inf(T) else if (eq(v.get("special").?.string, "inf")) std.math.inf(T) else return error.InvalidNumber,
                else => return error.InvalidNumber,
            };
        } else b.* = @floatCast(a);
    }
    return out;
}
fn scalar(comptime T: type, v: T) ![]T {
    const out = try std.heap.page_allocator.alloc(T, 1);
    out[0] = v;
    return out;
}
fn vectorFixed(comptime N: usize, comptime T: type, c: Case, a: []T, b: []T) ![]T {
    if (eq(c.operation, "valueCopy")) {
        const original = lm.Vec(N, T).initSlice(a);
        var copy = original;
        copy.set(c.inputs.mutate_index, @floatCast(c.inputs.mutate_value));
        const result = try std.heap.page_allocator.alloc(T, N * 2);
        @memcpy(result[0..N], &original.vec);
        @memcpy(result[N..], &copy.vec);
        return result;
    }
    const v = lm.Vec(N, T).initSlice(a);
    const w = lm.Vec(N, T).initSlice(b);
    const op = c.operation;
    if (eq(op, "dot")) return scalar(T, lm.vecops.dotVec(N, T, v, w));
    if (eq(op, "normL1")) return scalar(T, v.normL1());
    if (eq(op, "normL2")) return scalar(T, v.normL2());
    if (eq(op, "normL2Square")) return scalar(T, v.normL2Square());
    if (eq(op, "maxAbs")) return scalar(T, lm.vecops.maxAbsVec(N, T, v));
    if (eq(op, "allFinite")) return scalar(T, if (lm.vecops.allFiniteVec(N, T, v)) 1 else 0);
    if (eq(op, "sum")) return scalar(T, v.sum());
    if (eq(op, "mean")) return scalar(T, v.mean());
    if (eq(op, "min")) return scalar(T, v.min().val);
    if (eq(op, "max")) return scalar(T, v.max().val);
    const result = if (eq(op, "add")) v.add(w) else if (eq(op, "sub")) v.sub(w) else if (eq(op, "axpy")) lm.vecops.axpyVec(N, T, @floatCast(c.inputs.alpha), v, w) else return error.UnknownOperation;
    return values(T, &result.vec);
}
fn matrixFixed(comptime R: usize, comptime C: usize, comptime P: usize, comptime T: type, c: Case, a: []T, b: []T) ![]T {
    @setEvalBranchQuota(100000);
    const matrix = lm.Mat(R, C, T).initSlice(a);
    const op = c.operation;
    if (eq(op, "valueCopy")) {
        var copy = matrix;
        copy.set(c.inputs.mutate_index / C, c.inputs.mutate_index % C, @floatCast(c.inputs.mutate_value));
        const out = try std.heap.page_allocator.alloc(T, R * C * 2);
        for (0..R) |r| for (0..C) |cidx| {
            out[r * C + cidx] = matrix.get(r, cidx);
            out[R * C + r * C + cidx] = copy.get(r, cidx);
        };
        return out;
    }

    if (eq(op, "trace")) return scalar(T, matrix.trace());
    if (eq(op, "frobeniusNorm")) return scalar(T, lm.matops.frobeniusNormMat(R, C, T, matrix));
    if (eq(op, "maxAbs")) return scalar(T, lm.matops.maxAbsMat(R, C, T, matrix));
    if (eq(op, "allFinite")) return scalar(T, if (lm.matops.allFiniteMat(R, C, T, matrix)) 1 else 0);
    if (eq(op, "det") or eq(op, "inv") or eq(op, "inverseIdentity")) {
        if (comptime R == C and (R == 2 or R == 3 or R == 4)) {
            const ops = if (R == 2) lm.mattype.Mat22Ops else if (R == 3) lm.mattype.Mat33Ops else lm.mattype.Mat44Ops;
            if (eq(op, "det")) return scalar(T, ops.det(T, matrix));
            var result = try ops.invChecked(T, matrix, 0);
            if (eq(op, "inverseIdentity")) result = lm.matops.mulMatMat(R, C, R, T, matrix, result);
            const out = try std.heap.page_allocator.alloc(T, R * C);
            for (0..R) |r| for (0..C) |col| {
                out[r * C + col] = result.get(r, col);
            };
            return out;
        } else return error.UnsupportedShape;
    }
    if (eq(op, "outer") or eq(op, "outerAdd")) {
        const x = try values(T, c.inputs.x);
        defer std.heap.page_allocator.free(x);
        const vx = lm.Vec(R, T).initSlice(x);
        const vy = lm.Vec(C, T).initSlice(b);
        const result = if (eq(op, "outer")) lm.matops.outerMat(R, C, T, vx, vy) else lm.matops.outerAddMat(R, C, T, @floatCast(c.inputs.alpha), vx, vy, matrix);
        const out = try std.heap.page_allocator.alloc(T, R * C);
        for (0..R) |r| for (0..C) |col| {
            out[r * C + col] = result.get(r, col);
        };
        return out;
    }
    const count = if (eq(op, "matmul")) R * P else if (eq(op, "matvec")) R else R * C;
    const out = try std.heap.page_allocator.alloc(T, count);
    if (eq(op, "matmul")) {
        const result = lm.matops.mulMatMat(R, C, P, T, matrix, lm.Mat(C, P, T).initSlice(b));
        for (0..R) |r| for (0..P) |col| {
            out[r * P + col] = result.get(r, col);
        };
    } else if (eq(op, "matvec")) {
        const x = try values(T, c.inputs.x);
        defer std.heap.page_allocator.free(x);
        const result = lm.matops.mulVecMat(R, C, T, matrix, lm.Vec(C, T).initSlice(x));
        @memcpy(out, &result.vec);
    } else if (eq(op, "transpose")) {
        const result = matrix.transpose();
        for (0..C) |r| for (0..R) |col| {
            out[r * R + col] = result.get(r, col);
        };
    } else return error.UnknownOperation;
    return out;
}
fn solveFixed(comptime N: usize, comptime T: type, c: Case, a: []T, b: []T) ![]T {
    const matrix = lm.Mat(N, N, T).initSlice(a);
    const rhs = lm.Vec(N, T).initSlice(b);
    const op = c.operation;
    var result: lm.Vec(N, T) = undefined;
    if (eq(op, "gauss")) result = try lm.linsolve.solveGaussMatChecked(N, T, matrix, rhs) else if (eq(op, "lu")) result = try lm.linsolve.solveLUMatChecked(N, T, matrix, rhs) else if (eq(op, "cholesky")) result = try lm.linsolve.solveCholeskyMatChecked(N, T, matrix, rhs) else if (eq(op, "luReuse")) {
        const factor = try lm.linsolve.factorLUMatChecked(N, T, matrix);
        result = try lm.linsolve.solveLUFactoredMatChecked(N, T, factor, rhs);
    } else if (eq(op, "choleskyReuse")) {
        const factor = try lm.linsolve.factorCholeskyMatChecked(N, T, matrix);
        result = try lm.linsolve.solveCholeskyFactoredMatChecked(N, T, factor, rhs);
    } else return error.UnknownOperation;
    return values(T, &result.vec);
}
fn run(comptime T: type, c: Case) ![]T {
    const a = try values(T, c.inputs.a);
    defer std.heap.page_allocator.free(a);
    const b = try values(T, c.inputs.b);
    defer std.heap.page_allocator.free(b);
    const op = c.operation;
    const shape = c.inputs.shape;
    if (eq(c.suite, "extended")) return @import("extended_cases.zig").run(T, c, a, b);
    if (eq(c.suite, "types")) {
        if (eq(c.representation, "Vec")) return vectorFixed(3, T, c, a, b);
        if (eq(c.representation, "Mat")) return matrixFixed(2, 2, 2, T, c, a, b);
        if (eq(op, "viewMutation")) {
            const view_value = lm.VecView(T).init(a[c.inputs.view_start .. c.inputs.view_start + shape[0]]);
            view_value.set(c.inputs.mutate_index, @floatCast(c.inputs.mutate_value));
            return values(T, a);
        }
        return error.UnknownOperation;
    }
    if (eq(c.suite, "vecops")) {
        if (eq(c.representation, "Vec")) {
            inline for (.{ 1, 2, 3, 4, 6, 8, 12, 16 }) |n| {
                if (shape[0] == n) return vectorFixed(n, T, c, a, b);
            }
            return error.UnsupportedShape;
        }
        if (eq(c.representation, "slice")) {
            if (eq(op, "dot")) return scalar(T, lm.sliceops.dot(T, a, b));
            if (eq(op, "normL1")) return scalar(T, lm.sliceops.normL1(T, a));
            if (eq(op, "normL2")) return scalar(T, lm.sliceops.normL2(T, a));
            if (eq(op, "normL2Square")) return scalar(T, lm.sliceops.normL2Square(T, a));
            if (eq(op, "sum")) return scalar(T, lm.sliceops.sum(T, a));
            if (eq(op, "mean")) return scalar(T, lm.sliceops.mean(T, a));
            if (eq(op, "min")) return scalar(T, lm.sliceops.min(T, a).val);
            if (eq(op, "max")) return scalar(T, lm.sliceops.max(T, a).val);
        }
        const v = lm.VecView(T).init(a);
        const w = lm.VecView(T).init(b);
        if (eq(op, "dot")) return scalar(T, lm.vecops.dotVecView(T, v, w));
        if (eq(op, "normL1")) return scalar(T, v.normL1());
        if (eq(op, "normL2")) return scalar(T, v.normL2());
        if (eq(op, "normL2Square")) return scalar(T, v.normL2Square());
        if (eq(op, "maxAbs")) return scalar(T, lm.vecops.maxAbsVecView(T, v));
        if (eq(op, "allFinite")) return scalar(T, if (lm.vecops.allFiniteVecView(T, v)) 1 else 0);
        if (eq(op, "sum")) return scalar(T, v.sum());
        if (eq(op, "mean")) return scalar(T, v.mean());
        if (eq(op, "min")) return scalar(T, v.min().val);
        if (eq(op, "max")) return scalar(T, v.max().val);
        const out = try std.heap.page_allocator.alloc(T, a.len);
        if (eq(op, "axpy")) {
            @memcpy(out, b);
            lm.vecops.axpyVecView(T, @floatCast(c.inputs.alpha), v, lm.VecView(T).init(out));
        } else if (eq(c.representation, "VecView")) {
            @memcpy(out, a);
            const destination = lm.VecView(T).init(out);
            if (eq(op, "add")) destination.addInPlace(&w) else if (eq(op, "sub")) destination.subInPlace(&w) else if (eq(op, "mul")) destination.mulInPlace(&w) else if (eq(op, "div")) destination.divInPlace(&w) else return error.UnknownOperation;
        } else if (eq(op, "add")) lm.sliceops.add(T, a, b, out) else if (eq(op, "sub")) lm.sliceops.sub(T, a, b, out) else if (eq(op, "mul")) lm.sliceops.mul(T, a, b, out) else if (eq(op, "div")) lm.sliceops.div(T, a, b, out) else return error.UnknownOperation;
        return out;
    }
    if (eq(c.suite, "ndarray")) {
        var array = try lm.NDArray(T).init(std.heap.page_allocator, a, shape);
        defer array.deinit(std.heap.page_allocator);
        const idx = c.inputs.index;
        if (eq(op, "getFlatIdx")) return scalar(T, @floatFromInt(array.getFlatIdx(idx)));
        if (eq(op, "planeBase")) return scalar(T, @floatFromInt(array.planeBase(idx[0])));
        if (eq(op, "subBase2")) return scalar(T, @floatFromInt(array.subBase2(idx[0], idx[1])));
        if (eq(op, "offset2")) return scalar(T, @floatFromInt(array.offset2(idx[0], idx[1])));
        if (eq(op, "offset3")) return scalar(T, @floatFromInt(array.offset3(idx[0], idx[1], idx[2])));
        if (eq(op, "offset4")) return scalar(T, @floatFromInt(array.offset4(idx[0], idx[1], idx[2], idx[3])));
        if (eq(op, "calcFlatStride")) {
            const out = try std.heap.page_allocator.alloc(T, shape.len);
            for (out, 0..) |*v, i| v.* = @floatFromInt(array.calcFlatStride(i));
            return out;
        }
        if (eq(op, "getPlaneSlice")) return values(T, array.getPlaneSlice(idx[0]));
        if (eq(op, "fixedPrefixView")) {
            var view = try array.fixedPrefixView(std.heap.page_allocator, idx[0..1]);
            defer view.deinit(std.heap.page_allocator);
            return values(T, view.slice);
        }
        return error.UnknownOperation;
    }
    if (eq(c.suite, "rowops")) {
        const matrix = lm.MatView(T).init(a, shape[0], shape[1]);
        if (eq(op, "swapRows")) lm.rowops.swapRowsMatView(T, c.inputs.row_a, c.inputs.row_b, matrix) else if (eq(op, "scaleRow")) lm.rowops.scaleRowMatView(T, c.inputs.row_a, @floatCast(c.inputs.alpha), matrix) else if (eq(op, "addScaledRow")) lm.rowops.addScaledRowMatView(T, c.inputs.row_a, c.inputs.row_b, @floatCast(c.inputs.alpha), matrix) else return error.UnknownOperation;
        return values(T, a);
    }
    if (eq(c.representation, "Mat")) {
        if (eq(c.suite, "linsolve")) {
            inline for (.{ 1, 2, 3, 4, 6, 8, 12 }) |n| {
                if (shape[0] == n) return solveFixed(n, T, c, a, b);
            }
            return error.UnsupportedShape;
        }
        inline for (.{ .{ 1, 1, 1 }, .{ 2, 2, 2 }, .{ 3, 3, 3 }, .{ 4, 4, 4 }, .{ 6, 6, 6 }, .{ 8, 8, 8 }, .{ 12, 12, 12 }, .{ 2, 3, 4 }, .{ 6, 12, 8 }, .{ 12, 6, 3 } }) |dims| {
            if (shape[0] == dims[0] and shape[1] == dims[1] and shape[2] == dims[2]) return matrixFixed(dims[0], dims[1], dims[2], T, c, a, b);
        }
        return error.UnsupportedShape;
    }
    const matrix = lm.MatView(T).init(a, shape[0], shape[1]);
    if (eq(c.suite, "matops")) {
        if (eq(op, "outer") or eq(op, "outerAdd")) {
            const x = try values(T, c.inputs.x);
            defer std.heap.page_allocator.free(x);
            if (eq(op, "outer")) lm.matops.outerMatView(T, lm.VecView(T).init(x), lm.VecView(T).init(b), matrix) else lm.matops.outerAddMatView(T, @floatCast(c.inputs.alpha), lm.VecView(T).init(x), lm.VecView(T).init(b), matrix);
            return values(T, a);
        }

        if (eq(op, "trace")) return scalar(T, matrix.trace());
        if (eq(op, "frobeniusNorm")) return scalar(T, lm.matops.frobeniusNormMatView(T, matrix));
        if (eq(op, "maxAbs")) return scalar(T, lm.matops.maxAbsMatView(T, matrix));
        if (eq(op, "allFinite")) return scalar(T, if (lm.matops.allFiniteMatView(T, matrix)) 1 else 0);
        const count = if (eq(op, "matmul")) shape[0] * shape[2] else if (eq(op, "matvec")) shape[0] else a.len;
        const out = try std.heap.page_allocator.alloc(T, count);
        if (eq(op, "matmul")) lm.matops.mulMatMatView(T, matrix, lm.MatView(T).init(b, shape[1], shape[2]), lm.MatView(T).init(out, shape[0], shape[2])) else if (eq(op, "matvec")) {
            const x = try values(T, c.inputs.x);
            defer std.heap.page_allocator.free(x);
            lm.matops.mulVecMatView(T, matrix, lm.VecView(T).init(x), lm.VecView(T).init(out));
        } else if (eq(op, "transpose")) {
            var dest = lm.MatView(T).init(out, shape[1], shape[0]);
            matrix.transpose(&dest);
        } else return error.UnknownOperation;
        return out;
    }
    if (eq(c.suite, "linsolve")) {
        const n = shape[0];
        const work = try std.heap.page_allocator.alloc(T, n * n);
        defer std.heap.page_allocator.free(work);
        const scratch = try std.heap.page_allocator.alloc(T, n);
        defer std.heap.page_allocator.free(scratch);
        const perm = try std.heap.page_allocator.alloc(usize, n);
        defer std.heap.page_allocator.free(perm);
        const out = try std.heap.page_allocator.alloc(T, n);
        errdefer std.heap.page_allocator.free(out);
        const wm = lm.MatView(T).init(work, n, n);
        const rhs = lm.VecView(T).init(b);
        const wv = lm.VecView(T).init(scratch);
        const dst = lm.VecView(T).init(out);
        if (eq(op, "gauss")) try lm.linsolve.solveGaussMatViewChecked(T, matrix, rhs, wm, wv, dst) else if (eq(op, "lu")) try lm.linsolve.solveLUMatViewChecked(T, matrix, rhs, wm, perm, wv, dst) else if (eq(op, "cholesky")) try lm.linsolve.solveCholeskyMatViewChecked(T, matrix, rhs, wm, wv, dst) else if (eq(op, "luReuse")) {
            try lm.linsolve.factorLUMatViewChecked(T, matrix, wm, perm);
            try lm.linsolve.solveLUFactoredMatViewChecked(T, wm, perm, rhs, wv, dst);
        } else if (eq(op, "choleskyReuse")) {
            try lm.linsolve.factorCholeskyMatViewChecked(T, matrix, wm);
            try lm.linsolve.solveCholeskyFactoredMatViewChecked(T, wm, rhs, wv, dst);
        } else return error.UnknownOperation;
        return out;
    }
    return error.UnknownSuite;
}
fn emit(comptime T: type, c: Case, quiet: bool) !void {
    const result = run(T, c) catch |err| {
        if (c.expected_error == null or !eq(c.expected_error.?, @errorName(err))) {
            std.debug.print("Unexpected error for {s}: {s}\n", .{ c.id, @errorName(err) });
            return error.UnexpectedNumericalError;
        }
        if (quiet) return;
        std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"representation\":\"{s}\",\"status\":\"error\",\"error\":\"{s}\"}}\n", .{ c.id, c.representation, @errorName(err) });
        return;
    };
    defer std.heap.page_allocator.free(result);
    if (c.expected_error != null or c.expected == null) return error.ExpectedNumericalError;
    const expected = try values(f64, c.expected.?);
    defer std.heap.page_allocator.free(expected);
    if (result.len != expected.len) return error.ResultLengthMismatch;
    for (result, expected) |actual, want| {
        const value: f64 = @floatCast(actual);
        if (!(if (std.math.isNan(want)) std.math.isNan(value) else if (std.math.isInf(want)) value == want else std.math.isFinite(value) and @abs(value - want) <= c.tolerance.atol + c.tolerance.rtol * @abs(want))) {
            std.debug.print("Numerical mismatch for {s}: {d} vs {d}\n", .{ c.id, value, want });
            return error.NumericalMismatch;
        }
    }
    if (quiet) return;
    std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"representation\":\"{s}\",\"status\":\"ok\",\"result\":[", .{ c.id, c.representation });
    for (result, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        if (std.math.isNan(v)) std.debug.print("{{\"special\":\"nan\"}}", .{}) else if (std.math.isInf(v)) std.debug.print("{{\"special\":\"{s}inf\"}}", .{if (v < 0) "-" else ""}) else std.debug.print("{d}", .{v});
    }
    std.debug.print("]}}\n", .{});
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var quiet = true;
    var silent = false;
    for (args[1..]) |arg| {
        if (eq(arg, "--quiet")) {
            quiet = true;
            silent = true;
        } else if (eq(arg, "--jsonl")) quiet = false else return error.InvalidArgument;
    }

    const parsed = try std.json.parseFromSlice(struct { cases: []Case }, std.heap.page_allocator, @import("shared_cases").data, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.cases) |c| {
        if (eq(c.dtype, "f32")) try emit(f32, c, quiet) else try emit(f64, c, quiet);
    }
    if (quiet and !silent) std.debug.print("{d} shared Zig cases passed\n", .{parsed.value.cases.len});
}

/// Named historical unit tests execute shared cases through the same adapter.
/// The Python/native Mojo runners execute these identical IDs in the full corpus.
pub fn verifyLegacy(id: []const u8) !void {
    const registry = try std.json.parseFromSlice(struct { tests: []struct { id: []const u8, case_ids: []const []const u8 } }, std.testing.allocator, @import("shared_cases").legacy_registry, .{ .ignore_unknown_fields = true });
    defer registry.deinit();
    const parsed = try std.json.parseFromSlice(struct { cases: []Case }, std.testing.allocator, @import("shared_cases").legacy_cases, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (registry.value.tests) |entry| {
        if (!eq(entry.id, id)) continue;
        if (entry.case_ids.len == 0) return error.EmptyMigrationGroup;
        for (entry.case_ids) |case_id| {
            var found = false;
            for (parsed.value.cases) |c| {
                if (!eq(c.id, case_id)) continue;
                if (eq(c.dtype, "f32")) try emit(f32, c, true) else try emit(f64, c, true);
                found = true;
                break;
            }
            if (!found) return error.MissingMigrationCase;
        }
        return;
    }
    return error.UnmappedHistoricalTest;
}
