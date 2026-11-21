const std = @import("std");
const lm = @import("root.zig");
var jsonl = false;
const Timespec = extern struct { sec: isize, nsec: isize };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
fn now() u64 {
    var t: Timespec = undefined;
    std.debug.assert(clock_gettime(1, &t) == 0);
    return @as(u64, @intCast(t.sec)) * 1000000000 + @as(u64, @intCast(t.nsec));
}
const Case = struct { id: []const u8, operation: []const u8, dtype: []const u8, representation: []const u8, shape: []usize, warmup: usize, samples: usize, runs_per_sample: usize, expected_checksum: f64, input_family: []const u8 = "" };
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn kernel(comptime N: usize, comptime T: type, op: []const u8, a: *lm.Mat(N, N, T), b: *lm.Vec(N, T), out: *lm.Mat(N, N, T)) !T {
    @setEvalBranchQuota(100000);
    std.mem.doNotOptimizeAway(a);
    std.mem.doNotOptimizeAway(b);
    if (eq(op, "dot")) return lm.vecops.dotVec(N, T, b.*, b.*);
    if (eq(op, "normL2")) return lm.vecops.normL2Vec(N, T, b.*);
    if (eq(op, "axpy")) {
        const result = lm.vecops.axpyVec(N, T, 0.5, b.*, b.*);
        std.mem.doNotOptimizeAway(result);
        return result.vec[0];
    }
    if (eq(op, "matvec")) {
        const result = lm.matops.mulVecMat(N, N, T, a.*, b.*);
        std.mem.doNotOptimizeAway(result);
        return result.vec[0];
    }
    if (eq(op, "matmul")) {
        out.* = lm.matops.mulMatMat(N, N, N, T, a.*, a.*);
        std.mem.doNotOptimizeAway(out);
        return out.mat[0][0];
    }
    if (eq(op, "outer")) {
        out.* = lm.matops.outerMat(N, N, T, b.*, b.*);
        std.mem.doNotOptimizeAway(out);
        return out.get(0, 0);
    }
    if (eq(op, "outerAdd")) {
        out.* = lm.matops.outerAddMat(N, N, T, 0.5, b.*, b.*, a.*);
        std.mem.doNotOptimizeAway(out);
        return out.mat[0][0];
    }
    if (eq(op, "transpose")) {
        out.* = a.transpose();
        std.mem.doNotOptimizeAway(out);
        return out.mat[0][0];
    }
    if (eq(op, "rowops")) {
        out.* = a.*;
        lm.rowops.swapRowsMat(N, N, T, 0, N - 1, out);
        std.mem.doNotOptimizeAway(out);
        return out.mat[0][0];
    }
    if (eq(op, "gauss")) return (try lm.linsolve.solveGaussMat(N, T, a.*, b.*)).vec[0];
    if (eq(op, "lu")) return (try lm.linsolve.solveLUMat(N, T, a.*, b.*)).vec[0];
    if (eq(op, "cholesky")) return (try lm.linsolve.solveCholeskyMat(N, T, a.*, b.*)).vec[0];
    if (eq(op, "factorLU")) {
        const factor = try lm.linsolve.factorLUMat(N, T, a.*);
        std.mem.doNotOptimizeAway(factor);
        return factor.lu.get(0, 0);
    }
    if (eq(op, "factorCholesky")) {
        const factor = try lm.linsolve.factorCholeskyMat(N, T, a.*);
        std.mem.doNotOptimizeAway(factor);
        return factor.lower.get(0, 0);
    }
    if (comptime N >= 2 and N <= 4) {
        const ops = if (N == 2) lm.Mat22Ops else if (N == 3) lm.Mat33Ops else lm.Mat44Ops;
        if (eq(op, "det")) return ops.det(T, a.*);
        if (eq(op, "inv")) {
            out.* = ops.inv(T, a.*);
            std.mem.doNotOptimizeAway(out);
            return out.get(0, 0);
        }
    }
    return error.UnknownOperation;
}
fn fixed(comptime N: usize, comptime T: type, c: Case) !void {
    var a = lm.Mat(N, N, T).initFill(0.125);
    for (0..N) |i| a.set(i, i, @floatFromInt(N + 1));
    var b = lm.Vec(N, T).initOnes();
    var out = lm.Mat(N, N, T).initZeros();
    var batch: [16]lm.Vec(N, T) = undefined;
    for (&batch, 0..) |*rhs, j| rhs.* = b.mulScal(@floatFromInt(j + 1));
    if (eq(c.input_family, "pivot_required")) lm.rowops.swapRowsMat(N, N, T, 0, N - 1, &a);
    if (eq(c.input_family, "signed_alternating")) {
        for (0..N) |r| {
            b.set(r, if (r % 2 == 0) 1 else -1);
            for (0..N) |col| {
                if ((r + col) % 2 == 1) a.set(r, col, -a.get(r, col));
            }
        }
    }
    if (eq(c.input_family, "dense_random")) {
        for (0..N) |r| for (0..N) |col| {
            if (r != col) a.set(r, col, (@as(T, @floatFromInt((17 * r + 23 * col) % 29)) - 14) / 64);
        };
    }
    const reuse_lu = eq(c.operation, "luReuse") or eq(c.operation, "luBatch16");
    const reuse_chol = eq(c.operation, "choleskyReuse") or eq(c.operation, "choleskyBatch16");
    const lu: lm.linsolve.LUStack(N, T) = if (reuse_lu) try lm.linsolve.factorLUMat(N, T, a) else undefined;
    const chol: lm.linsolve.CholeskyStack(N, T) = if (reuse_chol) try lm.linsolve.factorCholeskyMat(N, T, a) else undefined;
    var checksum: T = 0;
    for (0..c.warmup) |_| {
        if (reuse_lu) {
            std.mem.doNotOptimizeAway(&batch);
            std.mem.doNotOptimizeAway(&lu);
            for (0..if (eq(c.operation, "luBatch16")) @as(usize, 16) else 1) |j| {
                const rb = batch[j];
                const result = try lm.linsolve.solveLUFactoredMat(N, T, lu, rb);
                std.mem.doNotOptimizeAway(result);
                checksum = result.get(0);
            }
        } else if (reuse_chol) {
            std.mem.doNotOptimizeAway(&batch);
            std.mem.doNotOptimizeAway(&chol);
            for (0..if (eq(c.operation, "choleskyBatch16")) @as(usize, 16) else 1) |j| {
                const rb = batch[j];
                const result = try lm.linsolve.solveCholeskyFactoredMat(N, T, chol, rb);
                std.mem.doNotOptimizeAway(result);
                checksum = result.get(0);
            }
        } else {
            checksum = try kernel(N, T, c.operation, &a, &b, &out);
        }
    }
    for (0..c.samples) |s| {
        const start = now();
        for (0..c.runs_per_sample) |_| {
            if (reuse_lu) {
                std.mem.doNotOptimizeAway(&batch);
                std.mem.doNotOptimizeAway(&lu);
                for (0..if (eq(c.operation, "luBatch16")) @as(usize, 16) else 1) |j| {
                    const rb = batch[j];
                    const result = try lm.linsolve.solveLUFactoredMat(N, T, lu, rb);
                    std.mem.doNotOptimizeAway(result);
                    checksum = result.get(0);
                }
            } else if (reuse_chol) {
                std.mem.doNotOptimizeAway(&batch);
                std.mem.doNotOptimizeAway(&chol);
                for (0..if (eq(c.operation, "choleskyBatch16")) @as(usize, 16) else 1) |j| {
                    const rb = batch[j];
                    const result = try lm.linsolve.solveCholeskyFactoredMat(N, T, chol, rb);
                    std.mem.doNotOptimizeAway(result);
                    checksum = result.get(0);
                }
            } else {
                checksum = try kernel(N, T, c.operation, &a, &b, &out);
            }
            std.mem.doNotOptimizeAway(checksum);
        }
        const elapsed = now() - start;
        if (!std.math.isFinite(checksum) or @abs(@as(f64, @floatCast(checksum)) - c.expected_checksum) > 1e-4 * @max(1, @abs(c.expected_checksum))) return error.InvalidChecksum;
        if (jsonl) std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"sample\":{d},\"ns\":{d},\"runs\":{d},\"checksum\":{d}}}\n", .{ c.id, s, elapsed, c.runs_per_sample, checksum });
    }
}
fn dynamic(comptime T: type, c: Case) !void {
    const batch = eq(c.operation, "dotBatch3");
    const count = c.shape[0] * (if (batch) @as(usize, 3) else 1);
    const a = try std.heap.page_allocator.alloc(T, count);
    defer std.heap.page_allocator.free(a);
    const b = try std.heap.page_allocator.alloc(T, count);
    defer std.heap.page_allocator.free(b);
    @memset(a, 1);
    @memset(b, 1);
    const v = lm.VecView(T).init(a);
    const w = lm.VecView(T).init(b);
    const out = try std.heap.page_allocator.alloc(T, a.len);
    defer std.heap.page_allocator.free(out);
    const perm = try std.heap.page_allocator.alloc(usize, a.len);
    defer std.heap.page_allocator.free(perm);
    for (perm, 0..) |*value, i| value.* = (17 * i) % a.len;
    for (0..c.samples + 1) |sample| {
        const runs = if (sample == 0) c.warmup else c.runs_per_sample;
        if (runs == 0) continue;
        var checksum: T = 0;
        const start = now();
        for (0..runs) |_| {
            std.mem.doNotOptimizeAway(a.ptr);
            std.mem.doNotOptimizeAway(b.ptr);
            const op = c.operation;
            if (eq(op, "dot")) checksum = lm.vecops.dotVecView(T, v, w) else if (eq(op, "sum")) checksum = lm.sliceops.sum(T, a) else if (eq(op, "mean")) checksum = lm.sliceops.mean(T, a) else if (eq(op, "min")) checksum = lm.sliceops.min(T, a).val else if (eq(op, "max")) checksum = lm.sliceops.max(T, a).val else if (eq(op, "normL1")) checksum = lm.vecops.normL1VecView(T, v) else if (eq(op, "normL2")) checksum = lm.vecops.normL2VecView(T, v) else if (eq(op, "normL2Square")) checksum = lm.vecops.normL2SquareVecView(T, v) else if (eq(op, "maxAbs")) checksum = lm.vecops.maxAbsVecView(T, v) else if (eq(op, "allFinite")) checksum = if (lm.vecops.allFiniteVecView(T, v)) 1 else 0 else {
                if (eq(op, "add")) lm.sliceops.add(T, a, b, out) else if (eq(op, "sub")) lm.sliceops.sub(T, a, b, out) else if (eq(op, "mul")) lm.sliceops.mul(T, a, b, out) else if (eq(op, "div")) lm.sliceops.div(T, a, b, out) else if (eq(op, "scale")) lm.sliceops.mulScal(T, a, 0.5, out) else if (eq(op, "fill")) @memset(out, 2) else if (eq(op, "permute")) lm.rowops.applyPermutationVecView(T, perm, v, lm.VecView(T).init(out)) else if (eq(op, "dotBatch3")) batchDotScalar(T, a, b, out[0..c.shape[0]]) else if (eq(op, "axpy")) {
                    @memcpy(out, b);
                    lm.vecops.axpyVecView(T, 0.5, v, lm.VecView(T).init(out));
                } else return error.UnknownOperation;
                std.mem.doNotOptimizeAway(out);
                checksum = out[0];
            }
            std.mem.doNotOptimizeAway(checksum);
        }
        const elapsed = now() - start;
        if (checksum != @as(T, @floatCast(c.expected_checksum))) return error.InvalidChecksum;
        if (sample > 0 and jsonl) std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"sample\":{d},\"ns\":{d},\"runs\":{d},\"checksum\":{d}}}\n", .{ c.id, sample - 1, elapsed, c.runs_per_sample, checksum });
    }
}
fn batchDotScalar(comptime T: type, a: []const T, b: []const T, out: []T) void {
    for (out, 0..) |*value, i| {
        var acc: T = 0;
        inline for (0..3) |k| acc += a[i * 3 + k] * b[i * 3 + k];
        value.* = acc;
    }
}
fn view(comptime T: type, c: Case) !void {
    const n = c.shape[0];
    const alloc = std.heap.page_allocator;
    const a = try alloc.alloc(T, n * n);
    defer alloc.free(a);
    const work = try alloc.alloc(T, n * n);
    defer alloc.free(work);
    const b = try alloc.alloc(T, n);
    defer alloc.free(b);
    const out = try alloc.alloc(T, n);
    defer alloc.free(out);
    const rhs = try alloc.alloc(T, n);
    defer alloc.free(rhs);
    const perm = try alloc.alloc(usize, n);
    defer alloc.free(perm);
    @memset(a, 0.125);
    @memset(b, 1);
    for (0..n) |i| a[i * n + i] = @floatFromInt(n + 1);
    if (eq(c.input_family, "pivot_required")) {
        for (0..n) |col| std.mem.swap(T, &a[col], &a[(n - 1) * n + col]);
    }
    if (eq(c.input_family, "signed_alternating")) {
        for (0..n) |r| {
            b[r] = if (r % 2 == 0) 1 else -1;
            for (0..n) |col| {
                if ((r + col) % 2 == 1) a[r * n + col] = -a[r * n + col];
            }
        }
    }
    if (eq(c.input_family, "dense_random")) {
        for (0..n) |r| for (0..n) |col| {
            if (r != col) a[r * n + col] = (@as(T, @floatFromInt((17 * r + 23 * col) % 29)) - 14) / 64;
        };
    }
    const a2 = try alloc.dupe(T, a);
    defer alloc.free(a2);
    const b2 = try alloc.dupe(T, b);
    defer alloc.free(b2);
    const matrix2 = lm.MatView(T).init(a2, n, n);
    const vector2 = lm.VecView(T).init(b2);
    var matrix = lm.MatView(T).init(a, n, n);
    var dest = lm.MatView(T).init(work, n, n);
    const vector = lm.VecView(T).init(b);
    const dst = lm.VecView(T).init(out);
    const scratch = lm.VecView(T).init(rhs);
    const op = c.operation;
    if (eq(op, "luReuse") or eq(op, "luBatch16")) try lm.linsolve.factorLUMatView(T, matrix, dest, perm);
    if (eq(op, "choleskyReuse") or eq(op, "choleskyBatch16")) try lm.linsolve.factorCholeskyMatView(T, matrix, dest);
    const batch = try alloc.alloc(T, 16 * n);
    defer alloc.free(batch);
    for (0..16) |j| @memset(batch[j * n ..][0..n], @as(T, @floatFromInt(j + 1)));
    var checksum: T = 0;
    for (0..c.samples + 1) |sample| {
        const runs = if (sample == 0) c.warmup else c.runs_per_sample;
        const start = now();
        for (0..runs) |_| {
            std.mem.doNotOptimizeAway(a.ptr);
            std.mem.doNotOptimizeAway(b.ptr);
            if (eq(op, "matvec")) {
                lm.matops.mulVecMatView(T, matrix, vector, dst);
                std.mem.doNotOptimizeAway(out);
                checksum = if (eq(op, "factorLU") or eq(op, "factorCholesky")) work[0] else out[0];
            } else if (eq(op, "matmul")) {
                lm.matops.mulMatMatView(T, matrix, matrix2, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "addMatrix")) {
                lm.MatViewOps(T).add(&matrix, &matrix2, &dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "subMatrix")) {
                lm.MatViewOps(T).sub(&matrix, &matrix2, &dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "mulMatrix")) {
                lm.MatViewOps(T).mulElemWise(&matrix, &matrix2, &dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "divMatrix")) {
                lm.MatViewOps(T).divElemWise(&matrix, &matrix2, &dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "scaleMatrix")) {
                lm.MatViewOps(T).mulScal(&matrix, 0.5, &dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "fillMatrix")) {
                dest.fill(2);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "matrixNorm")) {
                checksum = lm.matops.frobeniusNormMatView(T, matrix);
            } else if (eq(op, "matrixMaxAbs")) {
                checksum = lm.matops.maxAbsMatView(T, matrix);
            } else if (eq(op, "matrixAllFinite")) {
                checksum = if (lm.matops.allFiniteMatView(T, matrix)) @as(T, 1) else @as(T, 0);
            } else if (eq(op, "trace")) {
                checksum = matrix.trace();
            } else if (eq(op, "scaleRow")) {
                @memcpy(work, a);
                lm.rowops.scaleRowMatView(T, 0, 0.5, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "addScaledRow")) {
                @memcpy(work, a);
                lm.rowops.addScaledRowMatView(T, 0, n - 1, 0.5, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[(n - 1) * n];
            } else if (eq(op, "outer")) {
                lm.matops.outerMatView(T, vector, vector2, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "outerAdd")) {
                @memcpy(work, a);
                lm.matops.outerAddMatView(T, 0.5, vector, vector2, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "transpose")) {
                matrix.transpose(&dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "rowops")) {
                @memcpy(work, a);
                lm.rowops.swapRowsMatView(T, 0, n - 1, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "factorLU")) {
                try lm.linsolve.factorLUMatView(T, matrix, dest, perm);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else if (eq(op, "factorCholesky")) {
                try lm.linsolve.factorCholeskyMatView(T, matrix, dest);
                std.mem.doNotOptimizeAway(work);
                checksum = work[0];
            } else {
                if (eq(op, "luReuse") or eq(op, "choleskyReuse") or eq(op, "luBatch16") or eq(op, "choleskyBatch16")) {
                    std.mem.doNotOptimizeAway(work.ptr);
                    std.mem.doNotOptimizeAway(batch.ptr);
                    const count: usize = if (eq(op, "luBatch16") or eq(op, "choleskyBatch16")) 16 else 1;
                    for (0..count) |j| {
                        const rb = lm.VecView(T).init(batch[j * n ..][0..n]);
                        if (eq(op, "luReuse") or eq(op, "luBatch16")) try lm.linsolve.solveLUFactoredMatView(T, dest, perm, rb, scratch, dst) else try lm.linsolve.solveCholeskyFactoredMatView(T, dest, rb, scratch, dst);
                        std.mem.doNotOptimizeAway(out);
                    }
                } else if (eq(op, "gauss")) try lm.linsolve.solveGaussMatView(T, matrix, vector, dest, scratch, dst) else if (eq(op, "lu")) try lm.linsolve.solveLUMatView(T, matrix, vector, dest, perm, scratch, dst) else if (eq(op, "cholesky")) try lm.linsolve.solveCholeskyMatView(T, matrix, vector, dest, scratch, dst) else return error.UnknownOperation;
                std.mem.doNotOptimizeAway(out);
                checksum = out[0];
            }
        }
        const elapsed = now() - start;
        if (!std.math.isFinite(checksum) or @abs(@as(f64, @floatCast(checksum)) - c.expected_checksum) > 1e-4 * @max(1, @abs(c.expected_checksum))) return error.InvalidChecksum;
        if (sample > 0) if (jsonl) std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"sample\":{d},\"ns\":{d},\"runs\":{d},\"checksum\":{d}}}\n", .{ c.id, sample - 1, elapsed, c.runs_per_sample, checksum });
    }
}
fn ndarray(comptime T: type, c: Case) !void {
    const alloc = std.heap.page_allocator;
    var array = try lm.NDArray(T).initFlat(alloc, c.shape);
    defer alloc.free(array.slice);
    defer array.deinit(alloc);
    array.fill(1);
    const idx = try alloc.alloc(usize, c.shape.len);
    defer alloc.free(idx);
    for (idx, c.shape) |*v, dim| v.* = dim - 1;
    var prefix = try array.fixedPrefixView(alloc, idx[0..@min(1, idx.len - 1)]);
    defer prefix.deinit(alloc);
    var checksum: T = 0;
    for (0..c.samples + 1) |sample| {
        const runs = if (sample == 0) c.warmup else c.runs_per_sample;
        const start = now();
        for (0..runs) |_| {
            std.mem.doNotOptimizeAway(idx.ptr);
            std.mem.doNotOptimizeAway(array.slice.ptr);
            const op = c.operation;
            if (eq(op, "flatWrite")) {
                array.setFlat(array.slice.len - 1, 2);
                std.mem.doNotOptimizeAway(array.slice);
                checksum = array.getFlat(array.slice.len - 1);
            } else if (eq(op, "indexedWrite")) {
                array.set(idx, 2);
                std.mem.doNotOptimizeAway(array.slice);
                checksum = array.get(idx);
            } else if (eq(op, "getFlatIdx")) checksum = @floatFromInt(array.getFlatIdx(idx)) else if (eq(op, "flatRead")) checksum = array.slice[array.slice.len - 1] else if (eq(op, "indexedRead")) checksum = array.get(idx) else if (eq(op, "offset2")) checksum = @floatFromInt(array.offset2(idx[0], idx[1])) else if (eq(op, "offset3")) checksum = @floatFromInt(array.offset3(idx[0], idx[1], idx[2])) else if (eq(op, "offset4")) checksum = @floatFromInt(array.offset4(idx[0], idx[1], idx[2], idx[3])) else if (eq(op, "planeBase")) checksum = @floatFromInt(array.planeBase(idx[0])) else if (eq(op, "subBase2")) checksum = @floatFromInt(array.subBase2(idx[0], idx[1])) else if (eq(op, "fixedPrefixView")) checksum = prefix.getFlat(0) else if (eq(op, "stream")) {
                checksum = 0;
                for (array.slice) |v| checksum += v;
            } else return error.UnknownOperation;
            std.mem.doNotOptimizeAway(checksum);
        }
        const elapsed = now() - start;
        if (@as(f64, @floatCast(checksum)) != c.expected_checksum) return error.InvalidChecksum;
        if (sample > 0) if (jsonl) std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"sample\":{d},\"ns\":{d},\"runs\":{d},\"checksum\":{d}}}\n", .{ c.id, sample - 1, elapsed, c.runs_per_sample, checksum });
    }
}
fn rectangular(comptime R: usize, comptime C: usize, comptime P: usize, comptime T: type, c: Case) !void {
    var a = lm.Mat(R, C, T).initOnes();
    var b = lm.Mat(C, P, T).initOnes();
    var out = lm.Mat(R, P, T).initZeros();
    var trans = lm.Mat(C, R, T).initZeros();
    var rows = a;
    var x = lm.Vec(C, T).initOnes();
    var y = lm.Vec(R, T).initZeros();
    const av = lm.MatView(T).init(@as([*]T, @ptrCast(&a.mat))[0 .. R * C], R, C);
    const bv = lm.MatView(T).init(@as([*]T, @ptrCast(&b.mat))[0 .. C * P], C, P);
    const ov = lm.MatView(T).init(@as([*]T, @ptrCast(&out.mat))[0 .. R * P], R, P);
    var tv = lm.MatView(T).init(@as([*]T, @ptrCast(&trans.mat))[0 .. C * R], C, R);
    const rv = lm.MatView(T).init(@as([*]T, @ptrCast(&rows.mat))[0 .. R * C], R, C);
    var checksum: T = 0;
    for (0..c.samples + 1) |sample| {
        const count = if (sample == 0) c.warmup else c.runs_per_sample;
        const start = now();
        for (0..count) |_| {
            std.mem.doNotOptimizeAway(&a);
            std.mem.doNotOptimizeAway(&b);
            std.mem.doNotOptimizeAway(&x);
            if (eq(c.operation, "matmulRect")) {
                if (eq(c.representation, "Mat")) out = lm.matops.mulMatMat(R, C, P, T, a, b) else lm.matops.mulMatMatView(T, av, bv, ov);
                std.mem.doNotOptimizeAway(&out);
                checksum = out.get(0, 0);
            } else if (eq(c.operation, "matvecRect")) {
                if (eq(c.representation, "Mat")) y = lm.matops.mulVecMat(R, C, T, a, x) else lm.matops.mulVecMatView(T, av, lm.VecView(T).init(&x.vec), lm.VecView(T).init(&y.vec));
                std.mem.doNotOptimizeAway(&y);
                checksum = y.get(0);
            } else if (eq(c.operation, "transposeRect")) {
                if (eq(c.representation, "Mat")) trans = a.transpose() else av.transpose(&tv);
                std.mem.doNotOptimizeAway(&trans);
                checksum = trans.get(0, 0);
            } else {
                rows = a;
                if (eq(c.representation, "Mat")) lm.rowops.swapRowsMat(R, C, T, 0, R - 1, &rows) else lm.rowops.swapRowsMatView(T, 0, R - 1, rv);
                std.mem.doNotOptimizeAway(&rows);
                checksum = rows.get(0, 0);
            }
            std.mem.doNotOptimizeAway(checksum);
        }
        const elapsed = now() - start;
        if (checksum != @as(T, @floatCast(c.expected_checksum))) return error.InvalidChecksum;
        if (sample > 0) if (jsonl) std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"sample\":{d},\"ns\":{d},\"runs\":{d},\"checksum\":{d}}}\n", .{ c.id, sample - 1, elapsed, c.runs_per_sample, checksum });
    }
}
fn rectangularView(comptime T: type, c: Case) !void {
    const r = c.shape[0];
    const k = c.shape[1];
    const p = c.shape[2];
    const alloc = std.heap.page_allocator;
    const a = try alloc.alloc(T, r * k);
    defer alloc.free(a);
    const b = try alloc.alloc(T, k * p);
    defer alloc.free(b);
    const out = try alloc.alloc(T, r * p);
    defer alloc.free(out);
    @memset(a, 1);
    @memset(b, 1);
    const av = lm.MatView(T).init(a, r, k);
    const bv = lm.MatView(T).init(b, k, p);
    const ov = lm.MatView(T).init(out, r, p);
    var checksum: T = 0;
    for (0..c.samples + 1) |sample| {
        const count = if (sample == 0) c.warmup else c.runs_per_sample;
        const start = now();
        for (0..count) |_| {
            std.mem.doNotOptimizeAway(a.ptr);
            std.mem.doNotOptimizeAway(b.ptr);
            lm.matops.mulMatMatView(T, av, bv, ov);
            std.mem.doNotOptimizeAway(out);
            checksum = out[0];
            std.mem.doNotOptimizeAway(checksum);
        }
        const elapsed = now() - start;
        if (!std.math.isFinite(checksum) or @abs(@as(f64, @floatCast(checksum)) - c.expected_checksum) > 1e-4 * @max(1, @abs(c.expected_checksum))) return error.InvalidChecksum;
        if (sample > 0 and jsonl) std.debug.print("{{\"id\":\"{s}\",\"backend\":\"zig\",\"sample\":{d},\"ns\":{d},\"runs\":{d},\"checksum\":{d}}}\n", .{ c.id, sample - 1, elapsed, c.runs_per_sample, checksum });
    }
}
fn dispatch(comptime T: type, c: Case) !void {
    if (eq(c.operation, "matmulRect") and eq(c.representation, "MatView")) return rectangularView(T, c);
    if (std.mem.endsWith(u8, c.operation, "Rect")) {
        inline for (.{ .{ 2, 3, 4 }, .{ 3, 2, 3 }, .{ 4, 8, 3 }, .{ 6, 12, 8 }, .{ 12, 6, 3 } }) |dims| {
            if (c.shape[0] == dims[0] and c.shape[1] == dims[1] and c.shape[2] == dims[2]) return rectangular(dims[0], dims[1], dims[2], T, c);
        }
        return error.UnsupportedShape;
    }
    if (eq(c.representation, "NDArray")) return ndarray(T, c);
    if (eq(c.representation, "MatView")) return view(T, c);
    if (eq(c.representation, "VecView") or eq(c.representation, "slice")) return dynamic(T, c);
    inline for (.{ 2, 3, 4, 6, 8, 12, 16 }) |n| {
        if (c.shape[0] == n) return fixed(n, T, c);
    }
    return error.UnsupportedShape;
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| {
        if (eq(arg, "--jsonl")) jsonl = true else return error.InvalidArgument;
    }

    const parsed = try std.json.parseFromSlice(struct { cases: []Case }, std.heap.page_allocator, @import("bench_cases").data, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.cases) |c| {
        if (eq(c.dtype, "f32")) try dispatch(f32, c) else try dispatch(f64, c);
    }
    if (!jsonl) std.debug.print("{d} Zig benchmark workloads passed; use --jsonl for timings\n", .{parsed.value.cases.len});
}
