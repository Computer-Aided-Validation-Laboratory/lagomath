//! Execution adapters for shared method/property cases. No numerical truth here.
const std = @import("std");
const lm = @import("root.zig");
const alloc = std.heap.page_allocator;
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn copy(comptime T: type, a: []const T) ![]T {
    return alloc.dupe(T, a);
}
fn scalar(comptime T: type, a: T) ![]T {
    const out = try alloc.alloc(T, 1);
    out[0] = a;
    return out;
}
fn vector(comptime N: usize, comptime T: type, c: anytype, a: []T, b: []T) ![]T {
    var v = lm.Vec(N, T).initSlice(a);
    const op = c.operation;
    if (eq(op, "zeros")) v = lm.Vec(N, T).initZeros() else if (eq(op, "ones")) v = lm.Vec(N, T).initOnes() else if (eq(op, "fill")) v = lm.Vec(N, T).initFill(@floatCast(c.inputs.alpha)) else if (eq(op, "scale")) v = v.mulScal(@floatCast(c.inputs.alpha)) else if (eq(op, "applySqrt")) v = v.apply(std.math.sqrt) else if (eq(op, "applyAtan")) v = v.apply(std.math.atan) else if (eq(op, "applyExp")) v = v.apply(lm.sliceops.exp) else if (eq(op, "legacyNorms")) return copy(T, &.{ v.norm(), v.squaredNorm(), v.vecLen() }) else if (eq(op, "extremaIndices")) return copy(T, &.{ v.max().val, @floatFromInt(v.max().idx), v.min().val, @floatFromInt(v.min().idx) }) else if (eq(op, "coordinates")) {
        if (comptime N >= 4) return copy(T, &.{ v.x(), v.y(), v.z(), v.w() }) else return error.UnsupportedShape;
    } else if (eq(op, "cross")) {
        if (comptime N == 3) v = lm.Vec3Ops.cross(T, v, lm.Vec(3, T).initSlice(b)) else return error.UnsupportedShape;
    } else if (eq(op, "initVec2")) {
        if (comptime N == 2) v = lm.vectype.initVec2(T, a[0], a[1]) else return error.UnsupportedShape;
    } else if (eq(op, "initVec3")) {
        if (comptime N == 3) v = lm.vectype.initVec3(T, a[0], a[1], a[2]) else return error.UnsupportedShape;
    } else return error.UnknownOperation;
    return copy(T, &v.vec);
}
fn matrixFixed(comptime R: usize, comptime C: usize, comptime T: type, c: anytype, a: []T, b: []T) ![]T {
    var m = lm.Mat(R, C, T).initSlice(a);
    const op = c.operation;
    if (eq(op, "swapRowsMethod") or eq(op, "scaleRowMethod") or eq(op, "addScaledRowMethod")) {
        if (eq(op, "swapRowsMethod")) lm.rowops.swapRowsMat(R, C, T, c.inputs.index[0], c.inputs.index[1], &m) else if (eq(op, "scaleRowMethod")) lm.rowops.scaleRowMat(R, C, T, c.inputs.index[0], @floatCast(c.inputs.alpha), &m) else lm.rowops.addScaledRowMat(R, C, T, c.inputs.index[0], c.inputs.index[1], @floatCast(c.inputs.alpha), &m);
        return copy(T, @as([*]const T, @ptrCast(&m.mat))[0 .. R * C]);
    }
    if (eq(op, "invThreshold")) {
        if (comptime R == 2 and C == 2) {
            m = try lm.Mat22Ops.invChecked(T, m, @floatCast(c.inputs.alpha));
            return copy(T, @as([*]const T, @ptrCast(&m.mat))[0..4]);
        } else return error.UnsupportedShape;
    }
    if (eq(op, "mulVecMethod")) {
        const v = m.mulVec(lm.Vec(C, T).initSlice(b));
        return copy(T, &v.vec);
    }
    if (eq(op, "mulMatMethod") or eq(op, "mulMatRectMethod")) {
        if (comptime R == C) {
            m = if (eq(op, "mulMatMethod")) m.mulMat(lm.Mat(R, C, T).initSlice(b)) else m.mulMatRect(C, lm.Mat(R, C, T).initSlice(b));
            return copy(T, @as([*]const T, @ptrCast(&m.mat))[0 .. R * C]);
        } else return error.UnsupportedShape;
    }
    if (eq(op, "insertMat22")) {
        if (comptime R == 4 and C == 4) lm.Mat44Ops.insertMat22(T, &m, lm.Mat(2, 2, T).initFill(@floatCast(c.inputs.alpha)), c.inputs.index[0], c.inputs.index[1]) else return error.UnsupportedShape;
        return copy(T, @as([*]const T, @ptrCast(&m.mat))[0 .. R * C]);
    }
    if (eq(op, "zeros")) m = lm.Mat(R, C, T).initZeros() else if (eq(op, "ones")) m = lm.Mat(R, C, T).initOnes() else if (eq(op, "fill")) m = lm.Mat(R, C, T).initFill(@floatCast(c.inputs.alpha)) else if (eq(op, "identity")) m = lm.Mat(R, C, T).initIdentity() else if (eq(op, "diag")) m = lm.Mat(R, C, T).initDiag(@floatCast(c.inputs.alpha)) else if (eq(op, "scale")) m = m.mulScal(@floatCast(c.inputs.alpha)) else if (eq(op, "add")) m = m.add(lm.Mat(R, C, T).initSlice(b)) else if (eq(op, "sub")) m = m.sub(lm.Mat(R, C, T).initSlice(b)) else if (eq(op, "initRows")) m = lm.Mat(R, C, T).initRows(m.mat) else if (eq(op, "row")) {
        const v = m.getRowVec(c.inputs.index[0]);
        return copy(T, &v.vec);
    } else if (eq(op, "column")) {
        const v = m.getColVec(c.inputs.index[1]);
        return copy(T, &v.vec);
    } else if (eq(op, "insertRow")) m.insertRowVec(c.inputs.index[0], 0, C, lm.Vec(C, T).initFill(@floatCast(c.inputs.alpha))) else if (eq(op, "insertColumn")) m.insertColVec(0, c.inputs.index[0], R, lm.Vec(R, T).initFill(@floatCast(c.inputs.alpha))) else if (eq(op, "submatrix")) {
        if (comptime R >= 2 and C >= 2) {
            const sub = m.getSubMat(c.inputs.index[0], c.inputs.index[1], 2, 2);
            return copy(T, @as([*]const T, @ptrCast(&sub.mat))[0..4]);
        } else return error.UnsupportedShape;
    } else if (eq(op, "insertSubmatrix")) {
        if (comptime R >= 2 and C >= 2) m.insertSubMat(0, 0, 2, 2, lm.Mat(2, 2, T).initFill(@floatCast(c.inputs.alpha))) else return error.UnsupportedShape;
    } else if (eq(op, "adj")) {
        if (comptime R == 2 and C == 2) m = lm.Mat22Ops.adj(T, m) else return error.UnsupportedShape;
    } else if (eq(op, "mulVec3")) {
        if (comptime R == 4 and C == 4) {
            const v = lm.Mat44Ops.mulVec3(T, m, lm.Vec(3, T).initSlice(b));
            return copy(T, &v.vec);
        } else return error.UnsupportedShape;
    } else return error.UnknownOperation;
    return copy(T, @as([*]const T, @ptrCast(&m.mat))[0 .. R * C]);
}
pub fn run(comptime T: type, c: anytype, a: []T, b: []T) ![]T {
    const op = c.operation;
    const shape = c.inputs.shape;
    const idx = c.inputs.index;
    const alpha: T = @floatCast(c.inputs.alpha);
    if (eq(op, "integerArithmetic")) {
        var v = lm.Vec(4, i32).initFill(@intFromFloat(a[0]));
        v.set(idx[0], @intFromFloat(alpha));
        return copy(T, &.{ @floatFromInt(v.x()), @floatFromInt(v.y()), @floatFromInt(v.z()), @floatFromInt(v.w()), @floatFromInt(v.sum()) });
    }
    if (eq(op, "badPermutation") or eq(op, "luFactoredZero") or eq(op, "choleskyFactoredZero") or std.mem.endsWith(u8, op, "Scratch")) {
        const n = shape[0];
        const work = try alloc.alloc(T, n * n);
        defer alloc.free(work);
        const rhs = try alloc.alloc(T, if (std.mem.endsWith(u8, op, "Scratch")) n - 1 else n);
        defer alloc.free(rhs);
        const out = try alloc.alloc(T, n);
        errdefer alloc.free(out);
        const perm = try alloc.alloc(usize, n);
        defer alloc.free(perm);
        for (perm, 0..) |*v, i| v.* = i;
        const m = lm.MatView(T).init(a, n, n);
        const wm = lm.MatView(T).init(work, n, n);
        const bv = lm.VecView(T).init(b);
        const wv = lm.VecView(T).init(rhs);
        const dst = lm.VecView(T).init(out);
        if (eq(op, "badPermutation") or eq(op, "luFactoredZero")) try lm.linsolve.solveLUFactoredMatViewChecked(T, m, idx, bv, wv, dst) else if (eq(op, "choleskyFactoredZero")) try lm.linsolve.solveCholeskyFactoredMatViewChecked(T, m, bv, wv, dst) else if (eq(op, "gaussScratch")) try lm.linsolve.solveGaussMatViewChecked(T, m, bv, wm, wv, dst) else if (eq(op, "luScratch")) try lm.linsolve.solveLUMatViewChecked(T, m, bv, wm, perm, wv, dst) else try lm.linsolve.solveCholeskyMatViewChecked(T, m, bv, wm, wv, dst);
        return out;
    }
    if (eq(op, "permutation")) {
        if (eq(c.representation, "Vec")) {
            const perm: [3]usize = idx[0..3].*;
            const v = lm.rowops.applyPermutationVec(3, T, perm, lm.Vec(3, T).initSlice(a));
            return copy(T, &v.vec);
        }
        const out = try alloc.alloc(T, a.len);
        lm.rowops.applyPermutationVecView(T, idx, lm.VecView(T).init(a), lm.VecView(T).init(out));
        return out;
    }
    if (eq(c.representation, "Vec")) {
        if (eq(op, "componentExtrema")) {
            var vectors: [3]lm.Vec(3, T) = undefined;
            for (&vectors, 0..) |*v, i| v.* = lm.Vec(3, T).initSlice(a[i * 3 ..][0..3]);
            const out = try alloc.alloc(T, 6);
            for (0..3) |i| {
                out[i] = lm.vectype.Vec3SliceOps.max(T, &vectors, i);
                out[3 + i] = lm.vectype.Vec3SliceOps.min(T, &vectors, i);
            }
            return out;
        }
        inline for (.{ 1, 2, 3, 4, 6, 7, 8, 10, 12, 16 }) |n| if (a.len == n) return vector(n, T, c, a, b);
        return error.UnsupportedShape;
    }
    if (eq(c.representation, "Mat")) {
        if (eq(op, "luReconstruction") or eq(op, "choleskyReconstruction") or eq(op, "luMultiple") or eq(op, "choleskyMultiple")) {
            inline for (.{ 2, 3, 6, 12 }) |n| if (shape[0] == n) return solverFixed(n, T, c, a, b);
            return error.UnsupportedShape;
        }
        inline for (.{ .{ 2, 2 }, .{ 3, 3 }, .{ 4, 4 }, .{ 2, 3 }, .{ 3, 2 } }) |dims| if (shape[0] == dims[0] and shape[1] == dims[1]) return matrixFixed(dims[0], dims[1], T, c, a, b);
        return error.UnsupportedShape;
    }
    if (eq(c.representation, "slice") or eq(c.representation, "VecView")) {
        const v = lm.VecView(T).init(a);
        if (eq(op, "extremaIndices")) return copy(T, &.{ v.max().val, @floatFromInt(v.max().idx), v.min().val, @floatFromInt(v.min().idx) });
        if (eq(op, "legacyNorms")) return copy(T, &.{ v.norm(), v.squaredNorm(), v.vecLen() });
        if (eq(op, "rangeLen")) return scalar(T, @floatFromInt(lm.sliceops.rangeLen(@floatCast(a[0]), @floatCast(a[1]), @floatCast(a[2]))));
        if (eq(op, "partialApply")) {
            lm.sliceops.apply(T, a, b, lm.sliceops.exp);
            return copy(T, a);
        }
        if (eq(op, "addChecked")) try lm.sliceops.addChecked(T, a, b, a) else if (eq(op, "subChecked")) try lm.sliceops.subChecked(T, a, b, a) else if (eq(op, "mulChecked")) try lm.sliceops.mulChecked(T, a, b, a) else if (eq(op, "divChecked")) try lm.sliceops.divChecked(T, a, b, a) else if (eq(op, "mulScalChecked")) try lm.sliceops.mulScalChecked(T, a, alpha, b) else if (eq(op, "scale")) {
            if (eq(c.representation, "slice")) lm.sliceops.mulScal(T, a, alpha, a) else v.mulScalInPlace(alpha);
        } else if (eq(op, "zeros")) v.fill(0) else if (eq(op, "ones")) v.fill(1) else if (eq(op, "fill")) v.fill(alpha) else if (eq(op, "applySqrt")) {
            if (eq(c.representation, "slice")) lm.sliceops.apply(T, a, a, std.math.sqrt) else v.applyInPlace(std.math.sqrt);
        } else if (eq(op, "applyAtan")) {
            if (eq(c.representation, "slice")) lm.sliceops.apply(T, a, a, std.math.atan) else v.applyInPlace(std.math.atan);
        } else if (eq(op, "applyExp")) {
            if (eq(c.representation, "slice")) lm.sliceops.apply(T, a, a, lm.sliceops.exp) else v.applyInPlace(lm.sliceops.exp);
        } else return error.UnknownOperation;
        return copy(T, a);
    }
    if (eq(c.representation, "NDArray")) return ndarray(T, c, a, b);
    if (eq(c.representation, "MatView")) {
        if (eq(op, "luReconstruction") or eq(op, "choleskyReconstruction") or eq(op, "luMultiple") or eq(op, "choleskyMultiple")) return solver(T, c, a, b);
        const data = if (eq(op, "rowMutation")) a[1 .. a.len - 1] else a;
        var m = lm.MatView(T).init(data, shape[0], shape[1]);
        if (eq(op, "flatHelpers")) return copy(T, &.{ @floatFromInt(m.flatIndex(idx[0], idx[1])), @floatFromInt(m.rowBase(idx[0])), m.getFlat(m.flatIndex(idx[0], idx[1])) });
        if (eq(op, "mulVecView")) {
            const out = try alloc.alloc(T, shape[0]);
            const vec = lm.VecView(T).init(b);
            var dest = lm.VecView(T).init(out);
            lm.MatViewOps(T).mulVec(&m, &vec, &dest);
            return out;
        }
        if (eq(op, "mulSquareView") or eq(op, "mulView") or eq(op, "scaleOut") or eq(op, "addOut") or eq(op, "subOut")) {
            const out = try alloc.alloc(T, a.len);
            var dst = lm.MatView(T).init(out, shape[0], shape[1]);
            const other = lm.MatView(T).init(b, shape[0], shape[1]);
            if (eq(op, "mulSquareView")) lm.MatViewOps(T).mulSquare(&m, &other, &dst) else if (eq(op, "mulView")) lm.MatViewOps(T).mul(&m, &other, &dst) else if (eq(op, "scaleOut")) lm.MatViewOps(T).mulScal(&m, alpha, &dst) else if (eq(op, "addOut")) lm.MatViewOps(T).add(&m, &other, &dst) else lm.MatViewOps(T).sub(&m, &other, &dst);
            return out;
        }
        if (eq(op, "rowMutation")) {
            @memset(m.getSlice(idx[0]), alpha);
            return copy(T, a);
        }
        if (eq(op, "zeros")) m.fill(0) else if (eq(op, "fill")) m.fill(alpha) else if (eq(op, "identity")) m.identity() else if (eq(op, "diag")) m.fillDiag(0, alpha) else if (eq(op, "row")) return copy(T, m.getSlice(idx[0])) else if (eq(op, "column")) {
            const out = try alloc.alloc(T, shape[0]);
            for (out, 0..) |*v, i| v.* = m.get(i, idx[1]);
            return out;
        } else if (eq(op, "columnExtrema")) return copy(T, &.{ m.minByRow(idx[0]), m.maxByRow(idx[0]) }) else if (eq(op, "flatAccess")) m.setFlat(m.flatIndex(idx[0], idx[1]), alpha) else if (eq(op, "transposeSquare")) {
            var work = try lm.MatView(T).initAlloc(alloc, shape[0], shape[1]);
            defer alloc.free(work.slice);
            m.transposeSquare(&work);
        } else if (eq(op, "scale")) m.mulScalInPlace(alpha) else {
            const other = lm.MatView(T).init(b, shape[0], shape[1]);
            if (eq(op, "add")) m.addInPlace(&other) else if (eq(op, "sub")) m.subInPlace(&other) else if (eq(op, "mulInPlace")) m.mulInPlace(&other) else if (eq(op, "divInPlace")) m.divInPlace(&other) else if (eq(op, "mulOut")) lm.MatViewOps(T).mulElemWise(&m, &other, &m) else if (eq(op, "divOut")) lm.MatViewOps(T).divElemWise(&m, &other, &m) else return error.UnknownOperation;
        }
        return copy(T, a);
    }
    return error.UnknownRepresentation;
}
fn ndarray(comptime T: type, c: anytype, a: []T, b: []T) ![]T {
    const op = c.operation;
    const shape = c.inputs.shape;
    const alpha: T = @floatCast(c.inputs.alpha);
    var arr = try lm.NDArray(T).init(alloc, a, shape);
    defer arr.deinit(alloc);
    if (eq(op, "planeTail")) return copy(T, arr.getPlaneSlice(c.inputs.index[0]));
    if (eq(op, "getSlice")) return copy(T, arr.getSlice(c.inputs.index, @intFromFloat(c.inputs.alpha)));
    if (eq(op, "emptyPrefix")) {
        var view = try arr.fixedPrefixView(alloc, &.{});
        defer view.deinit(alloc);
        return copy(T, view.slice);
    }
    if (eq(op, "dupe")) {
        var dupe = try arr.dupe(alloc);
        defer dupe.deinit(alloc);
        defer alloc.free(dupe.slice);
        const out = try alloc.alloc(T, a.len * 2);
        @memcpy(out[0..a.len], a);
        @memcpy(out[a.len..], dupe.slice);
        return out;
    }
    if (eq(op, "initFlat")) {
        var owned = try lm.NDArray(T).initFlat(alloc, shape);
        defer owned.deinit(alloc);
        defer alloc.free(owned.slice);
        owned.fill(alpha);
        return copy(T, owned.slice);
    }
    if (eq(op, "matchDims")) {
        var same = try lm.NDArray(T).init(alloc, a, shape);
        defer same.deinit(alloc);
        var other = try lm.NDArray(T).init(alloc, a, &.{a.len});
        defer other.deinit(alloc);
        return copy(T, &.{ if (lm.ndarraytype.matchArrayDims(T, &arr, &same)) 1 else 0, if (lm.ndarraytype.matchArrayDims(T, &arr, &other)) 1 else 0 });
    }
    if (eq(op, "extractMat")) {
        const out = try alloc.alloc(T, shape[1] * shape[2]);
        var dst = lm.MatView(T).init(out, shape[1], shape[2]);
        try lm.ndarraytype.NDArrayOps(T).extractMat(alloc, &arr, c.inputs.index, 1, 2, &dst);
        return out;
    }
    if (eq(op, "getset")) {
        arr.set(c.inputs.index, alpha);
        arr.setFlat(arr.getFlatIdx(c.inputs.index), arr.get(c.inputs.index));
    } else if (eq(op, "fill")) arr.fill(alpha) else if (eq(op, "scale")) arr.mulScalInPlace(alpha) else {
        var other = try lm.NDArray(T).init(alloc, b, shape);
        defer other.deinit(alloc);
        if (eq(op, "add")) arr.addInPlace(&other) else if (eq(op, "sub")) arr.subInPlace(&other) else if (eq(op, "mulInPlace")) arr.mulInPlace(&other) else if (eq(op, "divInPlace")) arr.divInPlace(&other) else if (eq(op, "addOut")) lm.ndarraytype.NDArrayOps(T).add(&arr, &other, &arr) else if (eq(op, "subOut")) lm.ndarraytype.NDArrayOps(T).sub(&arr, &other, &arr) else if (eq(op, "mulOut")) lm.ndarraytype.NDArrayOps(T).mulElemWise(&arr, &other, &arr) else if (eq(op, "divOut")) lm.ndarraytype.NDArrayOps(T).divElemWise(&arr, &other, &arr) else return error.UnknownOperation;
    }
    return copy(T, a);
}
fn solver(comptime T: type, c: anytype, a: []T, b: []T) ![]T {
    const n = c.inputs.shape[0];
    const op = c.operation;
    const work = try alloc.alloc(T, n * n);
    defer alloc.free(work);
    const perm = try alloc.alloc(usize, n);
    defer alloc.free(perm);
    const matrix = lm.MatView(T).init(a, n, n);
    const factor = lm.MatView(T).init(work, n, n);
    const chol = std.mem.startsWith(u8, op, "cholesky");
    if (chol) try lm.linsolve.factorCholeskyMatViewChecked(T, matrix, factor) else try lm.linsolve.factorLUMatViewChecked(T, matrix, factor, perm);
    if (std.mem.endsWith(u8, op, "Multiple")) {
        const out = try alloc.alloc(T, b.len);
        errdefer alloc.free(out);
        const scratch = try alloc.alloc(T, n);
        defer alloc.free(scratch);
        for (0..b.len / n) |k| {
            const rhs = lm.VecView(T).init(b[k * n ..][0..n]);
            const dst = lm.VecView(T).init(out[k * n ..][0..n]);
            if (chol) try lm.linsolve.solveCholeskyFactoredMatViewChecked(T, factor, rhs, lm.VecView(T).init(scratch), dst) else try lm.linsolve.solveLUFactoredMatViewChecked(T, factor, perm, rhs, lm.VecView(T).init(scratch), dst);
        }
        return out;
    }
    var lower = try alloc.alloc(T, n * n);
    defer alloc.free(lower);
    var upper = try alloc.alloc(T, n * n);
    defer alloc.free(upper);
    @memset(lower, 0);
    @memset(upper, 0);
    for (0..n) |r| for (0..n) |col| {
        if (chol) {
            if (col <= r) lower[r * n + col] = work[r * n + col];
            if (r <= col) upper[r * n + col] = work[col * n + r];
        } else {
            lower[r * n + col] = if (r == col) 1 else if (r > col) work[r * n + col] else 0;
            upper[r * n + col] = if (r <= col) work[r * n + col] else 0;
        }
    };
    const product = try alloc.alloc(T, n * n);
    defer alloc.free(product);
    lm.matops.mulMatMatView(T, lm.MatView(T).init(lower, n, n), lm.MatView(T).init(upper, n, n), lm.MatView(T).init(product, n, n));
    const out = try alloc.alloc(T, n * n);
    for (0..n) |r| @memcpy(out[(if (chol) r else perm[r]) * n ..][0..n], product[r * n ..][0..n]);
    return out;
}

fn solverFixed(comptime N: usize, comptime T: type, c: anytype, a: []T, b: []T) ![]T {
    const m = lm.Mat(N, N, T).initSlice(a);
    const chol = std.mem.startsWith(u8, c.operation, "cholesky");
    var lu: lm.linsolve.LUStack(N, T) = undefined;
    var ch: lm.linsolve.CholeskyStack(N, T) = undefined;
    if (chol) ch = try lm.linsolve.factorCholeskyMatChecked(N, T, m) else lu = try lm.linsolve.factorLUMatChecked(N, T, m);
    if (std.mem.endsWith(u8, c.operation, "Multiple")) {
        const out = try alloc.alloc(T, b.len);
        errdefer alloc.free(out);
        for (0..b.len / N) |k| {
            const rhs = lm.Vec(N, T).initSlice(b[k * N ..][0..N]);
            const result = if (chol) try lm.linsolve.solveCholeskyFactoredMatChecked(N, T, ch, rhs) else try lm.linsolve.solveLUFactoredMatChecked(N, T, lu, rhs);
            @memcpy(out[k * N ..][0..N], &result.vec);
        }
        return out;
    }
    var lower = lm.Mat(N, N, T).initZeros();
    var upper = lm.Mat(N, N, T).initZeros();
    for (0..N) |r| for (0..N) |col| {
        if (chol) {
            if (col <= r) lower.set(r, col, ch.lower.get(r, col));
            if (r <= col) upper.set(r, col, ch.lower.get(col, r));
        } else {
            lower.set(r, col, if (r == col) 1 else if (r > col) lu.lu.get(r, col) else 0);
            upper.set(r, col, if (r <= col) lu.lu.get(r, col) else 0);
        }
    };
    const product = lm.matops.mulMatMat(N, N, N, T, lower, upper);
    const out = try alloc.alloc(T, N * N);
    for (0..N) |r| for (0..N) |col| {
        out[(if (chol) r else lu.permutation[r]) * N + col] = product.get(r, col);
    };
    return out;
}
