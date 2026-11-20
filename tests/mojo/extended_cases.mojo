"""Adapters for shared method/property cases; references live in JSON."""
import lagomath as lm
from std.collections import Array, Span
from std.math import sqrt, atan, exp
from std.python import PythonObject


def sqrtScalar[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    return sqrt(value)


def atanScalar[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    return Scalar[dtype](atan(Float64(value)))


def expScalar[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    return Scalar[dtype](exp(Float64(value)))


def collect[
    dtype: DType, o: Origin
](a: Span[Scalar[dtype], o]) -> List[Scalar[dtype]]:
    return List(a)


def vector[
    N: Int, dtype: DType
](
    op: String,
    a: List[Scalar[dtype]],
    b: List[Scalar[dtype]],
    alpha: Scalar[dtype],
) raises -> List[Scalar[dtype]]:
    var v = lm.Vec[N, dtype].initSlice(Span(a))
    if op == "zeros":
        v = lm.Vec[N, dtype].initZeros()
    elif op == "ones":
        v = lm.Vec[N, dtype].initOnes()
    elif op == "fill":
        v = lm.Vec[N, dtype].initFill(alpha)
    elif op == "scale":
        v = v.mulScal(alpha)
    elif op == "applySqrt":
        v = v.apply[sqrtScalar[dtype]]()
    elif op == "applyAtan":
        v = v.apply[atanScalar[dtype]]()
    elif op == "applyExp":
        v = v.apply[expScalar[dtype]]()
    elif op == "legacyNorms":
        return [v.norm(), v.squaredNorm(), v.vecLen()]
    elif op == "extremaIndices":
        return [
            v.max().val,
            Scalar[dtype](v.max().idx),
            v.min().val,
            Scalar[dtype](v.min().idx),
        ]
    elif op == "coordinates":
        return [v.x(), v.y(), v.z(), v.w()]
    elif op == "initVec2":
        var result = lm.initVec2(a[0], a[1])
        return collect(Span(result.vec))
    elif op == "initVec3":
        var result = lm.initVec3(a[0], a[1], a[2])
        return collect(Span(result.vec))
    elif op == "cross":
        var result = lm.Vec3Ops.cross(
            lm.Vec[3, dtype].initSlice(Span(a)),
            lm.Vec[3, dtype].initSlice(Span(b)),
        )
        return collect(Span(result.vec))
    else:
        raise Error("UnknownOperation")
    return collect(Span(v.vec))


def matrix[
    R: Int, C: Int, dtype: DType
](
    op: String,
    a: List[Scalar[dtype]],
    b: List[Scalar[dtype]],
    idx: List[Int],
    alpha: Scalar[dtype],
) raises -> List[Scalar[dtype]]:
    var m = lm.Mat[R, C, dtype].initSlice(Span(a))
    if (
        op == "swapRowsMethod"
        or op == "scaleRowMethod"
        or op == "addScaledRowMethod"
    ):
        if op == "swapRowsMethod":
            lm.rowops.swapRowsMat(idx[0], idx[1], m)
        elif op == "scaleRowMethod":
            lm.rowops.scaleRowMat(idx[0], alpha, m)
        else:
            lm.rowops.addScaledRowMat(idx[0], idx[1], alpha, m)
        return collect(Span(m.mat))
    if op == "invThreshold":
        var square = lm.Mat[2, 2, dtype].initSlice(Span(a))
        var value = lm.Mat22Ops.invChecked(square, alpha)
        return collect(Span(value.mat))
    if op == "mulVecMethod":
        var v = m.mulVec(lm.Vec[C, dtype].initSlice(Span(b)))
        return collect(Span(v.vec))
    if op == "mulMatRectMethod":
        var value = m.mulMatRect(lm.Mat[C, C, dtype].initSlice(Span(b)))
        return collect(Span(value.mat))
    if op == "mulMatMethod":
        comptime if R == C:
            var value = m.mulMat(lm.Mat[R, C, dtype].initSlice(Span(b)))
            return collect(Span(value.mat))
        else:
            raise Error("UnsupportedShape")
    if op == "insertMat22":
        var square = lm.Mat[4, 4, dtype].initSlice(Span(a))
        lm.Mat44Ops.insertMat22(
            square, lm.Mat[2, 2, dtype](alpha), idx[0], idx[1]
        )
        return collect(Span(square.mat))
    if op == "zeros":
        m = lm.Mat[R, C, dtype].initZeros()
    elif op == "ones":
        m = lm.Mat[R, C, dtype].initOnes()
    elif op == "fill":
        m = lm.Mat[R, C, dtype].initFill(alpha)
    elif op == "identity":
        m = lm.Mat[R, C, dtype].initIdentity()
    elif op == "diag":
        m = lm.Mat[R, C, dtype].initDiag(alpha)
    elif op == "scale":
        m = m.mulScal(alpha)
    elif op == "add":
        m = m.add(lm.Mat[R, C, dtype].initSlice(Span(b)))
    elif op == "sub":
        m = m.sub(lm.Mat[R, C, dtype].initSlice(Span(b)))
    elif op == "initRows":
        var rows = Array[Array[Scalar[dtype], C], R](
            fill=Array[Scalar[dtype], C](fill=0)
        )
        for r in range(R):
            for c in range(C):
                rows[r][c] = m.get(r, c)
        m = lm.Mat[R, C, dtype].initRows(rows)
    elif op == "row":
        var v = m.getRowVec(idx[0])
        return collect(Span(v.vec))
    elif op == "column":
        var v = m.getColVec(idx[1])
        return collect(Span(v.vec))
    elif op == "insertRow":
        m.insertRowVec(idx[0], 0, lm.Vec[C, dtype](alpha))
    elif op == "insertColumn":
        m.insertColVec(0, idx[0], lm.Vec[R, dtype](alpha))
    elif op == "submatrix":
        var sub = m.getSubMat[2, 2](idx[0], idx[1])
        return collect(Span(sub.mat))
    elif op == "insertSubmatrix":
        m.insertSubMat(0, 0, lm.Mat[2, 2, dtype](alpha))
    elif op == "adj":
        var square = lm.Mat[2, 2, dtype].initSlice(Span(a))
        var value = lm.Mat22Ops.adj(square)
        return collect(Span(value.mat))
    elif op == "mulVec3":
        var square = lm.Mat[4, 4, dtype].initSlice(Span(a))
        var value = lm.Mat44Ops.mulVec3(
            square, lm.Vec[3, dtype].initSlice(Span(b))
        )
        return collect(Span(value.vec))
    else:
        raise Error("UnknownOperation")
    return collect(Span(m.mat))


def solver[
    dtype: DType
](
    op: String, a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int
) raises -> List[Scalar[dtype]]:
    var work = List[Scalar[dtype]](length=n * n, fill=0)
    var perm = List[Int](length=n, fill=0)
    var m = lm.MatView(Span(a), n, n)
    var factor = lm.MatView(Span(work), n, n)
    var pp = Span(perm)
    var chol = op.startswith("cholesky")
    if chol:
        lm.linsolve.factorCholeskyMatView(m, factor)
    else:
        lm.linsolve.factorLUMatView(m, factor, pp)
    if op.endswith("Multiple"):
        var result = List[Scalar[dtype]](length=len(b), fill=0)
        var scratch = List[Scalar[dtype]](length=n, fill=0)
        var out = Span(result)
        var rhs = Span(b)
        for k in range(len(b) // n):
            var rb = lm.VecView(rhs[k * n : (k + 1) * n])
            var dst = lm.VecView(out[k * n : (k + 1) * n])
            var wv = lm.VecView(Span(scratch))
            if chol:
                lm.linsolve.solveCholeskyFactoredMatView(factor, rb, wv, dst)
            else:
                lm.linsolve.solveLUFactoredMatView(factor, pp, rb, wv, dst)
        return result^
    var lower = List[Scalar[dtype]](length=n * n, fill=0)
    var upper = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(n):
            if chol:
                if c <= r:
                    lower[r * n + c] = factor.get(r, c)
                if r <= c:
                    upper[r * n + c] = factor.get(c, r)
            else:
                lower[r * n + c] = 1 if r == c else (
                    factor.get(r, c) if r > c else 0
                )
                upper[r * n + c] = factor.get(r, c) if r <= c else 0
    var product = List[Scalar[dtype]](length=n * n, fill=0)
    lm.matops.mulMatMatView(
        lm.MatView(Span(lower), n, n),
        lm.MatView(Span(upper), n, n),
        lm.MatView(Span(product), n, n),
    )
    var result = List[Scalar[dtype]](length=n * n, fill=0)
    for r in range(n):
        for c in range(n):
            result[(r if chol else perm[r]) * n + c] = product[r * n + c]
    return result^


def run[
    dtype: DType
](
    c: PythonObject, var a: List[Scalar[dtype]], var b: List[Scalar[dtype]]
) raises -> List[Scalar[dtype]]:
    var inp = c["inputs"]
    var op = String(py=c["operation"])
    var rep = String(py=c["representation"])
    var alpha = Scalar[dtype](Float64(py=inp["alpha"]))
    var shape = List[Int]()
    var idx = List[Int]()
    for i in range(Int(py=inp["shape"].__len__())):
        shape.append(Int(py=inp["shape"][i]))
    for i in range(Int(py=inp["index"].__len__())):
        idx.append(Int(py=inp["index"][i]))
    if (
        op == "badPermutation"
        or op == "luFactoredZero"
        or op == "choleskyFactoredZero"
        or op.endswith("Scratch")
    ):
        var n = shape[0]
        var work = List[Scalar[dtype]](length=n * n, fill=0)
        var scratch = List[Scalar[dtype]](
            length=n - 1 if op.endswith("Scratch") else n, fill=0
        )
        var result = List[Scalar[dtype]](length=n, fill=0)
        var perm = List[Int](length=n, fill=0)
        for i in range(n):
            perm[i] = i
        var m = lm.MatView(Span(a), n, n)
        var wm = lm.MatView(Span(work), n, n)
        var bv = lm.VecView(Span(b))
        var wv = lm.VecView(Span(scratch))
        var dst = lm.VecView(Span(result))
        if op == "badPermutation" or op == "luFactoredZero":
            lm.linsolve.solveLUFactoredMatViewChecked(m, Span(idx), bv, wv, dst)
        elif op == "choleskyFactoredZero":
            lm.linsolve.solveCholeskyFactoredMatViewChecked(m, bv, wv, dst)
        elif op == "gaussScratch":
            lm.linsolve.solveGaussMatViewChecked(m, bv, wm, wv, dst)
        elif op == "luScratch":
            lm.linsolve.solveLUMatViewChecked(m, bv, wm, Span(perm), wv, dst)
        else:
            lm.linsolve.solveCholeskyMatViewChecked(m, bv, wm, wv, dst)
        return result^
    if op == "integerArithmetic":
        var v = lm.Vec[4, DType.int32](Int32(a[0]))
        v.set(idx[0], Int32(alpha))
        return [
            Scalar[dtype](v.x()),
            Scalar[dtype](v.y()),
            Scalar[dtype](v.z()),
            Scalar[dtype](v.w()),
            Scalar[dtype](v.sum()),
        ]
    if op == "permutation":
        var perm = Array[Int, 3](fill=0)
        for i in range(3):
            perm[i] = idx[i]
        if rep == "Vec":
            var v = lm.rowops.applyPermutationVec(
                perm, lm.Vec[3, dtype].initSlice(Span(a))
            )
            return collect(Span(v.vec))
        var result = List[Scalar[dtype]](length=len(a), fill=0)
        lm.rowops.applyPermutationVecView(
            Span(perm), lm.VecView(Span(a)), lm.VecView(Span(result))
        )
        return result^
    if rep == "Vec":
        if op == "componentExtrema":
            var values = List[lm.Vec[3, dtype]]()
            var data = Span(a)
            for i in range(3):
                values.append(
                    lm.Vec[3, dtype].initSlice(data[i * 3 : (i + 1) * 3])
                )
            var result = List[Scalar[dtype]]()
            for i in range(3):
                result.append(lm.Vec3SliceOps.max(Span(values), i))
            for i in range(3):
                result.append(lm.Vec3SliceOps.min(Span(values), i))
            return result^
        if len(a) == 2:
            return vector[2, dtype](op, a, b, alpha)
        if len(a) == 3:
            return vector[3, dtype](op, a, b, alpha)
        if len(a) == 4:
            return vector[4, dtype](op, a, b, alpha)
        if len(a) == 7:
            return vector[7, dtype](op, a, b, alpha)
        if len(a) == 10:
            return vector[10, dtype](op, a, b, alpha)
        raise Error("UnsupportedShape")
    if (
        op == "luReconstruction"
        or op == "choleskyReconstruction"
        or op == "luMultiple"
        or op == "choleskyMultiple"
    ):
        if rep == "Mat":
            if shape[0] == 2:
                return solverFixed[2, dtype](op, a, b)
            if shape[0] == 3:
                return solverFixed[3, dtype](op, a, b)
            if shape[0] == 6:
                return solverFixed[6, dtype](op, a, b)
            if shape[0] == 12:
                return solverFixed[12, dtype](op, a, b)
            raise Error("UnsupportedShape")
        return solver(op, a, b, shape[0])
    if rep == "Mat":
        if shape[0] == 2 and shape[1] == 2:
            return matrix[2, 2, dtype](op, a, b, idx, alpha)
        if shape[0] == 3 and shape[1] == 3:
            return matrix[3, 3, dtype](op, a, b, idx, alpha)
        if shape[0] == 4 and shape[1] == 4:
            return matrix[4, 4, dtype](op, a, b, idx, alpha)
        if shape[0] == 2 and shape[1] == 3:
            return matrix[2, 3, dtype](op, a, b, idx, alpha)
        if shape[0] == 3 and shape[1] == 2:
            return matrix[3, 2, dtype](op, a, b, idx, alpha)
        raise Error("UnsupportedShape")
    if rep == "VecView" or rep == "slice":
        var data = Span(a)
        var other = Span(b)
        var output = a.copy()
        var dst = Span(output)
        var v = lm.VecView(data)
        if op == "extremaIndices":
            return [
                v.max().val,
                Scalar[dtype](v.max().idx),
                v.min().val,
                Scalar[dtype](v.min().idx),
            ]
        if op == "legacyNorms":
            return [v.norm(), v.squaredNorm(), v.vecLen()]
        if op == "rangeLen":
            return [
                Scalar[dtype](
                    lm.sliceops.rangeLen(
                        Float64(a[0]), Float64(a[1]), Float64(a[2])
                    )
                )
            ]
        if op == "addChecked":
            lm.sliceops.addChecked(data, other, dst)
        elif op == "subChecked":
            lm.sliceops.subChecked(data, other, dst)
        elif op == "mulChecked":
            lm.sliceops.mulChecked(data, other, dst)
        elif op == "divChecked":
            lm.sliceops.divChecked(data, other, dst)
        elif op == "mulScalChecked":
            lm.sliceops.mulScalChecked(data, alpha, other)
        elif op == "partialApply":
            lm.sliceops.apply[dtype, _, _, expScalar[dtype]](data, other)
        elif op == "scale":
            if rep == "slice":
                lm.sliceops.mulScal(data, alpha, dst)
                return output^
            else:
                v.mulScalInPlace(alpha)
        elif op == "zeros":
            v.fill(0)
        elif op == "ones":
            v.fill(1)
        elif op == "fill":
            v.fill(alpha)
        elif op == "applySqrt":
            if rep == "slice":
                lm.sliceops.apply[dtype, _, _, sqrtScalar[dtype]](dst, data)
                return output^
            else:
                v.applyInPlace[sqrtScalar[dtype]]()
        elif op == "applyAtan":
            if rep == "slice":
                lm.sliceops.apply[dtype, _, _, atanScalar[dtype]](dst, data)
                return output^
            else:
                v.applyInPlace[atanScalar[dtype]]()
        elif op == "applyExp":
            if rep == "slice":
                lm.sliceops.apply[dtype, _, _, expScalar[dtype]](dst, data)
                return output^
            else:
                v.applyInPlace[expScalar[dtype]]()
        else:
            raise Error("UnknownOperation")
        return a^
    if rep == "MatView":
        var data = Span(a)
        if op == "rowMutation":
            var m = lm.MatView(data[1 : len(a) - 1], shape[0], shape[1])
            m.getSlice(idx[0]).fill(alpha)
            return a^
        var output = a.copy()
        var dst = lm.MatView(Span(output), shape[0], shape[1])
        var m = lm.MatView(data, shape[0], shape[1])
        if op == "mulVecView":
            var result = List[Scalar[dtype]](length=shape[0], fill=0)
            lm.MatViewOps[dtype].mulVec(
                m, lm.VecView(Span(b)), lm.VecView(Span(result))
            )
            return result^
        if (
            op == "mulSquareView"
            or op == "mulView"
            or op == "scaleOut"
            or op == "addOut"
            or op == "subOut"
        ):
            var other = lm.MatView(Span(b), shape[0], shape[1])
            if op == "mulSquareView":
                lm.MatViewOps[dtype].mulSquare(m, other, dst)
            elif op == "mulView":
                lm.MatViewOps[dtype].mul(m, other, dst)
            elif op == "scaleOut":
                lm.MatViewOps[dtype].mulScal(m, alpha, dst)
            elif op == "addOut":
                lm.MatViewOps[dtype].add(m, other, dst)
            else:
                lm.MatViewOps[dtype].sub(m, other, dst)
            return output^
        if op == "row":
            return collect(m.getSlice(idx[0]))
        if op == "column":
            var result = List[Scalar[dtype]]()
            for r in range(shape[0]):
                result.append(m.get(r, idx[1]))
            return result^
        if op == "flatHelpers":
            return [
                Scalar[dtype](m.flatIndex(idx[0], idx[1])),
                Scalar[dtype](m.rowBase(idx[0])),
                m.getFlat(m.flatIndex(idx[0], idx[1])),
            ]
        if op == "columnExtrema":
            return [m.minByRow(idx[0]), m.maxByRow(idx[0])]
        if op == "zeros":
            m.fill(0)
        elif op == "fill":
            m.fill(alpha)
        elif op == "identity":
            m.identity()
        elif op == "diag":
            m.fillDiag(0, alpha)
        elif op == "flatAccess":
            m.setFlat(m.flatIndex(idx[0], idx[1]), alpha)
        elif op == "scale":
            m.mulScalInPlace(alpha)
        elif op == "transposeSquare":
            var buffer = List[Scalar[dtype]]()
            var work = lm.MatView[dtype, origin_of(buffer)].initAlloc(
                buffer, shape[0], shape[1]
            )
            m.transposeSquare(work)
        else:
            var other = lm.MatView(Span(b), shape[0], shape[1])
            if op == "add":
                m.addInPlace(other)
            elif op == "sub":
                m.subInPlace(other)
            elif op == "mulInPlace":
                m.mulInPlace(other)
            elif op == "divInPlace":
                m.divInPlace(other)
            elif op == "mulOut":
                lm.MatViewOps[dtype].mulElemWise(m, other, dst)
                return output^
            elif op == "divOut":
                lm.MatViewOps[dtype].divElemWise(m, other, dst)
                return output^
            else:
                raise Error("UnknownOperation")
        return a^
    if rep == "NDArray":
        var output = a.copy()
        var dst = lm.NDArray(Span(output), shape)
        var arr = lm.NDArray(Span(a), shape)
        if op == "planeTail":
            return collect(arr.getPlaneSlice(idx[0]).data)
        if op == "getSlice":
            return collect(arr.getSlice(idx, Int(alpha)).data)
        if op == "emptyPrefix":
            var prefix = List[Int]()
            var view = arr.fixedPrefixView(prefix)
            return collect(view.data)
        if op == "dupe":
            var storage = List[Scalar[dtype]]()
            var dupe = arr.dupe(storage)
            var result = a.copy()
            for i in range(dupe.length):
                result.append(dupe.getFlat(i))
            return result^
        if op == "initFlat":
            var storage = List[Scalar[dtype]]()
            var owned = lm.NDArray[dtype, origin_of(storage)].initFlat(
                storage, shape
            )
            owned.fill(alpha)
            return collect(owned.data)
        if op == "matchDims":
            var sameData = a.copy()
            var same = lm.NDArray(Span(sameData), shape)
            var flat: List[Int] = [len(a)]
            var otherData = a.copy()
            var other = lm.NDArray(Span(otherData), flat)
            return [
                Scalar[dtype](Int(lm.matchArrayDims(arr, same))),
                Scalar[dtype](Int(lm.matchArrayDims(arr, other))),
            ]
        if op == "extractMat":
            var result = List[Scalar[dtype]](length=shape[1] * shape[2], fill=0)
            var dst = lm.MatView(Span(result), shape[1], shape[2])
            lm.NDArrayOps[dtype].extractMat(arr, idx, 1, 2, dst)
            return result^
        if op == "getset":
            arr.set(idx, alpha)
            arr.setFlat(arr.getFlatIdx(idx), arr.get(idx))
        elif op == "fill":
            arr.fill(alpha)
        elif op == "scale":
            arr.mulScalInPlace(alpha)
        else:
            var other = lm.NDArray(Span(b), shape)
            if op == "add":
                arr.addInPlace(other)
            elif op == "sub":
                arr.subInPlace(other)
            elif op == "mulInPlace":
                arr.mulInPlace(other)
            elif op == "divInPlace":
                arr.divInPlace(other)
            elif op == "addOut":
                lm.NDArrayOps[dtype].add(arr, other, dst)
                return output^
            elif op == "subOut":
                lm.NDArrayOps[dtype].sub(arr, other, dst)
                return output^
            elif op == "mulOut":
                lm.NDArrayOps[dtype].mulElemWise(arr, other, dst)
                return output^
            elif op == "divOut":
                lm.NDArrayOps[dtype].divElemWise(arr, other, dst)
                return output^
            else:
                raise Error("UnknownOperation")
        return a^
    raise Error("UnknownRepresentation")


def solverFixed[
    N: Int, dtype: DType
](op: String, a: List[Scalar[dtype]], b: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
]:
    var m = lm.Mat[N, N, dtype].initSlice(Span(a))
    var lu = lm.linsolve.LU[N, dtype]()
    var ch = lm.linsolve.Cholesky[N, dtype]()
    var chol = op.startswith("cholesky")
    if chol:
        ch = lm.linsolve.factorCholeskyMatChecked(m)
    else:
        lu = lm.linsolve.factorLUMatChecked(m)
    if op.endswith("Multiple"):
        var result = List[Scalar[dtype]]()
        var data = Span(b)
        for k in range(len(b) // N):
            var rhs = lm.Vec[N, dtype].initSlice(data[k * N : (k + 1) * N])
            var dst = lm.Vec[N, dtype]()
            if chol:
                dst = lm.linsolve.solveCholeskyFactoredMatChecked(ch, rhs)
            else:
                dst = lm.linsolve.solveLUFactoredMatChecked(lu, rhs)
            for i in range(N):
                result.append(dst.get(i))
        return result^
    var lower = lm.Mat[N, N, dtype]()
    var upper = lm.Mat[N, N, dtype]()
    for r in range(N):
        for c in range(N):
            if chol:
                if c <= r:
                    lower.set(r, c, ch.lower.get(r, c))
                if r <= c:
                    upper.set(r, c, ch.lower.get(c, r))
            else:
                lower.set(
                    r, c, 1 if r == c else (lu.lu.get(r, c) if r > c else 0)
                )
                upper.set(r, c, lu.lu.get(r, c) if r <= c else 0)
    var product = lm.matops.mulMatMat(lower, upper)
    var result = List[Scalar[dtype]](length=N * N, fill=0)
    for r in range(N):
        for c in range(N):
            result[(r if chol else lu.permutation[r]) * N + c] = product.get(
                r, c
            )
    return result^
