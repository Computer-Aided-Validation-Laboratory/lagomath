from std.sys import argv
import extended_cases
from std.math import abs, isfinite, isnan, isinf
import lagomath as lm
from std.collections import Span
from std.python import Python, PythonObject


# Python handles JSON transport only; numerical work uses native Mojo values.
def numbers[dtype: DType](raw: PythonObject) raises -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]]()
    var builtins = Python.import_module("builtins")
    for i in range(Int(py=raw.__len__())):
        var item = raw[i]
        if Bool(py=builtins.isinstance(item, builtins.dict)):
            result.append(
                Scalar[dtype](Float64(py=builtins.float(item["special"])))
            )
        else:
            result.append(Scalar[dtype](Float64(py=item)))
    return result^


def vectorFixed[
    N: Int, dtype: DType
](
    op: String,
    a: List[Scalar[dtype]],
    b: List[Scalar[dtype]],
    alpha: Scalar[dtype],
    mutate_index: Int = 0,
    mutate_value: Scalar[dtype] = 0,
) raises -> List[Scalar[dtype]]:
    var v = lm.Vec[N, dtype]()
    var w = lm.Vec[N, dtype]()
    for i in range(N):
        v.set(i, a[i])
        w.set(i, b[i])
    var result = List[Scalar[dtype]]()
    if op == "valueCopy":
        var copy = v.copy()
        copy.set(mutate_index, mutate_value)
        for i in range(N):
            result.append(v.get(i))
        for i in range(N):
            result.append(copy.get(i))
        return result^
    if op == "dot":
        result.append(lm.vecops.dotVec(v, w))
    elif op == "normL1":
        result.append(lm.vecops.normL1Vec(v))
    elif op == "normL2":
        result.append(lm.vecops.normL2Vec(v))
    elif op == "normL2Square":
        result.append(lm.vecops.normL2SquareVec(v))
    elif op == "maxAbs":
        result.append(lm.vecops.maxAbsVec(v))
    elif op == "allFinite":
        result.append(Scalar[dtype](Int(lm.vecops.allFiniteVec(v))))
    elif op == "axpy":
        var dst = lm.vecops.axpyVec(alpha, v, w)
        for i in range(N):
            result.append(dst.get(i))
    else:
        return vectorExtra(v, w, op)
    return result^


def vectorExtra[
    N: Int, dtype: DType
](v: lm.Vec[N, dtype], w: lm.Vec[N, dtype], op: String) raises -> List[
    Scalar[dtype]
]:
    var result = List[Scalar[dtype]]()
    if op == "sum":
        result.append(v.sum())
    elif op == "mean":
        result.append(v.mean())
    elif op == "min":
        result.append(v.min().val)
    elif op == "max":
        result.append(v.max().val)
    elif op == "add" or op == "sub":
        var dst = lm.Vec[N, dtype]()
        if op == "add":
            dst = v.add(w)
        else:
            dst = v.sub(w)
        for i in range(N):
            result.append(dst.get(i))
    else:
        raise Error("UnknownOperation")
    return result^


def matrixFixed[
    R: Int, C: Int, P: Int, dtype: DType
](
    op: String,
    a: List[Scalar[dtype]],
    b: List[Scalar[dtype]],
    x: List[Scalar[dtype]],
    alpha: Scalar[dtype] = 0,
    mutate_index: Int = 0,
    mutate_value: Scalar[dtype] = 0,
) raises -> List[Scalar[dtype]]:
    var matrix = lm.Mat[R, C, dtype]()
    for i in range(R * C):
        matrix.mat[i] = a[i]
    var result = List[Scalar[dtype]]()
    if op == "valueCopy":
        var copy = matrix.copy()
        copy.set(mutate_index // C, mutate_index % C, mutate_value)
        for i in range(R * C):
            result.append(matrix.mat[i])
        for i in range(R * C):
            result.append(copy.mat[i])
        return result^
    if op == "det" or op == "inv" or op == "inverseIdentity":
        comptime if R == 2 and C == 2:
            var square = lm.Mat[2, 2, dtype]()
            for i in range(4):
                square.mat[i] = matrix.mat[i]
            if op == "det":
                result.append(lm.Mat22Ops.det(square))
            else:
                var dst = lm.Mat22Ops.invChecked(square, 0)
                if op == "inverseIdentity":
                    dst = lm.matops.mulMatMat(square, dst)
                for i in range(R * C):
                    result.append(dst.mat[i])
            return result^
        comptime if R == 3 and C == 3:
            var square = lm.Mat[3, 3, dtype]()
            for i in range(9):
                square.mat[i] = matrix.mat[i]
            if op == "det":
                result.append(lm.Mat33Ops.det(square))
            else:
                var dst = lm.Mat33Ops.invChecked(square, 0)
                if op == "inverseIdentity":
                    dst = lm.matops.mulMatMat(square, dst)
                for i in range(R * C):
                    result.append(dst.mat[i])
            return result^
        comptime if R == 4 and C == 4:
            var square = lm.Mat[4, 4, dtype]()
            for i in range(16):
                square.mat[i] = matrix.mat[i]
            if op == "det":
                result.append(lm.Mat44Ops.det(square))
            else:
                var dst = lm.Mat44Ops.invChecked(square, 0)
                if op == "inverseIdentity":
                    dst = lm.matops.mulMatMat(square, dst)
                for i in range(R * C):
                    result.append(dst.mat[i])
            return result^
        raise Error("UnsupportedShape")
    if op == "outer" or op == "outerAdd":
        var vx = lm.Vec[R, dtype]()
        var vy = lm.Vec[C, dtype]()
        for i in range(R):
            vx.set(i, x[i])
        for i in range(C):
            vy.set(i, b[i])
        var dst = lm.Mat[R, C, dtype]()
        if op == "outer":
            dst = lm.matops.outerMat(vx, vy)
        else:
            dst = lm.matops.outerAddMat(alpha, vx, vy, matrix)
        for i in range(R * C):
            result.append(dst.mat[i])
        return result^
    if op == "trace":
        result.append(matrix.trace())
    elif op == "frobeniusNorm":
        result.append(lm.matops.frobeniusNormMat(matrix))
    elif op == "maxAbs":
        result.append(lm.matops.maxAbsMat(matrix))
    elif op == "allFinite":
        result.append(Scalar[dtype](Int(lm.matops.allFiniteMat(matrix))))
    elif op == "transpose":
        var dst = matrix.transpose()
        for i in range(R * C):
            result.append(dst.mat[i])
    elif op == "matvec":
        var vector = lm.Vec[C, dtype]()
        for i in range(C):
            vector.set(i, x[i])
        var dst = lm.matops.mulVecMat(matrix, vector)
        for i in range(R):
            result.append(dst.get(i))
    elif op == "matmul":
        var rhs = lm.Mat[C, P, dtype]()
        for i in range(C * P):
            rhs.mat[i] = b[i]
        var dst = lm.matops.mulMatMat(matrix, rhs)
        for i in range(R * P):
            result.append(dst.mat[i])
    else:
        raise Error("UnknownOperation")
    return result^


def solveFixed[
    N: Int, dtype: DType
](op: String, a: List[Scalar[dtype]], b: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
]:
    var matrix = lm.Mat[N, N, dtype]()
    var rhs = lm.Vec[N, dtype]()
    for i in range(N * N):
        matrix.mat[i] = a[i]
    for i in range(N):
        rhs.set(i, b[i])
    var dst = lm.Vec[N, dtype]()
    if op == "gauss":
        dst = lm.linsolve.solveGaussMatChecked(matrix, rhs)
    elif op == "lu":
        dst = lm.linsolve.solveLUMatChecked(matrix, rhs)
    elif op == "cholesky":
        dst = lm.linsolve.solveCholeskyMatChecked(matrix, rhs)
    elif op == "luReuse":
        var factor = lm.linsolve.factorLUMat(matrix)
        dst = lm.linsolve.solveLUFactoredMat(factor, rhs)
    elif op == "choleskyReuse":
        var factor = lm.linsolve.factorCholeskyMat(matrix)
        dst = lm.linsolve.solveCholeskyFactoredMat(factor, rhs)
    else:
        raise Error("UnknownOperation")
    var result = List[Scalar[dtype]]()
    for i in range(N):
        result.append(dst.get(i))
    return result^


def run[dtype: DType](c: PythonObject) raises -> List[Scalar[dtype]]:
    var inp = c["inputs"]
    var a = numbers[dtype](inp["a"])
    var b = numbers[dtype](inp.get("b", Python.list()))
    var shape = List[Int]()
    for i in range(Int(py=inp["shape"].__len__())):
        shape.append(Int(py=inp["shape"][i]))
    if String(py=c["suite"]) == "extended":
        return extended_cases.run(c, a^, b^)
    var av = lm.VecView(Span(a))
    var bv = lm.VecView(Span(b))
    var op = String(py=c["operation"])
    var suite = String(py=c["suite"])
    var result = List[Scalar[dtype]]()
    if suite == "types":
        var rep = String(py=c["representation"])
        if rep == "Vec":
            return vectorFixed[3, dtype](
                op,
                a,
                b,
                0,
                Int(py=inp["mutate_index"]),
                Scalar[dtype](Float64(py=inp["mutate_value"])),
            )
        if rep == "Mat":
            var x = List[Scalar[dtype]]()
            return matrixFixed[2, 2, 2, dtype](
                op,
                a,
                b,
                x,
                mutate_index=Int(py=inp["mutate_index"]),
                mutate_value=Scalar[dtype](Float64(py=inp["mutate_value"])),
            )
        var view = lm.VecView(
            Span(a)[
                Int(py=inp["view_start"]) : Int(py=inp["view_start"]) + shape[0]
            ]
        )
        view.set(
            Int(py=inp["mutate_index"]),
            Scalar[dtype](Float64(py=inp["mutate_value"])),
        )
        for i in range(len(a)):
            result.append(a[i])
        return result^
    var representation = String(py=c["representation"])
    if representation == "Vec":
        if shape[0] == 1:
            return vectorFixed[1, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 2:
            return vectorFixed[2, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 3:
            return vectorFixed[3, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 4:
            return vectorFixed[4, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 6:
            return vectorFixed[6, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 8:
            return vectorFixed[8, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 12:
            return vectorFixed[12, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 16:
            return vectorFixed[16, dtype](
                op, a, b, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        raise Error("UnsupportedShape")
    if representation == "Mat":
        if suite == "linsolve":
            if shape[0] == 1:
                return solveFixed[1, dtype](op, a, b)
            if shape[0] == 2:
                return solveFixed[2, dtype](op, a, b)
            if shape[0] == 3:
                return solveFixed[3, dtype](op, a, b)
            if shape[0] == 4:
                return solveFixed[4, dtype](op, a, b)
            if shape[0] == 6:
                return solveFixed[6, dtype](op, a, b)
            if shape[0] == 8:
                return solveFixed[8, dtype](op, a, b)
            if shape[0] == 12:
                return solveFixed[12, dtype](op, a, b)
            raise Error("UnsupportedShape")
        var x = numbers[dtype](inp["x"])
        if shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
            return matrixFixed[1, 1, 1, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 2 and shape[1] == 2 and shape[2] == 2:
            return matrixFixed[2, 2, 2, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 3 and shape[1] == 3 and shape[2] == 3:
            return matrixFixed[3, 3, 3, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 4 and shape[1] == 4 and shape[2] == 4:
            return matrixFixed[4, 4, 4, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 6 and shape[1] == 6 and shape[2] == 6:
            return matrixFixed[6, 6, 6, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 8 and shape[1] == 8 and shape[2] == 8:
            return matrixFixed[8, 8, 8, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 12 and shape[1] == 12 and shape[2] == 12:
            return matrixFixed[12, 12, 12, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 2 and shape[1] == 3 and shape[2] == 4:
            return matrixFixed[2, 3, 4, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 6 and shape[1] == 12 and shape[2] == 8:
            return matrixFixed[6, 12, 8, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        if shape[0] == 12 and shape[1] == 6 and shape[2] == 3:
            return matrixFixed[12, 6, 3, dtype](
                op, a, b, x, Scalar[dtype](Float64(py=inp.get("alpha", 0)))
            )
        raise Error("UnsupportedShape")
    if suite == "vecops":
        if representation == "slice":
            if op == "dot":
                result.append(lm.sliceops.dot(av.data, bv.data))
            elif op == "normL1":
                result.append(lm.sliceops.normL1(av.data))
            elif op == "normL2":
                result.append(lm.sliceops.normL2(av.data))
            elif op == "normL2Square":
                result.append(lm.sliceops.normL2Square(av.data))
            elif op == "sum":
                result.append(lm.sliceops.sum(av.data))
            elif op == "mean":
                result.append(lm.sliceops.mean(av.data))
            elif op == "min":
                result.append(lm.sliceops.min(av.data).val)
            elif op == "max":
                result.append(lm.sliceops.max(av.data).val)
            if len(result) > 0:
                return result^
        var value: Scalar[dtype] = 0
        if op == "dot":
            value = lm.vecops.dotVecView(av, bv)
        elif op == "normL1":
            value = lm.vecops.normL1VecView(av)
        elif op == "normL2":
            value = lm.vecops.normL2VecView(av)
        elif op == "normL2Square":
            value = lm.vecops.normL2SquareVecView(av)
        elif op == "maxAbs":
            value = lm.vecops.maxAbsVecView(av)
        elif op == "allFinite":
            value = Scalar[dtype](Int(lm.vecops.allFiniteVecView(av)))
        elif op == "sum":
            value = av.sum()
        elif op == "mean":
            value = av.mean()
        elif op == "min":
            value = av.min().val
        elif op == "max":
            value = av.max().val
        else:
            if op == "axpy":
                result = b.copy()
                var dst = lm.VecView(Span(result))
                lm.vecops.axpyVecView(
                    Scalar[dtype](Float64(py=inp["alpha"])), av, dst
                )
            else:
                result = a.copy()
                var dst = lm.VecView(Span(result))
                if representation == "slice":
                    if op == "add":
                        lm.sliceops.add(av.data, bv.data, dst.data)
                    elif op == "sub":
                        lm.sliceops.sub(av.data, bv.data, dst.data)
                    elif op == "mul":
                        lm.sliceops.mul(av.data, bv.data, dst.data)
                    elif op == "div":
                        lm.sliceops.div(av.data, bv.data, dst.data)
                    else:
                        raise Error("UnknownOperation")
                elif op == "add":
                    dst.addInPlace(bv)
                elif op == "sub":
                    dst.subInPlace(bv)
                elif op == "mul":
                    dst.mulInPlace(bv)
                elif op == "div":
                    dst.divInPlace(bv)
                else:
                    raise Error("UnknownOperation")
            return result^
        result.append(value)
        return result^
    if suite == "ndarray":
        var array = lm.NDArray(av.data, shape)
        var indices = List[Int]()
        for i in range(len(shape)):
            indices.append(Int(py=inp["index"][i]))
        if op == "getFlatIdx":
            result.append(Scalar[dtype](array.getFlatIdx(indices)))
        elif op == "planeBase":
            result.append(Scalar[dtype](array.planeBase(indices[0])))
        elif op == "subBase2":
            result.append(Scalar[dtype](array.subBase2(indices[0], indices[1])))
        elif op == "offset2":
            result.append(Scalar[dtype](array.offset2(indices[0], indices[1])))
        elif op == "offset3":
            result.append(
                Scalar[dtype](array.offset3(indices[0], indices[1], indices[2]))
            )
        elif op == "offset4":
            result.append(
                Scalar[dtype](
                    array.offset4(
                        indices[0], indices[1], indices[2], indices[3]
                    )
                )
            )
        elif op == "calcFlatStride":
            for i in range(len(shape)):
                result.append(Scalar[dtype](array.calcFlatStride(i)))
        elif op == "getPlaneSlice":
            var view = array.getPlaneSlice(indices[0])
            for i in range(view.length):
                result.append(view.get(i))
        elif op == "fixedPrefixView":
            var prefix = List[Int]()
            prefix.append(indices[0])
            var view = array.fixedPrefixView(prefix)
            for i in range(view.length):
                result.append(view.getFlat(i))
        else:
            raise Error("UnknownOperation")
        return result^
    var matrix = lm.MatView(av.data, shape[0], shape[1])
    if suite == "rowops":
        if op == "swapRows":
            lm.rowops.swapRowsMatView(
                Int(py=inp["row_a"]), Int(py=inp["row_b"]), matrix
            )
        elif op == "scaleRow":
            lm.rowops.scaleRowMatView(
                Int(py=inp["row_a"]),
                Scalar[dtype](Float64(py=inp["alpha"])),
                matrix,
            )
        elif op == "addScaledRow":
            lm.rowops.addScaledRowMatView(
                Int(py=inp["row_a"]),
                Int(py=inp["row_b"]),
                Scalar[dtype](Float64(py=inp["alpha"])),
                matrix,
            )
        else:
            raise Error("UnknownOperation")
        for i in range(len(a)):
            result.append(matrix.data[i])
        return result^
    if suite == "matops":
        if op == "outer" or op == "outerAdd":
            var x = numbers[dtype](inp["x"])
            var xv = lm.VecView(Span(x))
            if op == "outer":
                lm.matops.outerMatView(xv, bv, matrix)
            else:
                lm.matops.outerAddMatView(
                    Scalar[dtype](Float64(py=inp["alpha"])), xv, bv, matrix
                )
            for i in range(len(a)):
                result.append(matrix.data[i])
            return result^
        if op == "trace":
            result.append(matrix.trace())
        elif op == "frobeniusNorm":
            result.append(lm.matops.frobeniusNormMatView(matrix))
        elif op == "maxAbs":
            result.append(lm.matops.maxAbsMatView(matrix))
        elif op == "allFinite":
            result.append(
                Scalar[dtype](Int(lm.matops.allFiniteMatView(matrix)))
            )
        else:
            var count = len(a)
            if op == "matmul":
                count = shape[0] * shape[2]
            elif op == "matvec":
                count = shape[0]
            for i in range(count):
                result.append(0)
            var output = Span(result)
            if op == "matmul":
                lm.matops.mulMatMatView(
                    matrix,
                    lm.MatView(bv.data, shape[1], shape[2]),
                    lm.MatView(output, shape[0], shape[2]),
                )
            elif op == "matvec":
                var x = numbers[dtype](inp["x"])
                lm.matops.mulVecMatView(
                    matrix,
                    lm.VecView(Span(x)),
                    lm.VecView(output),
                )
            elif op == "transpose":
                matrix.transpose(lm.MatView(output, shape[1], shape[0]))
            else:
                raise Error("UnknownOperation")
        return result^
    if suite == "linsolve":
        var n = shape[0]
        var work = List[Scalar[dtype]]()
        var rhs = List[Scalar[dtype]]()
        var perm = List[Int]()
        for i in range(n * n):
            work.append(0)
        for i in range(n):
            rhs.append(0)
            result.append(0)
            perm.append(0)
        var wm = lm.MatView(Span(work), n, n)
        var wv = lm.VecView(Span(rhs))
        var dst = lm.VecView(Span(result))
        var pp = Span(perm)
        if op == "gauss":
            lm.linsolve.solveGaussMatViewChecked(matrix, bv, wm, wv, dst)
        elif op == "lu":
            lm.linsolve.solveLUMatViewChecked(matrix, bv, wm, pp, wv, dst)
        elif op == "cholesky":
            lm.linsolve.solveCholeskyMatViewChecked(matrix, bv, wm, wv, dst)
        elif op == "luReuse":
            lm.lu.factorLUMatView(matrix, wm, pp)
            lm.lu.solveLUFactoredMatView(wm, pp, bv, wv, dst)
        elif op == "choleskyReuse":
            lm.cholesky.factorCholeskyMatView(matrix, wm)
            lm.cholesky.solveCholeskyFactoredMatView(wm, bv, wv, dst)
        else:
            raise Error("UnknownOperation")
        return result^
    raise Error("UnknownSuite")


def emit[dtype: DType](c: PythonObject, json: PythonObject, jsonl: Bool) raises:
    var record = Python.dict()
    record["id"] = c["id"]
    record["backend"] = "mojo"
    record["representation"] = c["representation"]
    try:
        var result = run[dtype](c)
        if Bool(py=c.__contains__("expected_error")):
            raise Error("ExpectedNumericalError")
        var expected = numbers[DType.float64](c["expected"])
        if len(result) != len(expected):
            raise Error("ResultLengthMismatch")
        var atol = Float64(py=c["tolerance"]["atol"])
        var rtol = Float64(py=c["tolerance"]["rtol"])
        for i in range(len(result)):
            var actual = Float64(result[i])
            var want = expected[i]
            if isnan(want):
                if not isnan(actual):
                    raise Error("NumericalMismatch")
            elif isinf(want):
                if actual != want:
                    raise Error("NumericalMismatch")
            elif not isfinite(actual) or abs(actual - want) > atol + rtol * abs(
                want
            ):
                raise Error("NumericalMismatch")
        var output = Python.list()
        for i in range(len(result)):
            var actual = Float64(result[i])
            if not isfinite(actual):
                var tag = Python.dict()
                tag["special"] = "nan" if isnan(actual) else (
                    "-inf" if actual < 0 else "inf"
                )
                output.append(tag)
            else:
                output.append(actual)
        record["status"] = "ok"
        record["result"] = output
    except e:
        record["status"] = "error"
        record["error"] = String(e)
        if not Bool(py=c.__contains__("expected_error")) or String(e) != String(
            py=c["expected_error"]
        ):
            print(String(py=json.dumps(record)))
            raise Error("UnexpectedNumericalError")
    if jsonl:
        print(String(py=json.dumps(record)))


def main() raises:
    var jsonl = False
    var silent = False
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--jsonl":
            jsonl = True
        elif args[i] == "--quiet":
            silent = True
        else:
            raise Error("InvalidArgument")
    var json = Python.import_module("json")
    var builtins = Python.import_module("builtins")
    var data = json.loads(builtins.open("tests/cases/all.json").read())
    for i in range(Int(py=data["cases"].__len__())):
        var c = data["cases"][i]
        if String(py=c["dtype"]) == "f32":
            emit[DType.float32](c, json, jsonl)
        else:
            emit[DType.float64](c, json, jsonl)

    if not jsonl and not silent:
        print(Int(py=data["cases"].__len__()), "shared Mojo cases passed")
