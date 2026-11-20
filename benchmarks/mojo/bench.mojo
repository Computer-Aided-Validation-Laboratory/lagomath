from std.sys import argv
import lagomath as lm
from std.collections import Span
from std.math import abs
from std.python import Python, PythonObject
from std.time import perf_counter_ns
from std.benchmark import keep


@inline(.never)
def kernel[
    N: Int, dtype: DType
](
    op: String,
    a: lm.Mat[N, N, dtype],
    b: lm.Vec[N, dtype],
    mut dst: lm.Mat[N, N, dtype],
) raises -> Scalar[dtype]:
    if op == "dot":
        return lm.vecops.dotVec(b, b)
    if op == "normL2":
        return lm.vecops.normL2Vec(b)
    if op == "axpy":
        var value = lm.vecops.axpyVec(Scalar[dtype](0.5), b, b)
        keep(value)
        return value.get(0)
    if op == "matvec":
        var value = lm.matops.mulVecMat(a, b)
        keep(value)
        return value.get(0)
    if op == "matmul":
        dst = lm.matops.mulMatMat(a, a)
        keep(dst)
        return dst.get(0, 0)
    if op == "outer":
        dst = lm.matops.outerMat(b, b)
        keep(dst)
        return dst.get(0, 0)
    if op == "outerAdd":
        dst = lm.matops.outerAddMat(Scalar[dtype](0.5), b, b, a)
        keep(dst)
        return dst.get(0, 0)
    if op == "transpose":
        dst = a.transpose()
        keep(dst)
        return dst.get(0, 0)
    if op == "rowops":
        dst = a.copy()
        lm.rowops.swapRowsMat(0, N - 1, dst)
        keep(dst)
        return dst.get(0, 0)
    if op == "gauss":
        var value = lm.linsolve.solveGaussMat(a, b)
        keep(value)
        return value.get(0)
    if op == "lu":
        var value = lm.linsolve.solveLUMat(a, b)
        keep(value)
        return value.get(0)
    if op == "cholesky":
        var value = lm.linsolve.solveCholeskyMat(a, b)
        keep(value)
        return value.get(0)
    if op == "factorLU":
        var value = lm.linsolve.factorLUMat(a)
        keep(value)
        return value.lu.get(0, 0)
    if op == "factorCholesky":
        var value = lm.linsolve.factorCholeskyMat(a)
        keep(value)
        return value.lower.get(0, 0)
    comptime if N == 2:
        var square = lm.Mat[2, 2, dtype]()
        for i in range(4):
            square.mat[i] = a.mat[i]
        if op == "det":
            return lm.Mat22Ops.det(square)
        if op == "inv":
            var value = lm.Mat22Ops.inv(square)
            keep(value)
            return value.get(0, 0)
    comptime if N == 3:
        var square = lm.Mat[3, 3, dtype]()
        for i in range(9):
            square.mat[i] = a.mat[i]
        if op == "det":
            return lm.Mat33Ops.det(square)
        if op == "inv":
            var value = lm.Mat33Ops.inv(square)
            keep(value)
            return value.get(0, 0)
    comptime if N == 4:
        var square = lm.Mat[4, 4, dtype]()
        for i in range(16):
            square.mat[i] = a.mat[i]
        if op == "det":
            return lm.Mat44Ops.det(square)
        if op == "inv":
            var value = lm.Mat44Ops.inv(square)
            keep(value)
            return value.get(0, 0)
    raise Error("UnknownOperation")


def record(
    c: PythonObject,
    sample: Int,
    ns: Int,
    runs: Int,
    checksum: Float64,
    json: PythonObject,
) raises:
    if not Bool(py=c["_jsonl"]):
        return
    var r = Python.dict()
    r["id"] = c["id"]
    r["backend"] = "mojo"
    r["sample"] = sample
    r["ns"] = ns
    r["runs"] = runs
    r["checksum"] = checksum
    print(String(py=json.dumps(r)))


def fixed[N: Int, dtype: DType](c: PythonObject, json: PythonObject) raises:
    var a = lm.Mat[N, N, dtype](0.125)
    for i in range(N):
        a.set(i, i, Scalar[dtype](N + 1))
    var b = lm.Vec[N, dtype](1)
    var dst = lm.Mat[N, N, dtype]()
    var batch = List[lm.Vec[N, dtype]]()
    for j in range(16):
        batch.append(b.mulScal(Scalar[dtype](j + 1)))
    var op = String(py=c["operation"])
    var family = String(py=c["input_family"])
    if family == "pivot_required":
        lm.rowops.swapRowsMat(0, N - 1, a)
    elif family == "signed_alternating":
        for r in range(N):
            b.set(r, Scalar[dtype](1 if r % 2 == 0 else -1))
            for col in range(N):
                if (r + col) % 2 == 1:
                    a.set(r, col, -a.get(r, col))
    elif family == "dense_random":
        for r in range(N):
            for col in range(N):
                if r != col:
                    a.set(
                        r,
                        col,
                        Scalar[dtype]((17 * r + 23 * col) % 29 - 14) / 64,
                    )
    var lu = lm.linsolve.LU[N, dtype]()
    var chol = lm.linsolve.Cholesky[N, dtype]()
    if op == "luReuse" or op == "luBatch16":
        lu = lm.linsolve.factorLUMat(a)
    if op == "choleskyReuse" or op == "choleskyBatch16":
        chol = lm.linsolve.factorCholeskyMat(a)
    var checksum: Scalar[dtype] = 0
    var runs = Int(py=c["runs_per_sample"])
    var warmup = Int(py=c["warmup"])
    var samples = Int(py=c["samples"])
    for sample in range(-1, samples):
        var count = runs
        if sample < 0:
            count = warmup
        var start = perf_counter_ns()
        for iteration in range(count):
            keep(a)
            keep(b)
            if op == "luReuse" or op == "luBatch16":
                keep(batch)
                keep(lu)
                for j in range(16 if op == "luBatch16" else 1):
                    var rb = batch[j].copy()
                    var value = lm.linsolve.solveLUFactoredMat(lu, rb)
                    keep(value)
                    checksum = value.get(0)
            elif op == "choleskyReuse" or op == "choleskyBatch16":
                keep(batch)
                keep(chol)
                for j in range(16 if op == "choleskyBatch16" else 1):
                    var rb = batch[j].copy()
                    var value = lm.linsolve.solveCholeskyFactoredMat(chol, rb)
                    keep(value)
                    checksum = value.get(0)
            else:
                checksum = kernel(op, a, b, dst)
            keep(checksum)
        var elapsed = perf_counter_ns() - start
        if sample >= 0:
            record(c, sample, Int(elapsed), runs, Float64(checksum), json)


def dynamic[dtype: DType](c: PythonObject, json: PythonObject) raises:
    var n = Int(py=c["shape"][0])
    var a = List[Scalar[dtype]](length=n, fill=1)
    var b = List[Scalar[dtype]](length=n, fill=1)
    var av = lm.VecView(Span(a))
    var bv = lm.VecView(Span(b))
    var output = List[Scalar[dtype]](length=n, fill=0)
    var dst = Span(output)
    var op = String(py=c["operation"])
    var checksum: Scalar[dtype] = 0
    var runs = Int(py=c["runs_per_sample"])
    var warmup = Int(py=c["warmup"])
    var samples = Int(py=c["samples"])
    for sample in range(-1, samples):
        var count = runs
        if sample < 0:
            count = warmup
        var start = perf_counter_ns()
        for iteration in range(count):
            keep(av)
            keep(bv)
            if op == "dot":
                checksum = lm.vecops.dotVecView(av, bv)
            elif op == "sum":
                checksum = lm.sliceops.sum(av.data)
            elif op == "max":
                checksum = lm.sliceops.max(av.data).val
            elif op == "add":
                lm.sliceops.add(av.data, bv.data, dst)
                keep(dst)
                checksum = dst[0]
            elif op == "mul":
                lm.sliceops.mul(av.data, bv.data, dst)
                keep(dst)
                checksum = dst[0]
            else:
                raise Error("UnknownOperation")
            keep(checksum)
        var elapsed = perf_counter_ns() - start
        if checksum != Scalar[dtype](Float64(py=c["expected_checksum"])):
            raise Error("InvalidChecksum")
        if sample >= 0:
            record(c, sample, Int(elapsed), runs, Float64(checksum), json)


def views[dtype: DType](c: PythonObject, json: PythonObject) raises:
    var n = Int(py=c["shape"][0])
    var a = List[Scalar[dtype]](length=n * n, fill=0.125)
    for i in range(n):
        a[i * n + i] = Scalar[dtype](n + 1)
    var b = List[Scalar[dtype]](length=n, fill=1)
    var work = List[Scalar[dtype]](length=n * n, fill=0)
    var rhs = List[Scalar[dtype]](length=n, fill=0)
    var dst = List[Scalar[dtype]](length=n, fill=0)
    var perm = List[Int](length=n, fill=0)
    var family = String(py=c["input_family"])
    if family == "pivot_required":
        for col in range(n):
            var tmp = a[col]
            a[col] = a[(n - 1) * n + col]
            a[(n - 1) * n + col] = tmp
    elif family == "signed_alternating":
        for r in range(n):
            b[r] = Scalar[dtype](1 if r % 2 == 0 else -1)
            for col in range(n):
                if (r + col) % 2 == 1:
                    a[r * n + col] = -a[r * n + col]
    elif family == "dense_random":
        for r in range(n):
            for col in range(n):
                if r != col:
                    a[r * n + col] = (
                        Scalar[dtype]((17 * r + 23 * col) % 29 - 14) / 64
                    )
    var a2 = a.copy()
    var b2 = b.copy()
    var matrix2 = lm.MatView(Span(a2), n, n)
    var vector2 = lm.VecView(Span(b2))
    var matrix = lm.MatView(Span(a), n, n)
    var vector = lm.VecView(Span(b))
    var wm = lm.MatView(Span(work), n, n)
    var wv = lm.VecView(Span(rhs))
    var outv = lm.VecView(Span(dst))
    var pp = Span(perm)
    var op = String(py=c["operation"])
    if op == "luReuse" or op == "luBatch16":
        lm.linsolve.factorLUMatView(matrix, wm, pp)
    if op == "choleskyReuse" or op == "choleskyBatch16":
        lm.linsolve.factorCholeskyMatView(matrix, wm)
    var batch = List[Scalar[dtype]](length=16 * n, fill=0)
    for j in range(16):
        for i in range(n):
            batch[j * n + i] = Scalar[dtype](j + 1)
    var batches = Span(batch)
    var checksum: Scalar[dtype] = 0
    var runs = Int(py=c["runs_per_sample"])
    var samples = Int(py=c["samples"])
    var warmup = Int(py=c["warmup"])
    for sample in range(-1, samples):
        var count = runs
        if sample < 0:
            count = warmup
        var start = perf_counter_ns()
        for iteration in range(count):
            keep(matrix)
            keep(vector)
            if op == "matvec":
                lm.matops.mulVecMatView(matrix, vector, outv)
                keep(outv)
                checksum = outv.get(0)
            elif op == "matmul":
                lm.matops.mulMatMatView(matrix, matrix2, wm)
                keep(wm)
                checksum = wm.get(0, 0)
            elif op == "outer":
                lm.matops.outerMatView(vector, vector2, wm)
                keep(wm)
                checksum = wm.get(0, 0)
            elif op == "outerAdd":
                for r in range(n):
                    for col in range(n):
                        wm.set(r, col, matrix.get(r, col))
                lm.matops.outerAddMatView(
                    Scalar[dtype](0.5), vector, vector2, wm
                )
                keep(wm)
                checksum = wm.get(0, 0)
            elif op == "transpose":
                matrix.transpose(wm)
                keep(wm)
                checksum = wm.get(0, 0)
            elif op == "rowops":
                for r in range(n):
                    for col in range(n):
                        wm.set(r, col, matrix.get(r, col))
                lm.rowops.swapRowsMatView(0, n - 1, wm)
                keep(wm)
                checksum = wm.get(0, 0)
            elif op == "factorLU":
                lm.linsolve.factorLUMatView(matrix, wm, pp)
                keep(wm)
                checksum = wm.get(0, 0)
            elif op == "factorCholesky":
                lm.linsolve.factorCholeskyMatView(matrix, wm)
                keep(wm)
                checksum = wm.get(0, 0)
            else:
                if op == "gauss":
                    lm.linsolve.solveGaussMatView(matrix, vector, wm, wv, outv)
                elif op == "lu":
                    lm.linsolve.solveLUMatView(matrix, vector, wm, pp, wv, outv)
                elif op == "cholesky":
                    lm.linsolve.solveCholeskyMatView(
                        matrix, vector, wm, wv, outv
                    )
                elif op == "luReuse" or op == "luBatch16":
                    keep(batches)
                    keep(wm)
                    for j in range(16 if op == "luBatch16" else 1):
                        var rb = lm.VecView(batches[j * n : (j + 1) * n])
                        lm.linsolve.solveLUFactoredMatView(wm, pp, rb, wv, outv)
                        keep(outv)
                elif op == "choleskyReuse" or op == "choleskyBatch16":
                    keep(batches)
                    keep(wm)
                    for j in range(16 if op == "choleskyBatch16" else 1):
                        var rb = lm.VecView(batches[j * n : (j + 1) * n])
                        lm.linsolve.solveCholeskyFactoredMatView(
                            wm, rb, wv, outv
                        )
                        keep(outv)
                else:
                    raise Error("UnknownOperation")
                keep(outv)
                checksum = outv.get(0)
        var elapsed = perf_counter_ns() - start
        if abs(
            Float64(checksum) - Float64(py=c["expected_checksum"])
        ) > 1e-4 * max(1.0, abs(Float64(py=c["expected_checksum"]))):
            raise Error("InvalidChecksum")
        if sample >= 0:
            record(c, sample, Int(elapsed), runs, Float64(checksum), json)


def ndarrays[dtype: DType](c: PythonObject, json: PythonObject) raises:
    var shape = List[Int]()
    var idx = List[Int]()
    var total = 1
    for i in range(Int(py=c["shape"].__len__())):
        var n = Int(py=c["shape"][i])
        shape.append(n)
        idx.append(n - 1)
        total *= n
    var a = List[Scalar[dtype]](length=total, fill=1)
    var array = lm.NDArray(Span(a), shape)
    var prefix = List[Int]()
    if len(shape) > 1:
        prefix.append(idx[0])
    var view = array.fixedPrefixView(prefix)
    var checksum: Scalar[dtype] = 0
    var op = String(py=c["operation"])
    var runs = Int(py=c["runs_per_sample"])
    var samples = Int(py=c["samples"])
    var warmup = Int(py=c["warmup"])
    for sample in range(-1, samples):
        var count = runs
        if sample < 0:
            count = warmup
        var start = perf_counter_ns()
        for iteration in range(count):
            keep(idx)
            keep(array)
            if op == "flatWrite":
                array.setFlat(total - 1, 2)
                keep(array.data)
                checksum = array.getFlat(total - 1)
            elif op == "indexedWrite":
                array.set(idx, 2)
                keep(array.data)
                checksum = array.get(idx)
            elif op == "getFlatIdx":
                checksum = Scalar[dtype](array.getFlatIdx(idx))
            elif op == "flatRead":
                checksum = array.getFlat(total - 1)
            elif op == "indexedRead":
                checksum = array.get(idx)
            elif op == "offset2":
                checksum = Scalar[dtype](array.offset2(idx[0], idx[1]))
            elif op == "offset3":
                checksum = Scalar[dtype](array.offset3(idx[0], idx[1], idx[2]))
            elif op == "offset4":
                checksum = Scalar[dtype](
                    array.offset4(idx[0], idx[1], idx[2], idx[3])
                )
            elif op == "planeBase":
                checksum = Scalar[dtype](array.planeBase(idx[0]))
            elif op == "subBase2":
                checksum = Scalar[dtype](array.subBase2(idx[0], idx[1]))
            elif op == "fixedPrefixView":
                checksum = view.getFlat(0)
            elif op == "stream":
                checksum = 0
                for j in range(total):
                    checksum += array.getFlat(j)
            else:
                raise Error("UnknownOperation")
            keep(checksum)
        var elapsed = perf_counter_ns() - start
        if Float64(checksum) != Float64(py=c["expected_checksum"]):
            raise Error("InvalidChecksum")
        if sample >= 0:
            record(c, sample, Int(elapsed), runs, Float64(checksum), json)


def rectangular[
    R: Int, C: Int, P: Int, dtype: DType
](c: PythonObject, json: PythonObject) raises:
    var a = lm.Mat[R, C, dtype](1)
    var b = lm.Mat[C, P, dtype](1)
    var dst = lm.Mat[R, P, dtype]()
    var trans = lm.Mat[C, R, dtype]()
    var rows = a.copy()
    var x = lm.Vec[C, dtype](1)
    var y = lm.Vec[R, dtype]()
    var av = lm.MatView(Span(a.mat), R, C)
    var bv = lm.MatView(Span(b.mat), C, P)
    var ov = lm.MatView(Span(dst.mat), R, P)
    var tv = lm.MatView(Span(trans.mat), C, R)
    var rv = lm.MatView(Span(rows.mat), R, C)
    var xv = lm.VecView(Span(x.vec))
    var yv = lm.VecView(Span(y.vec))
    var runs = Int(py=c["runs_per_sample"])
    var fixed = String(py=c["representation"]) == "Mat"
    var op = String(py=c["operation"])
    var checksum: Scalar[dtype] = 0
    for sample in range(-1, Int(py=c["samples"])):
        var count = runs if sample >= 0 else Int(py=c["warmup"])
        var start = perf_counter_ns()
        for iteration in range(count):
            keep(a)
            keep(b)
            keep(x)
            if op == "matmulRect":
                if fixed:
                    dst = lm.matops.mulMatMat(a, b)
                else:
                    lm.matops.mulMatMatView(av, bv, ov)
                keep(dst)
                checksum = dst.get(0, 0)
            elif op == "matvecRect":
                if fixed:
                    y = lm.matops.mulVecMat(a, x)
                else:
                    lm.matops.mulVecMatView(av, xv, yv)
                keep(y)
                checksum = y.get(0)
            elif op == "transposeRect":
                if fixed:
                    trans = a.transpose()
                else:
                    av.transpose(tv)
                keep(trans)
                checksum = trans.get(0, 0)
            else:
                rows = a.copy()
                if fixed:
                    lm.rowops.swapRowsMat(0, R - 1, rows)
                else:
                    lm.rowops.swapRowsMatView(0, R - 1, rv)
                keep(rows)
                checksum = rows.get(0, 0)
            keep(checksum)
        var elapsed = perf_counter_ns() - start
        if checksum != Scalar[dtype](Float64(py=c["expected_checksum"])):
            raise Error("InvalidChecksum")
        if sample >= 0:
            record(c, sample, Int(elapsed), runs, Float64(checksum), json)


def dispatch[dtype: DType](c: PythonObject, json: PythonObject) raises:
    if String(py=c["operation"]).endswith("Rect"):
        var r = Int(py=c["shape"][0])
        var k = Int(py=c["shape"][1])
        if r == 2:
            rectangular[2, 3, 4, dtype](c, json)
        elif r == 3:
            rectangular[3, 2, 3, dtype](c, json)
        elif r == 4:
            rectangular[4, 8, 3, dtype](c, json)
        elif r == 6:
            rectangular[6, 12, 8, dtype](c, json)
        elif r == 12:
            rectangular[12, 6, 3, dtype](c, json)
        else:
            raise Error("UnsupportedShape")
        return
    if String(py=c["representation"]) == "MatView":
        views[dtype](c, json)
        return
    if String(py=c["representation"]) == "NDArray":
        ndarrays[dtype](c, json)
        return
    if (
        String(py=c["representation"]) == "VecView"
        or String(py=c["representation"]) == "slice"
    ):
        dynamic[dtype](c, json)
        return
    var n = Int(py=c["shape"][0])
    if n == 2:
        fixed[2, dtype](c, json)
        return
    if n == 3:
        fixed[3, dtype](c, json)
        return
    if n == 4:
        fixed[4, dtype](c, json)
        return
    if n == 6:
        fixed[6, dtype](c, json)
        return
    if n == 8:
        fixed[8, dtype](c, json)
        return
    if n == 12:
        fixed[12, dtype](c, json)
        return
    if n == 16:
        fixed[16, dtype](c, json)
        return
    raise Error("UnsupportedShape")


def main() raises:
    var jsonl = False
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--jsonl":
            jsonl = True
        else:
            raise Error("InvalidArgument")
    var json = Python.import_module("json")
    var builtins = Python.import_module("builtins")
    var os = Python.import_module("os")
    var data = json.loads(
        builtins.open(
            os.environ.get(
                "LM_BENCH_MANIFEST", "benchmarks/cases/manifest.json"
            )
        ).read()
    )
    for i in range(Int(py=data["cases"].__len__())):
        var c = data["cases"][i]
        c["_jsonl"] = jsonl
        if String(py=c["dtype"]) == "f32":
            dispatch[DType.float32](c, json)
        else:
            dispatch[DType.float64](c, json)

    if not jsonl:
        print(
            Int(py=data["cases"].__len__()),
            "Mojo benchmark workloads passed; use --jsonl for timings",
        )
