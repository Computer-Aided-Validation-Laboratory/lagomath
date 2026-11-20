from vectype import Vec
from vecviewtype import VecView
from std.math import abs, isfinite


def dotVec[
    N: Int, dtype: DType
](a: Vec[N, dtype], b: Vec[N, dtype]) -> Scalar[dtype]:
    return a.dot(b)


def normL1Vec[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return a.normL1()


def normL2Vec[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return a.normL2()


def normL2SquareVec[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return a.normL2Square()


def maxAbsVec[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    comptime assert N > 0

    var value: Scalar[dtype] = 0
    for i in range(N):
        value = max(value, abs(a.get(i)))
    return value


def allFiniteVec[N: Int, dtype: DType](a: Vec[N, dtype]) -> Bool:
    for i in range(N):
        if not isfinite(a.get(i)):
            return False
    return True


def axpyVec[
    N: Int, dtype: DType
](alpha: Scalar[dtype], x: Vec[N, dtype], y: Vec[N, dtype]) -> Vec[N, dtype]:
    var dst = Vec[N, dtype]()
    for i in range(N):
        dst.set(i, alpha * x.get(i) + y.get(i))
    return dst^


def dotVecView[
    dtype: DType, o_a: Origin, o_b: Origin
](a: VecView[dtype, o_a], b: VecView[dtype, o_b]) -> Scalar[dtype]:
    return a.dot(b)


def normL1VecView[
    dtype: DType, o_a: Origin
](a: VecView[dtype, o_a]) -> Scalar[dtype]:
    return a.normL1()


def normL2VecView[
    dtype: DType, o_a: Origin
](a: VecView[dtype, o_a]) -> Scalar[dtype]:
    return a.normL2()


def normL2SquareVecView[
    dtype: DType, o_a: Origin
](a: VecView[dtype, o_a]) -> Scalar[dtype]:
    return a.normL2Square()


def maxAbsVecView[
    dtype: DType, o_a: Origin
](a: VecView[dtype, o_a]) -> Scalar[dtype]:
    debug_assert(a.length > 0, "nonempty maximum")

    var value: Scalar[dtype] = 0
    for i in range(a.length):
        value = max(value, abs(a.get(i)))
    return value


def allFiniteVecView[dtype: DType, o_a: Origin](a: VecView[dtype, o_a]) -> Bool:
    for i in range(a.length):
        if not isfinite(a.get(i)):
            return False
    return True


def axpyVecView[
    dtype: DType, o_x: Origin, o_y: Origin[mut=True]
](alpha: Scalar[dtype], x: VecView[dtype, o_x], y: VecView[dtype, o_y]):
    debug_assert(x.length == y.length, "axpy length")
    for i in range(x.length):
        y.set(i, alpha * x.get(i) + y.get(i))


# Historical canonical aliases retain their documented norm meanings.
comptime squaredNormVec = normL2SquareVec
comptime squaredNormVecView = normL2SquareVecView
comptime normVec = normL2Vec
comptime normVecView = normL2VecView
