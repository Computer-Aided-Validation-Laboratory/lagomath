from vectype import Vec
from mattype import Mat
from vecviewtype import VecView
from matviewtype import MatView
from std.math import sqrt, abs, isfinite


def mulVecMatView[
    dtype: DType, o_a: Origin, o_x: Origin, o_dst: Origin[mut=True]
](a: MatView[dtype, o_a], x: VecView[dtype, o_x], dst: VecView[dtype, o_dst]):
    debug_assert(
        a.cols_num == x.length and a.rows_num == dst.length, "matvec shape"
    )
    for r in range(a.rows_num):
        var value: Scalar[dtype] = 0
        for c in range(a.cols_num):
            value += a.get(r, c) * x.get(c)
        dst.set(r, value)


def mulMatMatView[
    dtype: DType, o_a: Origin, o_b: Origin, o_dst: Origin[mut=True]
](a: MatView[dtype, o_a], b: MatView[dtype, o_b], dst: MatView[dtype, o_dst]):
    debug_assert(
        a.cols_num == b.rows_num
        and a.rows_num == dst.rows_num
        and b.cols_num == dst.cols_num,
        "matmul shape",
    )
    for r in range(a.rows_num):
        for c in range(b.cols_num):
            var value: Scalar[dtype] = 0
            for k in range(a.cols_num):
                value += a.get(r, k) * b.get(k, c)
            dst.set(r, c, value)


def mulVecMat[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], x: Vec[C, dtype]) -> Vec[R, dtype]:
    var dst = Vec[R, dtype]()
    for r in range(R):
        var value: Scalar[dtype] = 0
        for c in range(C):
            value += a.get(r, c) * x.get(c)
        dst.set(r, value)
    return dst^


def mulMatMat[
    R: Int, C: Int, P: Int, dtype: DType
](a: Mat[R, C, dtype], b: Mat[C, P, dtype]) -> Mat[R, P, dtype]:
    var dst = Mat[R, P, dtype]()
    for r in range(R):
        for c in range(P):
            var value: Scalar[dtype] = 0
            for k in range(C):
                value += a.get(r, k) * b.get(k, c)
            dst.set(r, c, value)
    return dst^


def maxAbsMat[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype]) -> Scalar[dtype]:
    comptime assert R > 0 and C > 0

    var value: Scalar[dtype] = 0
    for i in range(R * C):
        value = max(value, abs(a.mat[i]))
    return value


def frobeniusNormMat[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype]) -> Scalar[dtype]:
    var value: Scalar[dtype] = 0
    for i in range(R * C):
        value += a.mat[i] * a.mat[i]
    return sqrt(value)


def allFiniteMat[R: Int, C: Int, dtype: DType](a: Mat[R, C, dtype]) -> Bool:
    for i in range(R * C):
        if not isfinite(a.mat[i]):
            return False
    return True


def maxAbsMatView[
    dtype: DType, o_a: Origin
](a: MatView[dtype, o_a]) -> Scalar[dtype]:
    debug_assert(a.rows_num > 0 and a.cols_num > 0, "nonempty matrix maximum")

    var value: Scalar[dtype] = 0
    for i in range(a.rows_num * a.cols_num):
        value = max(value, abs(a.data[i]))
    return value


def frobeniusNormMatView[
    dtype: DType, o_a: Origin
](a: MatView[dtype, o_a]) -> Scalar[dtype]:
    var value: Scalar[dtype] = 0
    for i in range(a.rows_num * a.cols_num):
        value += a.data[i] * a.data[i]
    return sqrt(value)


def allFiniteMatView[dtype: DType, o_a: Origin](a: MatView[dtype, o_a]) -> Bool:
    for i in range(a.rows_num * a.cols_num):
        if not isfinite(a.data[i]):
            return False
    return True


def outerMatView[
    dtype: DType, o_a: Origin, o_b: Origin, o_dst: Origin[mut=True]
](a: VecView[dtype, o_a], b: VecView[dtype, o_b], dst: MatView[dtype, o_dst]):
    for r in range(a.length):
        for c in range(b.length):
            dst.set(r, c, a.get(r) * b.get(c))


def outerAddMatView[
    dtype: DType, o_a: Origin, o_b: Origin, o_dst: Origin[mut=True]
](
    alpha: Scalar[dtype],
    a: VecView[dtype, o_a],
    b: VecView[dtype, o_b],
    dst: MatView[dtype, o_dst],
):
    for r in range(a.length):
        for c in range(b.length):
            dst.set(r, c, dst.get(r, c) + alpha * a.get(r) * b.get(c))


def outerMat[
    R: Int, C: Int, dtype: DType
](a: Vec[R, dtype], b: Vec[C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    for r in range(R):
        for c in range(C):
            dst.set(r, c, a.get(r) * b.get(c))
    return dst^


def outerAddMat[
    R: Int, C: Int, dtype: DType
](
    alpha: Scalar[dtype],
    a: Vec[R, dtype],
    b: Vec[C, dtype],
    matrix: Mat[R, C, dtype],
) -> Mat[R, C, dtype]:
    var dst = matrix.copy()
    for r in range(R):
        for c in range(C):
            dst.set(r, c, dst.get(r, c) + alpha * a.get(r) * b.get(c))
    return dst^
