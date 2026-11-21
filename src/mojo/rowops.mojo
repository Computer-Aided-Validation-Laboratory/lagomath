from matviewtype import MatView
from vecviewtype import VecView


def swapRowsMatView[
    dtype: DType, o_m: Origin[mut=True]
](a: Int, b: Int, m: MatView[dtype, o_m]):
    for c in range(m.cols_num):
        var tmp = m.get(a, c)
        m.set(a, c, m.get(b, c))
        m.set(b, c, tmp)


def scaleRowMatView[
    dtype: DType, o_m: Origin[mut=True]
](row: Int, alpha: Scalar[dtype], m: MatView[dtype, o_m]):
    for c in range(m.cols_num):
        m.set(row, c, m.get(row, c) * alpha)


def addScaledRowMatView[
    dtype: DType, o_m: Origin[mut=True]
](src: Int, dst: Int, alpha: Scalar[dtype], m: MatView[dtype, o_m]):
    for c in range(m.cols_num):
        m.set(dst, c, m.get(dst, c) + alpha * m.get(src, c))


from mattype import Mat
from vectype import Vec
from std.collections import Array, Span


def swapRowsMat[
    R: Int, C: Int, dtype: DType
](row_a: Int, row_b: Int, mut matrix: Mat[R, C, dtype]):
    for c in range(C):
        var temp = matrix.get(row_a, c)
        matrix.set(row_a, c, matrix.get(row_b, c))
        matrix.set(row_b, c, temp)


def scaleRowMat[
    R: Int, C: Int, dtype: DType
](row: Int, alpha: Scalar[dtype], mut matrix: Mat[R, C, dtype]):
    for c in range(C):
        matrix.set(row, c, matrix.get(row, c) * alpha)


def addScaledRowMat[
    R: Int, C: Int, dtype: DType
](src: Int, dst: Int, alpha: Scalar[dtype], mut matrix: Mat[R, C, dtype]):
    for c in range(C):
        matrix.set(dst, c, matrix.get(dst, c) + alpha * matrix.get(src, c))


def applyPermutationVec[
    N: Int, dtype: DType
](perm: Array[Int, N], src: Vec[N, dtype]) -> Vec[N, dtype]:
    var result = Vec[N, dtype]()
    for i in range(N):
        result.set(i, src.get(perm[i]))
    return result^


def applyPermutationVecView[
    dtype: DType,
    op: Origin[mut=True],
    osrc: Origin[mut=True],
    od: Origin[mut=True],
](perm: Span[Int, op], src: VecView[dtype, osrc], dst: VecView[dtype, od]):
    debug_assert(
        len(perm) == src.length and src.length == dst.length,
        "permutation length",
    )
    for i in range(len(perm)):
        dst.set(i, src.get(perm[i]))
