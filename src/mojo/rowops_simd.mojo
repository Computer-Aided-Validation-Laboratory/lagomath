"""Explicit SIMD row updates, including rows sharing one matrix origin."""
from std.collections import Span
from matviewtype import MatView
from mattype import Mat


def rowOpSIMD[
    op: Int, dtype: DType, om: Origin[mut=True]
](
    m: Span[Scalar[dtype], om],
    cols: Int,
    src: Int,
    dst: Int,
    alpha: Scalar[dtype],
    start: Int = 0,
):
    debug_assert(
        cols > 0
        and src >= 0
        and dst >= 0
        and (src + 1) * cols <= len(m)
        and (dst + 1) * cols <= len(m)
        and start >= 0
        and start <= cols,
        "row shape",
    )
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](alpha)
    var c = start
    while c + W <= cols:
        var x = m.unsafe_ptr().unsafe_load[width=W](src * cols + c)
        var y = m.unsafe_ptr().unsafe_load[width=W](dst * cols + c)
        comptime if op == 0:
            m.unsafe_ptr().unsafe_store(src * cols + c, y)
            m.unsafe_ptr().unsafe_store(dst * cols + c, x)
        elif op == 1:
            m.unsafe_ptr().unsafe_store(dst * cols + c, y * scale)
        elif op == 2:
            m.unsafe_ptr().unsafe_store(dst * cols + c, y + scale * x)
        else:
            m.unsafe_ptr().unsafe_store(dst * cols + c, y - scale * x)
        c += W
    while c < cols:
        comptime if op == 0:
            var temp = m[src * cols + c]
            m[src * cols + c] = m[dst * cols + c]
            m[dst * cols + c] = temp
        elif op == 1:
            m[dst * cols + c] *= alpha
        elif op == 2:
            m[dst * cols + c] += alpha * m[src * cols + c]
        else:
            m[dst * cols + c] -= alpha * m[src * cols + c]
        c += 1


def swapRowsMatViewSIMD[
    dtype: DType, om: Origin[mut=True]
](a: Int, b: Int, m: MatView[dtype, om]):
    rowOpSIMD[0](m.data, m.cols_num, a, b, Scalar[dtype](0))


def swapRowsMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Int, b: Int, mut m: Mat[R, C, dtype]):
    rowOpSIMD[0](Span(m.mat), C, a, b, Scalar[dtype](0))


def scaleRowMatViewSIMD[
    dtype: DType, om: Origin[mut=True]
](row: Int, alpha: Scalar[dtype], m: MatView[dtype, om]):
    rowOpSIMD[1](m.data, m.cols_num, row, row, alpha)


def scaleRowMatSIMD[
    R: Int, C: Int, dtype: DType
](row: Int, alpha: Scalar[dtype], mut m: Mat[R, C, dtype]):
    rowOpSIMD[1](Span(m.mat), C, row, row, alpha)


def addScaledRowMatViewSIMD[
    dtype: DType, om: Origin[mut=True]
](src: Int, dst: Int, alpha: Scalar[dtype], m: MatView[dtype, om]):
    rowOpSIMD[2](m.data, m.cols_num, src, dst, alpha)


def addScaledRowMatSIMD[
    R: Int, C: Int, dtype: DType
](src: Int, dst: Int, alpha: Scalar[dtype], mut m: Mat[R, C, dtype]):
    rowOpSIMD[2](Span(m.mat), C, src, dst, alpha)


def subtractScaledRowMatViewSIMD[
    dtype: DType, om: Origin[mut=True]
](src: Int, dst: Int, alpha: Scalar[dtype], m: MatView[dtype, om], start: Int):
    rowOpSIMD[3](m.data, m.cols_num, src, dst, alpha, start)


from vectype import Vec
from vecviewtype import VecView
from std.collections import Array


def applyPermutationVecSIMD[
    N: Int, dtype: DType
](perm: Array[Int, N], a: Vec[N, dtype]) -> Vec[N, dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var dst = Vec[N, dtype]()
    var i = 0
    while i + W <= N:
        var v = SIMD[dtype, W](0)
        comptime for j in range(W):
            debug_assert(
                perm[i + j] >= 0 and perm[i + j] < N, "permutation index"
            )
            v[j] = a.get(perm[i + j])
        Span(dst.vec).unsafe_ptr().unsafe_store(i, v)
        i += W
    while i < N:
        dst.set(i, a.get(perm[i]))
        i += 1
    return dst^


def applyPermutationVecViewSIMD[
    dtype: DType, op: Origin[mut=True], oa: Origin, od: Origin[mut=True]
](perm: Span[Int, op], a: VecView[dtype, oa], dst: VecView[dtype, od]):
    debug_assert(
        len(perm) == a.length and dst.length == a.length, "permutation length"
    )
    comptime W = 8 if dtype == DType.float32 else 4
    var i = 0
    while i + W <= a.length:
        var v = SIMD[dtype, W](0)
        comptime for j in range(W):
            v[j] = a.get(perm[i + j])
        dst.data.unsafe_ptr().unsafe_store(i, v)
        i += W
    while i < a.length:
        dst.set(i, a.get(perm[i]))
        i += 1
