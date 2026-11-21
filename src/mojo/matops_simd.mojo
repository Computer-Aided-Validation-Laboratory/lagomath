"""Opt-in row-major SIMD. Product and transpose output must be disjoint
from inputs. All storage belongs to the caller.
"""
from std.collections import Span, Array
from vectype import Vec
from mattype import Mat
from vecviewtype import VecView
from matviewtype import MatView
import sliceops_simd as core


def mulMatSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    r: Int,
    k: Int,
    c: Int,
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(
        len(a) >= r * k and len(b) >= k * c and len(dst) >= r * c,
        "product shape",
    )
    comptime W = 8 if dtype == DType.float32 else 4
    var row = 0
    while row < r:
        var col = 0
        while col + W <= c:
            var acc = Array[SIMD[dtype, W], 4](fill=SIMD[dtype, W](0))
            for inner in range(k):
                var bv = b.unsafe_ptr().unsafe_load[width=W](inner * c + col)
                comptime for lane in range(4):
                    if row + lane < r:
                        acc[lane] += (
                            SIMD[dtype, W](a[(row + lane) * k + inner]) * bv
                        )
            comptime for lane in range(4):
                if row + lane < r:
                    dst.unsafe_ptr().unsafe_store(
                        (row + lane) * c + col, acc[lane]
                    )
            col += W
        while col < c:
            for rr in range(row, min(row + 4, r)):
                var v: Scalar[dtype] = 0
                for inner in range(k):
                    v += a[rr * k + inner] * b[inner * c + col]
                dst[rr * c + col] = v
            col += 1
        row += 4


def mulMatMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], b: MatView[dtype, ob], dst: MatView[dtype, od]):
    debug_assert(
        a.cols_num == b.rows_num
        and dst.rows_num == a.rows_num
        and dst.cols_num == b.cols_num,
        "product shape",
    )
    mulMatSIMD(
        a.rows_num,
        a.cols_num,
        b.cols_num,
        a.data.as_imm(),
        b.data.as_imm(),
        dst.data,
    )


def mulMatMatSIMD[
    R: Int, K: Int, C: Int, dtype: DType
](a: Mat[R, K, dtype], b: Mat[K, C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    mulMatSIMD(R, K, C, Span(a.mat), Span(b.mat), Span(dst.mat))
    return dst^


def mulVecMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], b: VecView[dtype, ob], dst: VecView[dtype, od]):
    debug_assert(
        a.cols_num == b.length and a.rows_num == dst.length, "matvec shape"
    )
    var aa = a.data.as_imm()
    var bb = b.data.as_imm()
    for r in range(a.rows_num):
        dst.set(r, core.dotSIMD(aa[r * a.cols_num : (r + 1) * a.cols_num], bb))


def mulVecMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], b: Vec[C, dtype]) -> Vec[R, dtype]:
    var dst = Vec[R, dtype]()
    var aa = Span(a.mat)
    var bb = Span(b.vec)
    for r in range(R):
        dst.set(r, core.dotSIMD(aa[r * C : (r + 1) * C], bb))
    return dst^


def transposeSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](r: Int, c: Int, a: Span[Scalar[dtype], oa], dst: Span[Scalar[dtype], od]):
    debug_assert(len(a) >= r * c and len(dst) >= r * c, "transpose shape")
    comptime W = 8 if dtype == DType.float32 else 4
    var rr = 0
    while rr + W <= r:
        var cc = 0
        while cc + W <= c:
            var tile = Array[SIMD[dtype, W], W](fill=SIMD[dtype, W](0))
            comptime for i in range(W):
                tile[i] = a.unsafe_ptr().unsafe_load[width=W]((rr + i) * c + cc)
            comptime for j in range(W):
                var v = SIMD[dtype, W](0)
                comptime for i in range(W):
                    v[i] = tile[i][j]
                dst.unsafe_ptr().unsafe_store((cc + j) * r + rr, v)
            cc += W
        for j in range(cc, c):
            for i in range(W):
                dst[j * r + rr + i] = a[(rr + i) * c + j]
        rr += W
    for i in range(rr, r):
        for j in range(c):
            dst[j * r + i] = a[i * c + j]


def transposeMatViewSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], dst: MatView[dtype, od]):
    debug_assert(
        a.rows_num == dst.cols_num and a.cols_num == dst.rows_num,
        "transpose shape",
    )
    transposeSIMD(a.rows_num, a.cols_num, a.data.as_imm(), dst.data)


def transposeMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype]) -> Mat[C, R, dtype]:
    var dst = Mat[C, R, dtype]()
    transposeSIMD(R, C, Span(a.mat), Span(dst.mat))
    return dst^


def outerMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: VecView[dtype, oa], b: VecView[dtype, ob], dst: MatView[dtype, od]):
    debug_assert(
        a.length == dst.rows_num and b.length == dst.cols_num, "outer shape"
    )
    for r in range(dst.rows_num):
        core.mulScalSIMD(
            b.data.as_imm(),
            a.get(r),
            dst.data[r * dst.cols_num : (r + 1) * dst.cols_num],
        )


def outerAddMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    alpha: Scalar[dtype],
    a: VecView[dtype, oa],
    b: VecView[dtype, ob],
    dst: MatView[dtype, od],
):
    debug_assert(
        a.length == dst.rows_num and b.length == dst.cols_num, "outer shape"
    )
    for r in range(dst.rows_num):
        core.axpyInPlaceSIMD(
            alpha * a.get(r),
            b.data.as_imm(),
            dst.data[r * dst.cols_num : (r + 1) * dst.cols_num],
        )


def outerMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Vec[R, dtype], b: Vec[C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    var out = Span(dst.mat)
    for r in range(R):
        core.mulScalSIMD(Span(b.vec), a.get(r), out[r * C : (r + 1) * C])
    return dst^


def outerAddMatSIMD[
    R: Int, C: Int, dtype: DType
](
    alpha: Scalar[dtype],
    a: Vec[R, dtype],
    b: Vec[C, dtype],
    m: Mat[R, C, dtype],
) -> Mat[R, C, dtype]:
    var dst = m.copy()
    var out = Span(dst.mat)
    for r in range(R):
        core.axpyInPlaceSIMD(
            alpha * a.get(r), Span(b.vec), out[r * C : (r + 1) * C]
        )
    return dst^


def maxAbsMatViewSIMD[
    dtype: DType, oa: Origin
](a: MatView[dtype, oa]) -> Scalar[dtype]:
    return core.maxAbsSIMD(a.data.as_imm()[0 : a.rows_num * a.cols_num])


def maxAbsMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype]) -> Scalar[dtype]:
    return core.maxAbsSIMD(Span(a.mat))


def allFiniteMatViewSIMD[
    dtype: DType, oa: Origin
](a: MatView[dtype, oa]) -> Bool:
    return core.allFiniteSIMD(a.data.as_imm()[0 : a.rows_num * a.cols_num])


def allFiniteMatSIMD[R: Int, C: Int, dtype: DType](a: Mat[R, C, dtype]) -> Bool:
    return core.allFiniteSIMD(Span(a.mat))


def frobeniusNormMatViewSIMD[
    dtype: DType, oa: Origin
](a: MatView[dtype, oa]) -> Scalar[dtype]:
    return core.normL2SIMD(a.data.as_imm()[0 : a.rows_num * a.cols_num])


def frobeniusNormMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype]) -> Scalar[dtype]:
    return core.normL2SIMD(Span(a.mat))


def addMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], b: MatView[dtype, ob], dst: MatView[dtype, od]):
    debug_assert(
        a.rows_num == b.rows_num
        and a.cols_num == b.cols_num
        and dst.rows_num == a.rows_num
        and dst.cols_num == a.cols_num,
        "elementwise shape",
    )
    core.addSIMD(
        a.data.as_imm()[0 : a.rows_num * a.cols_num],
        b.data.as_imm()[0 : a.rows_num * a.cols_num],
        dst.data[0 : a.rows_num * a.cols_num],
    )


def addMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], b: Mat[R, C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    core.addSIMD(Span(a.mat), Span(b.mat), Span(dst.mat))
    return dst^


def subMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], b: MatView[dtype, ob], dst: MatView[dtype, od]):
    debug_assert(
        a.rows_num == b.rows_num
        and a.cols_num == b.cols_num
        and dst.rows_num == a.rows_num
        and dst.cols_num == a.cols_num,
        "elementwise shape",
    )
    core.subSIMD(
        a.data.as_imm()[0 : a.rows_num * a.cols_num],
        b.data.as_imm()[0 : a.rows_num * a.cols_num],
        dst.data[0 : a.rows_num * a.cols_num],
    )


def subMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], b: Mat[R, C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    core.subSIMD(Span(a.mat), Span(b.mat), Span(dst.mat))
    return dst^


def mulElemWiseMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], b: MatView[dtype, ob], dst: MatView[dtype, od]):
    debug_assert(
        a.rows_num == b.rows_num
        and a.cols_num == b.cols_num
        and dst.rows_num == a.rows_num
        and dst.cols_num == a.cols_num,
        "elementwise shape",
    )
    core.mulSIMD(
        a.data.as_imm()[0 : a.rows_num * a.cols_num],
        b.data.as_imm()[0 : a.rows_num * a.cols_num],
        dst.data[0 : a.rows_num * a.cols_num],
    )


def mulElemWiseMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], b: Mat[R, C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    core.mulSIMD(Span(a.mat), Span(b.mat), Span(dst.mat))
    return dst^


def divElemWiseMatViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], b: MatView[dtype, ob], dst: MatView[dtype, od]):
    debug_assert(
        a.rows_num == b.rows_num
        and a.cols_num == b.cols_num
        and dst.rows_num == a.rows_num
        and dst.cols_num == a.cols_num,
        "elementwise shape",
    )
    core.divSIMD(
        a.data.as_imm()[0 : a.rows_num * a.cols_num],
        b.data.as_imm()[0 : a.rows_num * a.cols_num],
        dst.data[0 : a.rows_num * a.cols_num],
    )


def divElemWiseMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], b: Mat[R, C, dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    core.divSIMD(Span(a.mat), Span(b.mat), Span(dst.mat))
    return dst^


def mulScalMatViewSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](a: MatView[dtype, oa], alpha: Scalar[dtype], dst: MatView[dtype, od]):
    debug_assert(
        a.rows_num == dst.rows_num and a.cols_num == dst.cols_num, "scale shape"
    )
    core.mulScalSIMD(
        a.data.as_imm()[0 : a.rows_num * a.cols_num],
        alpha,
        dst.data[0 : a.rows_num * a.cols_num],
    )


def mulScalMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype], alpha: Scalar[dtype]) -> Mat[R, C, dtype]:
    var dst = Mat[R, C, dtype]()
    core.mulScalSIMD(Span(a.mat), alpha, Span(dst.mat))
    return dst^


def fillMatViewSIMD[
    dtype: DType, od: Origin[mut=True]
](a: MatView[dtype, od], value: Scalar[dtype]):
    core.fillSIMD(a.data[0 : a.rows_num * a.cols_num], value)


def traceMatViewSIMD[
    dtype: DType, oa: Origin
](a: MatView[dtype, oa]) -> Scalar[dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var n = min(a.rows_num, a.cols_num)
    var acc = SIMD[dtype, W](0)
    var i = 0
    while i + W <= n:
        var v = SIMD[dtype, W](0)
        comptime for j in range(W):
            v[j] = a.get(i + j, i + j)
        acc += v
        i += W
    var value = acc.reduce_add()
    while i < n:
        value += a.get(i, i)
        i += 1
    return value


def traceMatSIMD[
    R: Int, C: Int, dtype: DType
](a: Mat[R, C, dtype]) -> Scalar[dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var acc = SIMD[dtype, W](0)
    var i = 0
    while i + W <= min(R, C):
        var v = SIMD[dtype, W](0)
        comptime for j in range(W):
            v[j] = a.get(i + j, i + j)
        acc += v
        i += W
    var value = acc.reduce_add()
    while i < min(R, C):
        value += a.get(i, i)
        i += 1
    return value
