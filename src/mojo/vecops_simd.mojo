"""Opt-in explicit SIMD vectors; fixed values keep inline storage."""
from std.collections import Span
from vectype import Vec
from vecviewtype import VecView
from valueindex import ValIdx
import sliceops_simd as core


def dotVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype], b: Vec[N, dtype]) -> Scalar[dtype]:
    return core.dotSIMD(Span(a.vec), Span(b.vec))


def dotVecViewSIMD[
    dtype: DType, oa: Origin, ob: Origin
](a: VecView[dtype, oa], b: VecView[dtype, ob]) -> Scalar[dtype]:
    return core.dotSIMD(a.data.as_imm(), b.data.as_imm())


def normL1VecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return core.normL1SIMD(Span(a.vec))


def normL1VecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Scalar[dtype]:
    return core.normL1SIMD(a.data.as_imm())


def normL2VecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return core.normL2SIMD(Span(a.vec))


def normL2VecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Scalar[dtype]:
    return core.normL2SIMD(a.data.as_imm())


def normL2SquareVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype]) -> Scalar[dtype]:
    return core.normL2SquareSIMD(Span(a.vec))


def normL2SquareVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Scalar[dtype]:
    return core.normL2SquareSIMD(a.data.as_imm())


def maxAbsVecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return core.maxAbsSIMD(Span(a.vec))


def maxAbsVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Scalar[dtype]:
    return core.maxAbsSIMD(a.data.as_imm())


def allFiniteVecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> Bool:
    return core.allFiniteSIMD(Span(a.vec))


def allFiniteVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Bool:
    return core.allFiniteSIMD(a.data.as_imm())


def sumVecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return core.sumSIMD(Span(a.vec))


def sumVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Scalar[dtype]:
    return core.sumSIMD(a.data.as_imm())


def meanVecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> Scalar[dtype]:
    return core.meanSIMD(Span(a.vec))


def meanVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> Scalar[dtype]:
    return core.meanSIMD(a.data.as_imm())


def minVecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> ValIdx[dtype]:
    return core.minSIMD(Span(a.vec))


def minVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> ValIdx[dtype]:
    return core.minSIMD(a.data.as_imm())


def maxVecSIMD[N: Int, dtype: DType](a: Vec[N, dtype]) -> ValIdx[dtype]:
    return core.maxSIMD(Span(a.vec))


def maxVecViewSIMD[
    dtype: DType, oa: Origin
](a: VecView[dtype, oa]) -> ValIdx[dtype]:
    return core.maxSIMD(a.data.as_imm())


def addVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype], b: Vec[N, dtype]) -> Vec[N, dtype]:
    var out = Vec[N, dtype]()
    core.addSIMD(Span(a.vec), Span(b.vec), Span(out.vec))
    return out^


def addVecViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: VecView[dtype, oa], b: VecView[dtype, ob], dst: VecView[dtype, od]):
    core.addSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def subVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype], b: Vec[N, dtype]) -> Vec[N, dtype]:
    var out = Vec[N, dtype]()
    core.subSIMD(Span(a.vec), Span(b.vec), Span(out.vec))
    return out^


def subVecViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: VecView[dtype, oa], b: VecView[dtype, ob], dst: VecView[dtype, od]):
    core.subSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def mulVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype], b: Vec[N, dtype]) -> Vec[N, dtype]:
    var out = Vec[N, dtype]()
    core.mulSIMD(Span(a.vec), Span(b.vec), Span(out.vec))
    return out^


def mulVecViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: VecView[dtype, oa], b: VecView[dtype, ob], dst: VecView[dtype, od]):
    core.mulSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def divVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype], b: Vec[N, dtype]) -> Vec[N, dtype]:
    var out = Vec[N, dtype]()
    core.divSIMD(Span(a.vec), Span(b.vec), Span(out.vec))
    return out^


def divVecViewSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](a: VecView[dtype, oa], b: VecView[dtype, ob], dst: VecView[dtype, od]):
    core.divSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def axpyVecSIMD[
    N: Int, dtype: DType
](alpha: Scalar[dtype], a: Vec[N, dtype], b: Vec[N, dtype]) -> Vec[N, dtype]:
    var out = Vec[N, dtype]()
    core.axpySIMD(alpha, Span(a.vec), Span(b.vec), Span(out.vec))
    return out^


def axpyVecViewSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](alpha: Scalar[dtype], a: VecView[dtype, oa], dst: VecView[dtype, od]):
    core.axpyInPlaceSIMD(alpha, a.data.as_imm(), dst.data)


def mulScalVecSIMD[
    N: Int, dtype: DType
](a: Vec[N, dtype], alpha: Scalar[dtype]) -> Vec[N, dtype]:
    var out = Vec[N, dtype]()
    core.mulScalSIMD(Span(a.vec), alpha, Span(out.vec))
    return out^


def mulScalVecViewSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](a: VecView[dtype, oa], alpha: Scalar[dtype], dst: VecView[dtype, od]):
    core.mulScalSIMD(a.data.as_imm(), alpha, dst.data)


def dotBatchSIMD[
    N: Int, dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(dst) * N and len(b) == len(a), "batch dot shape")
    comptime W = 8 if dtype == DType.float32 else 4
    var i = 0
    while i + W <= len(dst):
        var acc = SIMD[dtype, W](0)
        comptime for k in range(N):
            var av = SIMD[dtype, W](0)
            var bv = SIMD[dtype, W](0)
            comptime for j in range(W):
                av[j] = a[(i + j) * N + k]
                bv[j] = b[(i + j) * N + k]
            acc += av * bv
        dst.unsafe_ptr().unsafe_store(i, acc)
        i += W
    while i < len(dst):
        var value: Scalar[dtype] = 0
        comptime for k in range(N):
            value += a[i * N + k] * b[i * N + k]
        dst[i] = value
        i += 1


comptime normVecSIMD = normL2VecSIMD
comptime normVecViewSIMD = normL2VecViewSIMD
comptime squaredNormVecSIMD = normL2SquareVecSIMD
comptime squaredNormVecViewSIMD = normL2SquareVecViewSIMD
