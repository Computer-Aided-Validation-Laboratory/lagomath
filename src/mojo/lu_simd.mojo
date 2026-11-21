import sliceops_simd as simd
from rowops_simd import subtractScaledRowMatViewSIMD
from contracts import checkSolve, checkFactor, checkFactored
from std.collections import Span
from matviewtype import MatView
from vecviewtype import VecView
from matops_simd import maxAbsMatViewSIMD, allFiniteMatViewSIMD
from vecops_simd import allFiniteVecViewSIMD
from rowops_simd import swapRowsMatViewSIMD
from gauss_simd import epsilonSIMD
from std.math import abs


def factorLUMatViewSIMD[
    dtype: DType,
    o_a: Origin,
    o_work: Origin[mut=True],
    o_perm: Origin[mut=True],
](
    a: MatView[dtype, o_a],
    work: MatView[dtype, o_work],
    perm: Span[Int, o_perm],
) raises:
    var n = a.rows_num
    debug_assert(
        n > 0
        and a.cols_num == n
        and work.rows_num == n
        and work.cols_num == n
        and len(perm) == n,
        "LU factor shape",
    )
    var scale = maxAbsMatViewSIMD(a)
    if scale == 0:
        raise Error("SingularMatrix")
    for r in range(n):
        perm[r] = r
    simd.divScalSIMD(a.data.as_imm()[0 : n * n], scale, work.data[0 : n * n])
    var tol = epsilonSIMD[dtype]() * Scalar[dtype](n)
    for k in range(n):
        var pivot = k
        var magnitude = abs(work.get(k, k))
        for r in range(k + 1, n):
            if abs(work.get(r, k)) > magnitude:
                magnitude = abs(work.get(r, k))
                pivot = r
        if magnitude <= tol:
            raise Error("SingularMatrix")
        if pivot != k:
            swapRowsMatViewSIMD(k, pivot, work)
            var temp = perm[k]
            perm[k] = perm[pivot]
            perm[pivot] = temp
        for r in range(k + 1, n):
            var factor = work.get(r, k) / work.get(k, k)
            work.set(r, k, factor)
            subtractScaledRowMatViewSIMD(k, r, factor, work, k + 1)
    for r in range(n):
        for c in range(r, n):
            work.set(r, c, work.get(r, c) * scale)


def solveLUFactoredMatViewSIMD[
    dtype: DType,
    o_work: Origin[mut=True],
    o_b: Origin,
    o_rhs: Origin[mut=True],
    o_dst: Origin[mut=True],
    o_perm: Origin[mut=True],
](
    work: MatView[dtype, o_work],
    perm: Span[Int, o_perm],
    b: VecView[dtype, o_b],
    rhs: VecView[dtype, o_rhs],
    dst: VecView[dtype, o_dst],
) raises:
    var n = work.rows_num
    debug_assert(
        n > 0
        and work.cols_num == n
        and len(perm) == n
        and b.length == n
        and rhs.length == n
        and dst.length == n,
        "LU solve shape",
    )
    for i in range(n):
        if work.get(i, i) == 0:
            raise Error("SingularMatrix")
    var scale: Scalar[dtype] = 0
    for r in range(n):
        for c in range(r, n):
            scale = max(scale, abs(work.get(r, c)))
    if scale == 0:
        raise Error("SingularMatrix")
    for r in range(n):
        var value = b.get(perm[r]) / scale
        value = simd.subtractProductsSIMD(
            value,
            work.data.as_imm()[r * n : r * n + r],
            rhs.data.as_imm()[0:r],
            Scalar[dtype](1),
        )
        rhs.set(r, value)
    for rev in range(n):
        var r = n - 1 - rev
        var value = rhs.get(r)
        value = simd.subtractProductsSIMD(
            value,
            work.data.as_imm()[r * n + r + 1 : (r + 1) * n],
            dst.data.as_imm()[r + 1 : n],
            scale,
        )
        dst.set(r, value / (work.get(r, r) / scale))


def solveLUMatViewSIMD[
    dtype: DType,
    o_a: Origin,
    o_b: Origin,
    o_work: Origin[mut=True],
    o_rhs: Origin[mut=True],
    o_dst: Origin[mut=True],
    o_perm: Origin[mut=True],
](
    a: MatView[dtype, o_a],
    b: VecView[dtype, o_b],
    work: MatView[dtype, o_work],
    perm: Span[Int, o_perm],
    rhs: VecView[dtype, o_rhs],
    dst: VecView[dtype, o_dst],
) raises:
    factorLUMatViewSIMD(a, work, perm)
    solveLUFactoredMatViewSIMD(work, perm, b, rhs, dst)


def solveLUMatViewCheckedSIMD[
    dtype: DType,
    o_a: Origin,
    o_b: Origin,
    o_work: Origin[mut=True],
    o_rhs: Origin[mut=True],
    o_dst: Origin[mut=True],
    o_perm: Origin[mut=True],
](
    a: MatView[dtype, o_a],
    b: VecView[dtype, o_b],
    work: MatView[dtype, o_work],
    perm: Span[Int, o_perm],
    rhs: VecView[dtype, o_rhs],
    dst: VecView[dtype, o_dst],
) raises:
    checkSolve(a, b, work, rhs, dst)
    var n = a.rows_num
    if (
        n <= 0
        or a.cols_num != n
        or b.length != n
        or work.rows_num != n
        or work.cols_num != n
        or rhs.length != n
        or dst.length != n
    ):
        raise Error("ShapeMismatch")
    if not allFiniteMatViewSIMD(a) or not allFiniteVecViewSIMD(b):
        raise Error("NonFiniteInput")
    if len(perm) != n:
        raise Error("ShapeMismatch")
    solveLUMatViewSIMD(a, b, work, perm, rhs, dst)


def factorLUMatViewCheckedSIMD[
    dtype: DType, oa: Origin, ow: Origin[mut=True], op: Origin[mut=True]
](a: MatView[dtype, oa], work: MatView[dtype, ow], perm: Span[Int, op]) raises:
    checkFactor(a, work)
    var n = a.rows_num
    if (
        n <= 0
        or a.cols_num != n
        or work.rows_num != n
        or work.cols_num != n
        or len(perm) != n
    ):
        raise Error("ShapeMismatch")
    if not allFiniteMatViewSIMD(a):
        raise Error("NonFiniteInput")
    factorLUMatViewSIMD(a, work, perm)


def solveLUFactoredMatViewCheckedSIMD[
    dtype: DType,
    ow: Origin[mut=True],
    op: Origin[mut=True],
    ob: Origin,
    orhs: Origin[mut=True],
    od: Origin[mut=True],
](
    work: MatView[dtype, ow],
    perm: Span[Int, op],
    b: VecView[dtype, ob],
    rhs: VecView[dtype, orhs],
    dst: VecView[dtype, od],
) raises:
    checkFactored(work, b, rhs, dst)
    var n = work.rows_num
    if (
        n <= 0
        or work.cols_num != n
        or len(perm) != n
        or b.length != n
        or rhs.length != n
        or dst.length != n
    ):
        raise Error("ShapeMismatch")
    if not allFiniteMatViewSIMD(work) or not allFiniteVecViewSIMD(b):
        raise Error("NonFiniteInput")
    for i in range(n):
        if perm[i] < 0 or perm[i] >= n:
            raise Error("InvalidPermutation")
        for j in range(i):
            if perm[j] == perm[i]:
                raise Error("InvalidPermutation")
    solveLUFactoredMatViewSIMD(work, perm, b, rhs, dst)
