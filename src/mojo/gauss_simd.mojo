import sliceops_simd as simd
from rowops_simd import subtractScaledRowMatViewSIMD
from contracts import checkSolve, checkFactor, checkFactored
from matviewtype import MatView
from vecviewtype import VecView
from matops_simd import maxAbsMatViewSIMD, allFiniteMatViewSIMD
from vecops_simd import allFiniteVecViewSIMD
from rowops_simd import swapRowsMatViewSIMD
from std.math import abs, isfinite


def epsilonSIMD[dtype: DType]() -> Scalar[dtype]:
    comptime if dtype == DType.float32:
        return 1.1920928955078125e-7
    else:
        return 2.220446049250313e-16


def solveMatViewSIMD[
    dtype: DType,
    o_a: Origin,
    o_b: Origin,
    o_work: Origin[mut=True],
    o_rhs: Origin[mut=True],
    o_dst: Origin[mut=True],
](
    a: MatView[dtype, o_a],
    b: VecView[dtype, o_b],
    work: MatView[dtype, o_work],
    rhs: VecView[dtype, o_rhs],
    dst: VecView[dtype, o_dst],
) raises:
    var n = a.rows_num
    debug_assert(
        n > 0
        and a.cols_num == n
        and b.length == n
        and work.rows_num == n
        and work.cols_num == n
        and rhs.length == n
        and dst.length == n,
        "solve shape",
    )
    var scale = maxAbsMatViewSIMD(a)
    if scale == 0:
        raise Error("SingularMatrix")
    simd.divScalSIMD(b.data.as_imm(), scale, rhs.data)
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
            var temp = rhs.get(k)
            rhs.set(k, rhs.get(pivot))
            rhs.set(pivot, temp)
        for r in range(k + 1, n):
            var factor = work.get(r, k) / work.get(k, k)
            work.set(r, k, 0)
            subtractScaledRowMatViewSIMD(k, r, factor, work, k + 1)
            rhs.set(r, rhs.get(r) - factor * rhs.get(k))
    for rev in range(n):
        var r = n - 1 - rev
        var value = rhs.get(r)
        value = simd.subtractProductsSIMD(
            value,
            work.data.as_imm()[r * n + r + 1 : (r + 1) * n],
            dst.data.as_imm()[r + 1 : n],
            Scalar[dtype](1),
        )
        dst.set(r, value / work.get(r, r))


def solveMatViewCheckedSIMD[
    dtype: DType,
    o_a: Origin,
    o_b: Origin,
    o_work: Origin[mut=True],
    o_rhs: Origin[mut=True],
    o_dst: Origin[mut=True],
](
    a: MatView[dtype, o_a],
    b: VecView[dtype, o_b],
    work: MatView[dtype, o_work],
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
    solveMatViewSIMD(a, b, work, rhs, dst)
