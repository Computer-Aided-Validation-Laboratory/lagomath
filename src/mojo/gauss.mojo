from contracts import checkSolve, checkFactor, checkFactored
from matviewtype import MatView
from vecviewtype import VecView
from matops import maxAbsMatView, allFiniteMatView
from vecops import allFiniteVecView
from rowops import swapRowsMatView
from std.math import abs, isfinite


def epsilon[dtype: DType]() -> Scalar[dtype]:
    comptime if dtype == DType.float32:
        return 1.1920928955078125e-7
    else:
        return 2.220446049250313e-16


def solveMatView[
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
    var scale = maxAbsMatView(a)
    if scale == 0:
        raise Error("SingularMatrix")
    for r in range(n):
        rhs.set(r, b.get(r) / scale)
        for c in range(n):
            work.set(r, c, a.get(r, c) / scale)
    var tol = epsilon[dtype]() * Scalar[dtype](n)
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
            swapRowsMatView(k, pivot, work)
            var temp = rhs.get(k)
            rhs.set(k, rhs.get(pivot))
            rhs.set(pivot, temp)
        for r in range(k + 1, n):
            var factor = work.get(r, k) / work.get(k, k)
            work.set(r, k, 0)
            for c in range(k + 1, n):
                work.set(r, c, work.get(r, c) - factor * work.get(k, c))
            rhs.set(r, rhs.get(r) - factor * rhs.get(k))
    for rev in range(n):
        var r = n - 1 - rev
        var value = rhs.get(r)
        for c in range(r + 1, n):
            value -= work.get(r, c) * dst.get(c)
        dst.set(r, value / work.get(r, r))


def solveMatViewChecked[
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
    if not allFiniteMatView(a) or not allFiniteVecView(b):
        raise Error("NonFiniteInput")
    solveMatView(a, b, work, rhs, dst)
