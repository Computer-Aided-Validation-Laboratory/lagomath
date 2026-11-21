from contracts import checkSolve, checkFactor, checkFactored
from matviewtype import MatView
from vecviewtype import VecView
from matops import maxAbsMatView, allFiniteMatView
from vecops import allFiniteVecView
from std.math import sqrt, isfinite


def factorCholeskyMatView[
    dtype: DType, o_a: Origin, o_work: Origin[mut=True]
](a: MatView[dtype, o_a], work: MatView[dtype, o_work]) raises:
    var n = a.rows_num
    var scale = maxAbsMatView(a)
    if scale == 0:
        raise Error("NotPositiveDefinite")
    work.fill(0)
    for r in range(n):
        for c in range(r + 1):
            var value = a.get(r, c) / scale
            for k in range(c):
                value -= work.get(r, k) * work.get(c, k)
            if r == c:
                if value <= 0 or not isfinite(value):
                    raise Error("NotPositiveDefinite")
                work.set(r, c, sqrt(value))
            else:
                work.set(r, c, value / work.get(c, c))
    var root_scale = sqrt(scale)
    for r in range(n):
        for c in range(r + 1):
            work.set(r, c, work.get(r, c) * root_scale)


def solveCholeskyFactoredMatView[
    dtype: DType,
    o_work: Origin[mut=True],
    o_b: Origin,
    o_rhs: Origin[mut=True],
    o_dst: Origin[mut=True],
](
    work: MatView[dtype, o_work],
    b: VecView[dtype, o_b],
    rhs: VecView[dtype, o_rhs],
    dst: VecView[dtype, o_dst],
) raises:
    var n = work.rows_num
    debug_assert(
        n > 0
        and work.cols_num == n
        and b.length == n
        and rhs.length == n
        and dst.length == n,
        "Cholesky solve shape",
    )
    for i in range(n):
        if work.get(i, i) <= 0:
            raise Error("NotPositiveDefinite")
    for r in range(n):
        var value = b.get(r)
        for c in range(r):
            value -= work.get(r, c) * rhs.get(c)
        rhs.set(r, value / work.get(r, r))
    for rev in range(n):
        var r = n - 1 - rev
        var value = rhs.get(r)
        for c in range(r + 1, n):
            value -= work.get(c, r) * dst.get(c)
        dst.set(r, value / work.get(r, r))


def solveCholeskyMatView[
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
    factorCholeskyMatView(a, work)
    solveCholeskyFactoredMatView(work, b, rhs, dst)


def solveCholeskyMatViewChecked[
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
    solveCholeskyMatView(a, b, work, rhs, dst)


def factorCholeskyMatViewChecked[
    dtype: DType, oa: Origin, ow: Origin[mut=True]
](a: MatView[dtype, oa], work: MatView[dtype, ow]) raises:
    checkFactor(a, work)
    var n = a.rows_num
    if n <= 0 or a.cols_num != n or work.rows_num != n or work.cols_num != n:
        raise Error("ShapeMismatch")
    if not allFiniteMatView(a):
        raise Error("NonFiniteInput")
    factorCholeskyMatView(a, work)


def solveCholeskyFactoredMatViewChecked[
    dtype: DType,
    ow: Origin[mut=True],
    ob: Origin,
    orhs: Origin[mut=True],
    od: Origin[mut=True],
](
    work: MatView[dtype, ow],
    b: VecView[dtype, ob],
    rhs: VecView[dtype, orhs],
    dst: VecView[dtype, od],
) raises:
    checkFactored(work, b, rhs, dst)
    var n = work.rows_num
    if (
        n <= 0
        or work.cols_num != n
        or b.length != n
        or rhs.length != n
        or dst.length != n
    ):
        raise Error("ShapeMismatch")
    if not allFiniteMatView(work) or not allFiniteVecView(b):
        raise Error("NonFiniteInput")
    solveCholeskyFactoredMatView(work, b, rhs, dst)
