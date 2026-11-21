"""Checked shapes. Tracked Span aliases are rejected by Mojo ownership checking.
Untracked foreign storage is checked at the C ABI boundary before wrapping.
"""
from matviewtype import MatView
from vecviewtype import VecView


def checkFactor[
    dtype: DType, oa: Origin, ow: Origin[mut=True]
](a: MatView[dtype, oa], work: MatView[dtype, ow]) raises:
    var n = a.rows_num
    if n <= 0 or a.cols_num != n or work.rows_num != n or work.cols_num != n:
        raise Error("ShapeMismatch")


def checkFactored[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    ow: Origin[mut=True],
    od: Origin[mut=True],
](
    a: MatView[dtype, oa],
    b: VecView[dtype, ob],
    work: VecView[dtype, ow],
    dst: VecView[dtype, od],
) raises:
    var n = a.rows_num
    if (
        n <= 0
        or a.cols_num != n
        or b.length != n
        or work.length != n
        or dst.length != n
    ):
        raise Error("ShapeMismatch")


def checkSolve[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    om: Origin[mut=True],
    ow: Origin[mut=True],
    od: Origin[mut=True],
](
    a: MatView[dtype, oa],
    b: VecView[dtype, ob],
    matrix: MatView[dtype, om],
    work: VecView[dtype, ow],
    dst: VecView[dtype, od],
) raises:
    checkFactor(a, matrix)
    checkFactored(a, b, work, dst)
