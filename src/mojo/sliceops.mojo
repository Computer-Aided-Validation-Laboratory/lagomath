from std.collections import Span
from vecviewtype import VecView
from valueindex import ValIdx
from std.math import sqrt


def dot[
    dtype: DType, oa: Origin, ob: Origin
](a: Span[Scalar[dtype], oa], b: Span[Scalar[dtype], ob]) -> Scalar[dtype]:
    return VecView(a).dot(VecView(b))


def normL1[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return VecView(a).normL1()


def normL2[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return VecView(a).normL2()


def normL2Square[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return VecView(a).normL2Square()


def sum[dtype: DType, oa: Origin](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return VecView(a).sum()


def mean[dtype: DType, oa: Origin](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return VecView(a).mean()


def min[dtype: DType, oa: Origin](a: Span[Scalar[dtype], oa]) -> ValIdx[dtype]:
    return VecView(a).min()


def max[dtype: DType, oa: Origin](a: Span[Scalar[dtype], oa]) -> ValIdx[dtype]:
    return VecView(a).max()


def add[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(b) and len(a) == len(dst), "slice length")
    for i in range(len(a)):
        dst[i : i + 1].fill(a[i] + b[i])


def addChecked[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    add(a, b, dst)


def sub[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(b) and len(a) == len(dst), "slice length")
    for i in range(len(a)):
        dst[i : i + 1].fill(a[i] - b[i])


def subChecked[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    sub(a, b, dst)


def mul[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(b) and len(a) == len(dst), "slice length")
    for i in range(len(a)):
        dst[i : i + 1].fill(a[i] * b[i])


def mulChecked[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    mul(a, b, dst)


def div[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(b) and len(a) == len(dst), "slice length")
    for i in range(len(a)):
        dst[i : i + 1].fill(a[i] / b[i])


def divChecked[
    dtype: DType,
    oa: Origin,
    ob: Origin,
    od: Origin[mut=True],
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    div(a, b, dst)


from std.math import exp as native_exp, ceil


def exp[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    return native_exp(value)


def apply[
    dtype: DType,
    od: Origin[mut=True],
    os: Origin,
    func: def(Scalar[dtype]) thin -> Scalar[dtype],
](dst: Span[Scalar[dtype], od], source: Span[Scalar[dtype], os]):
    debug_assert(len(dst) >= len(source), "apply destination length")
    for i in range(len(source)):
        dst[i : i + 1].fill(func(source[i]))


def rangeLen(start: Float64, stop: Float64, step: Float64) -> Int:
    return Int(ceil((stop - start) / step))


def slicePrint[dtype: DType, o: Origin](value: Span[Scalar[dtype], o]):
    VecView(value).vecPrint()


def norm[dtype: DType, o: Origin](a: Span[Scalar[dtype], o]) -> Scalar[dtype]:
    return normL2Square(a)


def vecLen[dtype: DType, o: Origin](a: Span[Scalar[dtype], o]) -> Scalar[dtype]:
    return normL2(a)


def mulScal[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    alpha: Scalar[dtype],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(dst), "slice length")
    for i in range(len(a)):
        dst[i : i + 1].fill(a[i] * alpha)


def mulScalChecked[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    alpha: Scalar[dtype],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(dst):
        raise Error("LengthMismatch")
    mulScal(a, alpha, dst)
