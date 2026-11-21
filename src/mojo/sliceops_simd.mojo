"""Opt-in SIMD over borrowed spans. Full blocks are bounds-checked by loop
limits; scalar tails never over-read. Dedicated in-place entrypoints preserve
tracked borrow rules. Lane reduction order can differ from scalar.
"""
from std.collections import Span
from std.math import abs, sqrt, isfinite
from valueindex import ValIdx


def dotSIMD[
    dtype: DType, oa: Origin, ob: Origin
](a: Span[Scalar[dtype], oa], b: Span[Scalar[dtype], ob]) -> Scalar[dtype]:
    debug_assert(len(a) == len(b), "dot length")
    comptime W = 8 if dtype == DType.float32 else 4
    var acc = SIMD[dtype, W](0)
    var i = 0
    while i + W <= len(a):
        acc += a.unsafe_ptr().unsafe_load[width=W](
            i
        ) * b.unsafe_ptr().unsafe_load[width=W](i)
        i += W
    var result = acc.reduce_add()
    while i < len(a):
        result += a[i] * b[i]
        i += 1
    return result


def sumSIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    debug_assert(len(a) > 0, "nonempty sum")
    comptime W = 8 if dtype == DType.float32 else 4
    var acc = SIMD[dtype, W](0)
    var i = 0
    while i + W <= len(a):
        acc += a.unsafe_ptr().unsafe_load[width=W](i)
        i += W
    var result = acc.reduce_add()
    while i < len(a):
        result += a[i]
        i += 1
    return result


def meanSIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return sumSIMD(a) / Scalar[dtype](len(a))


def normL1SIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var acc = SIMD[dtype, W](0)
    var i = 0
    while i + W <= len(a):
        acc += abs(a.unsafe_ptr().unsafe_load[width=W](i))
        i += W
    var result = acc.reduce_add()
    while i < len(a):
        result += abs(a[i])
        i += 1
    return result


def normL2SquareSIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return dotSIMD(a.as_imm(), a.as_imm())


def normL2SIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    return sqrt(dotSIMD(a.as_imm(), a.as_imm()))


def maxAbsSIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> Scalar[dtype]:
    debug_assert(len(a) > 0, "nonempty maximum")
    comptime W = 8 if dtype == DType.float32 else 4
    var acc = SIMD[dtype, W](0)
    var i = 0
    while i + W <= len(a):
        var v = abs(a.unsafe_ptr().unsafe_load[width=W](i))
        acc = v.gt(acc).select(v, acc)
        i += W
    var result = acc.reduce_max()
    while i < len(a):
        result = max(result, abs(a[i]))
        i += 1
    return result


def allFiniteSIMD[dtype: DType, oa: Origin](a: Span[Scalar[dtype], oa]) -> Bool:
    comptime W = 8 if dtype == DType.float32 else 4
    var i = 0
    while i + W <= len(a):
        if not isfinite(a.unsafe_ptr().unsafe_load[width=W](i)).reduce_and():
            return False
        i += W
    while i < len(a):
        if not isfinite(a[i]):
            return False
        i += 1
    return True


def extremaSIMD[
    maximum: Bool, dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> ValIdx[dtype]:
    debug_assert(len(a) > 0, "nonempty extremum")
    comptime W = 8 if dtype == DType.float32 else 4
    var value = a[0]
    var index = 0
    var i = 1
    while i + W <= len(a):
        var v = a.unsafe_ptr().unsafe_load[width=W](i)
        var mask = v.gt(SIMD[dtype, W](value)) if maximum else v.lt(
            SIMD[dtype, W](value)
        )
        if mask.reduce_or():
            for j in range(i, i + W):
                if a[j] > value if maximum else a[j] < value:
                    value = a[j]
                    index = j
        i += W
    while i < len(a):
        if a[i] > value if maximum else a[i] < value:
            value = a[i]
            index = i
        i += 1
    return ValIdx(value, index)


def minSIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> ValIdx[dtype]:
    return extremaSIMD[False](a)


def maxSIMD[
    dtype: DType, oa: Origin
](a: Span[Scalar[dtype], oa]) -> ValIdx[dtype]:
    return extremaSIMD[True](a)


def binarySIMD[
    op: Int, dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(b) and len(dst) == len(a), "elementwise length")
    comptime W = 8 if dtype == DType.float32 else 4
    var i = 0
    while i + W <= len(a):
        var x = a.unsafe_ptr().unsafe_load[width=W](i)
        var y = b.unsafe_ptr().unsafe_load[width=W](i)
        var value = x + y
        comptime if op == 1:
            value = x - y
        elif op == 2:
            value = x * y
        elif op == 3:
            value = x / y
        dst.unsafe_ptr().unsafe_store(i, value)
        i += W
    while i < len(a):
        comptime if op == 0:
            dst[i] = a[i] + b[i]
        elif op == 1:
            dst[i] = a[i] - b[i]
        elif op == 2:
            dst[i] = a[i] * b[i]
        else:
            dst[i] = a[i] / b[i]
        i += 1


def addSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    binarySIMD[0](a, b, dst)


def addCheckedSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    addSIMD(a, b, dst)


def subSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    binarySIMD[1](a, b, dst)


def subCheckedSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    subSIMD(a, b, dst)


def mulSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    binarySIMD[2](a, b, dst)


def mulCheckedSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    mulSIMD(a, b, dst)


def divSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    binarySIMD[3](a, b, dst)


def divCheckedSIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(b) or len(a) != len(dst):
        raise Error("LengthMismatch")
    divSIMD(a, b, dst)


def mulScalSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    alpha: Scalar[dtype],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(dst), "scale length")
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](alpha)
    var i = 0
    while i + W <= len(a):
        dst.unsafe_ptr().unsafe_store(
            i, a.unsafe_ptr().unsafe_load[width=W](i) * scale
        )
        i += W
    while i < len(a):
        dst[i] = a[i] * alpha
        i += 1


def mulScalCheckedSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    alpha: Scalar[dtype],
    dst: Span[Scalar[dtype], od],
) raises:
    if len(a) != len(dst):
        raise Error("LengthMismatch")
    mulScalSIMD(a, alpha, dst)


def fillSIMD[
    dtype: DType, od: Origin[mut=True]
](dst: Span[Scalar[dtype], od], value: Scalar[dtype]):
    comptime W = 8 if dtype == DType.float32 else 4
    var v = SIMD[dtype, W](value)
    var i = 0
    while i + W <= len(dst):
        dst.unsafe_ptr().unsafe_store(i, v)
        i += W
    while i < len(dst):
        dst[i] = value
        i += 1


def axpySIMD[
    dtype: DType, oa: Origin, ob: Origin, od: Origin[mut=True]
](
    alpha: Scalar[dtype],
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(b) and len(dst) == len(a), "axpy length")
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](alpha)
    var i = 0
    while i + W <= len(a):
        var v = scale * a.unsafe_ptr().unsafe_load[width=W](
            i
        ) + b.unsafe_ptr().unsafe_load[width=W](i)
        dst.unsafe_ptr().unsafe_store(i, v)
        i += W
    while i < len(a):
        dst[i] = alpha * a[i] + b[i]
        i += 1


def swapSIMD[
    dtype: DType, oa: Origin[mut=True], ob: Origin[mut=True]
](a: Span[Scalar[dtype], oa], b: Span[Scalar[dtype], ob]):
    debug_assert(len(a) == len(b), "swap length")
    comptime W = 8 if dtype == DType.float32 else 4
    var i = 0
    while i + W <= len(a):
        var x = a.unsafe_ptr().unsafe_load[width=W](i)
        var y = b.unsafe_ptr().unsafe_load[width=W](i)
        a.unsafe_ptr().unsafe_store(i, y)
        b.unsafe_ptr().unsafe_store(i, x)
        i += W
    while i < len(a):
        var v = a[i]
        a[i] = b[i]
        b[i] = v
        i += 1


def subtractProductsSIMD[
    dtype: DType, oa: Origin, ob: Origin
](
    initial: Scalar[dtype],
    a: Span[Scalar[dtype], oa],
    b: Span[Scalar[dtype], ob],
    divisor: Scalar[dtype],
) -> Scalar[dtype]:
    debug_assert(len(a) == len(b), "product length")
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](divisor)
    var value = initial
    var i = 0
    while i + W <= len(a):
        var products = (
            a.unsafe_ptr().unsafe_load[width=W](i) / scale
        ) * b.unsafe_ptr().unsafe_load[width=W](i)
        comptime for j in range(W):
            value -= products[j]
        i += W
    while i < len(a):
        value -= (a[i] / divisor) * b[i]
        i += 1
    return value


comptime normSIMD = normL2SquareSIMD
comptime vecLenSIMD = normL2SIMD


def axpyInPlaceSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](
    alpha: Scalar[dtype],
    a: Span[Scalar[dtype], oa],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(dst), "axpy length")
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](alpha)
    var i = 0
    while i + W <= len(a):
        var v = scale * a.unsafe_ptr().unsafe_load[width=W](
            i
        ) + dst.unsafe_ptr().unsafe_load[width=W](i)
        dst.unsafe_ptr().unsafe_store(i, v)
        i += W
    while i < len(a):
        dst[i] = alpha * a[i] + dst[i]
        i += 1


def mulScalInPlaceSIMD[
    dtype: DType, od: Origin[mut=True]
](alpha: Scalar[dtype], dst: Span[Scalar[dtype], od]):
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](alpha)
    var i = 0
    while i + W <= len(dst):
        dst.unsafe_ptr().unsafe_store(
            i, dst.unsafe_ptr().unsafe_load[width=W](i) * scale
        )
        i += W
    while i < len(dst):
        dst[i] *= alpha
        i += 1


def divScalSIMD[
    dtype: DType, oa: Origin, od: Origin[mut=True]
](
    a: Span[Scalar[dtype], oa],
    scale: Scalar[dtype],
    dst: Span[Scalar[dtype], od],
):
    debug_assert(len(a) == len(dst), "scale length")
    comptime W = 8 if dtype == DType.float32 else 4
    var v = SIMD[dtype, W](scale)
    var i = 0
    while i + W <= len(a):
        dst.unsafe_ptr().unsafe_store(
            i, a.unsafe_ptr().unsafe_load[width=W](i) / v
        )
        i += W
    while i < len(a):
        dst[i] = a[i] / scale
        i += 1


def subtractStridedProductsSIMD[
    dtype: DType, oa: Origin, ob: Origin
](
    initial: Scalar[dtype],
    a: Span[Scalar[dtype], oa],
    stride: Int,
    b: Span[Scalar[dtype], ob],
) -> Scalar[dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var value = initial
    var i = 0
    while i + W <= len(b):
        var v = SIMD[dtype, W](0)
        comptime for j in range(W):
            v[j] = a[(i + j) * stride]
        var products = v * b.unsafe_ptr().unsafe_load[width=W](i)
        comptime for j in range(W):
            value -= products[j]
        i += W
    while i < len(b):
        value -= a[i * stride] * b[i]
        i += 1
    return value
