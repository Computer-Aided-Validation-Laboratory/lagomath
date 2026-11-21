"""SIMD across independent cofactors of inline 2x2/3x3/4x4 matrices."""
from mattype import Mat
from std.collections import Array
from std.math import abs, isfinite


def cofactorsSIMD[
    N: Int, dtype: DType
](a: Mat[N, N, dtype], start: Int, stride: Int) -> SIMD[
    dtype, 8 if dtype == DType.float32 else 4
]:
    comptime assert N >= 2 and N <= 4
    comptime W = 8 if dtype == DType.float32 else 4
    var minor = Array[SIMD[dtype, W], (N - 1) * (N - 1)](fill=SIMD[dtype, W](0))
    var signs = SIMD[dtype, W](1)
    comptime for lane in range(W):
        var p = start + lane * stride
        if p < N * N:
            var remove_r = p % N
            var remove_c = p // N
            signs[lane] = Scalar[dtype](
                1 if (remove_r + remove_c) % 2 == 0 else -1
            )
            comptime for r in range(N - 1):
                comptime for c in range(N - 1):
                    minor[r * (N - 1) + c][lane] = a.get(
                        r + Int(r >= remove_r), c + Int(c >= remove_c)
                    )
    comptime if N == 2:
        return signs * minor[0]
    elif N == 3:
        return signs * (minor[0] * minor[3] - minor[1] * minor[2])
    else:
        return signs * (
            minor[0] * (minor[4] * minor[8] - minor[5] * minor[7])
            - minor[1] * (minor[3] * minor[8] - minor[5] * minor[6])
            + minor[2] * (minor[3] * minor[7] - minor[4] * minor[6])
        )


def detMatSIMD[N: Int, dtype: DType](a: Mat[N, N, dtype]) -> Scalar[dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var first = SIMD[dtype, W](0)
    comptime for i in range(N):
        first[i] = a.get(0, i)
    var terms = cofactorsSIMD(a, 0, N) * first
    var value = terms[0]
    comptime for i in range(1, N):
        value += terms[i]
    return value


def inverseSIMD[
    N: Int, dtype: DType
](a: Mat[N, N, dtype], det: Scalar[dtype]) -> Mat[N, N, dtype]:
    comptime W = 8 if dtype == DType.float32 else 4
    var scale = SIMD[dtype, W](1 / det)
    var dst = Mat[N, N, dtype]()
    var start = 0
    while start < N * N:
        var v = cofactorsSIMD(a, start, 1) * scale
        comptime for lane in range(W):
            if start + lane < N * N:
                dst.set((start + lane) // N, (start + lane) % N, v[lane])
        start += W
    return dst^


def invMatSIMD[N: Int, dtype: DType](a: Mat[N, N, dtype]) -> Mat[N, N, dtype]:
    var det = detMatSIMD(a)
    debug_assert(isfinite(det) and det != 0, "invertible matrix")
    return inverseSIMD(a, det)


def invMatCheckedSIMD[
    N: Int, dtype: DType
](a: Mat[N, N, dtype], min_abs_det: Scalar[dtype]) raises -> Mat[N, N, dtype]:
    var det = detMatSIMD(a)
    if not isfinite(det) or abs(det) <= min_abs_det:
        raise Error("SingularMatrix")
    return inverseSIMD(a, det)
