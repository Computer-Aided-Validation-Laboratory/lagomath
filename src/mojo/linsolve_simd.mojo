from linsolve import LU, Cholesky
import sliceops_simd as simd
from rowops_simd import subtractScaledRowMatViewSIMD
from matviewtype import MatView
from vecviewtype import VecView
from gauss_simd import (
    solveMatViewSIMD as solveGaussMatViewSIMD,
    solveMatViewCheckedSIMD as solveGaussMatViewCheckedSIMD,
)
from lu_simd import (
    factorLUMatViewSIMD,
    solveLUFactoredMatViewSIMD,
    solveLUMatViewSIMD,
    solveLUMatViewCheckedSIMD,
)
from cholesky_simd import (
    factorCholeskyMatViewSIMD,
    solveCholeskyFactoredMatViewSIMD,
    solveCholeskyMatViewSIMD,
    solveCholeskyMatViewCheckedSIMD,
)

from std.collections import Array, Span
from vectype import Vec
from mattype import Mat


def solveGaussMatSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveGaussMatViewSIMD(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveGaussMatCheckedSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveGaussMatViewCheckedSIMD(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveLUMatSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    var perm = Array[Int, N](fill=0)
    solveLUMatViewSIMD(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        Span(perm),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveLUMatCheckedSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    var perm = Array[Int, N](fill=0)
    solveLUMatViewCheckedSIMD(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        Span(perm),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveCholeskyMatSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveCholeskyMatViewSIMD(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveCholeskyMatCheckedSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveCholeskyMatViewCheckedSIMD(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def factorLUMatSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype]) raises -> LU[N, dtype]:
    var a = matrix.copy()
    var result = LU[N, dtype]()
    factorLUMatViewSIMD(
        MatView(Span(a.mat), N, N),
        MatView(Span(result.lu.mat), N, N),
        Span(result.permutation),
    )
    return result^


def solveLUFactoredMatSIMD[
    N: Int, dtype: DType
](factor: LU[N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var lu = factor.copy()
    var b = vector.copy()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveLUFactoredMatViewSIMD(
        MatView(Span(lu.lu.mat), N, N),
        Span(lu.permutation),
        VecView(Span(b.vec)),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def factorCholeskyMatSIMD[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype]) raises -> Cholesky[N, dtype]:
    var a = matrix.copy()
    var result = Cholesky[N, dtype]()
    factorCholeskyMatViewSIMD(
        MatView(Span(a.mat), N, N),
        MatView(Span(result.lower.mat), N, N),
    )
    return result^


def solveCholeskyFactoredMatSIMD[
    N: Int, dtype: DType
](factor: Cholesky[N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var chol = factor.copy()
    var b = vector.copy()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveCholeskyFactoredMatViewSIMD(
        MatView(Span(chol.lower.mat), N, N),
        VecView(Span(b.vec)),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


from lu_simd import (
    factorLUMatViewCheckedSIMD,
    solveLUFactoredMatViewCheckedSIMD,
)
from cholesky_simd import (
    factorCholeskyMatViewCheckedSIMD,
    solveCholeskyFactoredMatViewCheckedSIMD,
)
from matops_simd import allFiniteMatSIMD
from vecops_simd import allFiniteVecSIMD


def factorLUMatCheckedSIMD[
    N: Int, dtype: DType
](a: Mat[N, N, dtype]) raises -> LU[N, dtype]:
    if not allFiniteMatSIMD(a):
        raise Error("NonFiniteInput")
    return factorLUMatSIMD(a)


def factorCholeskyMatCheckedSIMD[
    N: Int, dtype: DType
](a: Mat[N, N, dtype]) raises -> Cholesky[N, dtype]:
    if not allFiniteMatSIMD(a):
        raise Error("NonFiniteInput")
    return factorCholeskyMatSIMD(a)


def solveLUFactoredMatCheckedSIMD[
    N: Int, dtype: DType
](factor: LU[N, dtype], b: Vec[N, dtype]) raises -> Vec[N, dtype]:
    if not allFiniteMatSIMD(factor.lu) or not allFiniteVecSIMD(b):
        raise Error("NonFiniteInput")
    for i in range(N):
        if factor.permutation[i] < 0 or factor.permutation[i] >= N:
            raise Error("InvalidPermutation")
        for j in range(i):
            if factor.permutation[j] == factor.permutation[i]:
                raise Error("InvalidPermutation")
    return solveLUFactoredMatSIMD(factor, b)


def solveCholeskyFactoredMatCheckedSIMD[
    N: Int, dtype: DType
](factor: Cholesky[N, dtype], b: Vec[N, dtype]) raises -> Vec[N, dtype]:
    if not allFiniteMatSIMD(factor.lower) or not allFiniteVecSIMD(b):
        raise Error("NonFiniteInput")
    return solveCholeskyFactoredMatSIMD(factor, b)


comptime LUSIMD = LU
comptime CholeskySIMD = Cholesky
