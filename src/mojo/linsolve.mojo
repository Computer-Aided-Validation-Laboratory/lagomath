from matviewtype import MatView
from vecviewtype import VecView
from gauss import (
    solveMatView as solveGaussMatView,
    solveMatViewChecked as solveGaussMatViewChecked,
)
from lu import (
    factorLUMatView,
    solveLUFactoredMatView,
    solveLUMatView,
    solveLUMatViewChecked,
)
from cholesky import (
    factorCholeskyMatView,
    solveCholeskyFactoredMatView,
    solveCholeskyMatView,
    solveCholeskyMatViewChecked,
)

from std.collections import Array, Span
from vectype import Vec
from mattype import Mat


def solveGaussMat[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveGaussMatView(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveGaussMatChecked[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveGaussMatViewChecked(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveLUMat[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    var perm = Array[Int, N](fill=0)
    solveLUMatView(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        Span(perm),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveLUMatChecked[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    var perm = Array[Int, N](fill=0)
    solveLUMatViewChecked(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        Span(perm),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveCholeskyMat[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveCholeskyMatView(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def solveCholeskyMatChecked[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var a = matrix.copy()
    var b = vector.copy()
    var work = Mat[N, N, dtype]()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveCholeskyMatViewChecked(
        MatView(Span(a.mat), N, N),
        VecView(Span(b.vec)),
        MatView(Span(work.mat), N, N),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


struct LU[N: Int, dtype: DType](Copyable, Movable):
    var lu: Mat[Self.N, Self.N, Self.dtype]
    var permutation: Array[Int, Self.N]

    def __init__(out self):
        self.lu = Mat[Self.N, Self.N, Self.dtype]()
        self.permutation = Array[Int, Self.N](fill=0)


struct Cholesky[N: Int, dtype: DType](Copyable, Movable):
    var lower: Mat[Self.N, Self.N, Self.dtype]

    def __init__(out self):
        self.lower = Mat[Self.N, Self.N, Self.dtype]()


def factorLUMat[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype]) raises -> LU[N, dtype]:
    var a = matrix.copy()
    var result = LU[N, dtype]()
    factorLUMatView(
        MatView(Span(a.mat), N, N),
        MatView(Span(result.lu.mat), N, N),
        Span(result.permutation),
    )
    return result^


def solveLUFactoredMat[
    N: Int, dtype: DType
](factor: LU[N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var lu = factor.copy()
    var b = vector.copy()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveLUFactoredMatView(
        MatView(Span(lu.lu.mat), N, N),
        Span(lu.permutation),
        VecView(Span(b.vec)),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


def factorCholeskyMat[
    N: Int, dtype: DType
](matrix: Mat[N, N, dtype]) raises -> Cholesky[N, dtype]:
    var a = matrix.copy()
    var result = Cholesky[N, dtype]()
    factorCholeskyMatView(
        MatView(Span(a.mat), N, N),
        MatView(Span(result.lower.mat), N, N),
    )
    return result^


def solveCholeskyFactoredMat[
    N: Int, dtype: DType
](factor: Cholesky[N, dtype], vector: Vec[N, dtype]) raises -> Vec[N, dtype]:
    var chol = factor.copy()
    var b = vector.copy()
    var rhs = Vec[N, dtype]()
    var result = Vec[N, dtype]()
    solveCholeskyFactoredMatView(
        MatView(Span(chol.lower.mat), N, N),
        VecView(Span(b.vec)),
        VecView(Span(rhs.vec)),
        VecView(Span(result.vec)),
    )
    return result^


from lu import factorLUMatViewChecked, solveLUFactoredMatViewChecked
from cholesky import (
    factorCholeskyMatViewChecked,
    solveCholeskyFactoredMatViewChecked,
)
from matops import allFiniteMat
from vecops import allFiniteVec


def factorLUMatChecked[
    N: Int, dtype: DType
](a: Mat[N, N, dtype]) raises -> LU[N, dtype]:
    if not allFiniteMat(a):
        raise Error("NonFiniteInput")
    return factorLUMat(a)


def factorCholeskyMatChecked[
    N: Int, dtype: DType
](a: Mat[N, N, dtype]) raises -> Cholesky[N, dtype]:
    if not allFiniteMat(a):
        raise Error("NonFiniteInput")
    return factorCholeskyMat(a)


def solveLUFactoredMatChecked[
    N: Int, dtype: DType
](factor: LU[N, dtype], b: Vec[N, dtype]) raises -> Vec[N, dtype]:
    if not allFiniteMat(factor.lu) or not allFiniteVec(b):
        raise Error("NonFiniteInput")
    for i in range(N):
        if factor.permutation[i] < 0 or factor.permutation[i] >= N:
            raise Error("InvalidPermutation")
        for j in range(i):
            if factor.permutation[j] == factor.permutation[i]:
                raise Error("InvalidPermutation")
    return solveLUFactoredMat(factor, b)


def solveCholeskyFactoredMatChecked[
    N: Int, dtype: DType
](factor: Cholesky[N, dtype], b: Vec[N, dtype]) raises -> Vec[N, dtype]:
    if not allFiniteMat(factor.lower) or not allFiniteVec(b):
        raise Error("NonFiniteInput")
    return solveCholeskyFactoredMat(factor, b)
