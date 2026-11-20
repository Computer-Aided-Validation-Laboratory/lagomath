from std.collections import Span
from valueindex import ValIdx
from std.collections import Array
from std.math import sqrt, abs


struct Vec[N: Int, dtype: DType](Copyable, Movable):
    var vec: Array[Scalar[Self.dtype], Self.N]

    def __init__(out self, fill: Scalar[Self.dtype] = 0):
        self.vec = Array[Scalar[Self.dtype], Self.N](fill=fill)

    @staticmethod
    def initFill(value: Scalar[Self.dtype]) -> Self:
        return Self(value)

    @staticmethod
    def initZeros() -> Self:
        return Self(0)

    @staticmethod
    def initOnes() -> Self:
        return Self(1)

    def get(self, i: Int) -> Scalar[Self.dtype]:
        return self.vec[i]

    def set(mut self, i: Int, value: Scalar[Self.dtype]):
        self.vec[i] = value

    def dot(self, other: Self) -> Scalar[Self.dtype]:
        var value: Scalar[Self.dtype] = 0
        for i in range(Self.N):
            value += self.vec[i] * other.vec[i]
        return value

    def normL1(self) -> Scalar[Self.dtype]:
        var value: Scalar[Self.dtype] = 0
        for i in range(Self.N):
            value += abs(self.vec[i])
        return value

    def normL2Square(self) -> Scalar[Self.dtype]:
        return self.dot(self)

    def normL2(self) -> Scalar[Self.dtype]:
        return sqrt(self.normL2Square())

    def sum(self) -> Scalar[Self.dtype]:
        debug_assert(Self.N > 0, "nonempty sum")
        var value: Scalar[Self.dtype] = 0
        for i in range(Self.N):
            value += self.vec[i]
        return value

    def mean(self) -> Scalar[Self.dtype]:
        return self.sum() / Scalar[Self.dtype](Self.N)

    def min(self) -> ValIdx[Self.dtype]:
        debug_assert(Self.N > 0, "nonempty extremum")
        var value = self.get(0)
        var idx = 0
        for i in range(1, Self.N):
            if self.get(i) < value:
                value = self.get(i)
                idx = i
        return ValIdx(value, idx)

    def max(self) -> ValIdx[Self.dtype]:
        debug_assert(Self.N > 0, "nonempty extremum")
        var value = self.get(0)
        var idx = 0
        for i in range(1, Self.N):
            if self.get(i) > value:
                value = self.get(i)
                idx = i
        return ValIdx(value, idx)

    def add(self, other: Self) -> Self:
        var result = Self()
        for i in range(Self.N):
            result.set(i, self.get(i) + other.get(i))
        return result^

    def sub(self, other: Self) -> Self:
        var result = Self()
        for i in range(Self.N):
            result.set(i, self.get(i) - other.get(i))
        return result^

    def mulScal(self, alpha: Scalar[Self.dtype]) -> Self:
        var result = Self()
        for i in range(Self.N):
            result.set(i, self.get(i) * alpha)
        return result^

    def norm(self) -> Scalar[Self.dtype]:
        return self.normL2Square()

    def squaredNorm(self) -> Scalar[Self.dtype]:
        return self.normL2Square()

    def vecLen(self) -> Scalar[Self.dtype]:
        return self.normL2()

    @staticmethod
    def initSlice[o: Origin](source: Span[Scalar[Self.dtype], o]) -> Self:
        debug_assert(len(source) >= Self.N, "source length")
        var result = Self()
        for i in range(Self.N):
            result.vec[i] = source[i]
        return result^

    def x(self) -> Scalar[Self.dtype]:
        return self.get(0)

    def y(self) -> Scalar[Self.dtype]:
        return self.get(1)

    def z(self) -> Scalar[Self.dtype]:
        return self.get(2)

    def w(self) -> Scalar[Self.dtype]:
        return self.get(3)

    def apply[
        func: def(Scalar[Self.dtype]) thin -> Scalar[Self.dtype]
    ](self) -> Self:
        var result = Self()
        for i in range(Self.N):
            result.set(i, func(self.get(i)))
        return result^

    def vecPrint(self):
        for i in range(Self.N):
            print(self.get(i))


struct Vec3Ops:
    @staticmethod
    def cross[
        dtype: DType
    ](a: Vec[3, dtype], b: Vec[3, dtype]) -> Vec[3, dtype]:
        var result = Vec[3, dtype]()
        result.set(0, a.get(1) * b.get(2) - a.get(2) * b.get(1))
        result.set(1, a.get(0) * b.get(2) - a.get(2) * b.get(0))
        result.set(2, a.get(0) * b.get(1) - a.get(1) * b.get(0))
        return result^


comptime Vec2T[dtype: DType] = Vec[2, dtype]
comptime Vec3T[dtype: DType] = Vec[3, dtype]
comptime Vec2f = Vec[2, DType.float64]
comptime Vec3f = Vec[3, DType.float64]


def initVec2[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Vec[2, dtype]:
    var result = Vec[2, dtype]()
    result.set(0, x)
    result.set(1, y)
    return result^


def initVec3[
    dtype: DType
](x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]) -> Vec[3, dtype]:
    var result = Vec[3, dtype]()
    result.set(0, x)
    result.set(1, y)
    result.set(2, z)
    return result^


struct Vec3SliceOps:
    @staticmethod
    def max[
        dtype: DType, o: Origin
    ](values: Span[Vec[3, dtype], o], index: Int) -> Scalar[dtype]:
        debug_assert(
            len(values) > 0 and index >= 0 and index < 3, "component reduction"
        )
        var result = values[0].get(index)
        for i in range(1, len(values)):
            result = max(result, values[i].get(index))
        return result

    @staticmethod
    def min[
        dtype: DType, o: Origin
    ](values: Span[Vec[3, dtype], o], index: Int) -> Scalar[dtype]:
        debug_assert(
            len(values) > 0 and index >= 0 and index < 3, "component reduction"
        )
        var result = values[0].get(index)
        for i in range(1, len(values)):
            result = min(result, values[i].get(index))
        return result
