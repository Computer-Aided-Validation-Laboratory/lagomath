from std.collections import Span
from valueindex import ValIdx
from std.math import sqrt, abs


struct VecView[dtype: DType, origin: Origin](Copyable, Movable):
    var data: Span[Scalar[Self.dtype], Self.origin]
    var length: Int

    def __init__(out self, data: Span[Scalar[Self.dtype], Self.origin]):
        self.data = data
        self.length = len(data)

    @staticmethod
    def init(data: Span[Scalar[Self.dtype], Self.origin]) -> Self:
        return Self(data)

    def get(self, i: Int) -> Scalar[Self.dtype]:
        debug_assert(i >= 0 and i < self.length, "vector index")
        return self.data[i]

    def set(self, i: Int, value: Scalar[Self.dtype]) where Self.origin.mut:
        debug_assert(i >= 0 and i < self.length, "vector index")
        comptime if Self.origin.mut:
            self.data[i : i + 1].fill(value)

    def fill(self, value: Scalar[Self.dtype]) where Self.origin.mut:
        for i in range(self.length):
            comptime if Self.origin.mut:
                self.data[i : i + 1].fill(value)

    def dot[
        o_other: Origin
    ](self, other: VecView[Self.dtype, o_other]) -> Scalar[Self.dtype]:
        debug_assert(self.length == other.length, "vector length")
        var value: Scalar[Self.dtype] = 0
        for i in range(self.length):
            value += self.data[i] * other.data[i]
        return value

    def normL1(self) -> Scalar[Self.dtype]:
        var value: Scalar[Self.dtype] = 0
        for i in range(self.length):
            value += abs(self.data[i])
        return value

    def normL2Square(self) -> Scalar[Self.dtype]:
        var value: Scalar[Self.dtype] = 0
        for i in range(self.length):
            value += self.data[i] * self.data[i]
        return value

    def normL2(self) -> Scalar[Self.dtype]:
        return sqrt(self.normL2Square())

    def sum(self) -> Scalar[Self.dtype]:
        debug_assert(self.length > 0, "nonempty sum")
        var value: Scalar[Self.dtype] = 0
        for i in range(self.length):
            value += self.data[i]
        return value

    def mean(self) -> Scalar[Self.dtype]:
        return self.sum() / Scalar[Self.dtype](self.length)

    def min(self) -> ValIdx[Self.dtype]:
        debug_assert(self.length > 0, "nonempty extremum")
        var value = self.get(0)
        var idx = 0
        for i in range(1, self.length):
            if self.get(i) < value:
                value = self.get(i)
                idx = i
        return ValIdx(value, idx)

    def max(self) -> ValIdx[Self.dtype]:
        debug_assert(self.length > 0, "nonempty extremum")
        var value = self.get(0)
        var idx = 0
        for i in range(1, self.length):
            if self.get(i) > value:
                value = self.get(i)
                idx = i
        return ValIdx(value, idx)

    def addInPlace[
        o_other: Origin
    ](self, other: VecView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(self.length == other.length, "vector length")
        for i in range(self.length):
            self.set(i, self.get(i) + other.get(i))

    def subInPlace[
        o_other: Origin
    ](self, other: VecView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(self.length == other.length, "vector length")
        for i in range(self.length):
            self.set(i, self.get(i) - other.get(i))

    def mulInPlace[
        o_other: Origin
    ](self, other: VecView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(self.length == other.length, "vector shape")
        for i in range(self.length):
            self.set(i, self.get(i) * other.get(i))

    def divInPlace[
        o_other: Origin
    ](self, other: VecView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(self.length == other.length, "vector shape")
        for i in range(self.length):
            self.set(i, self.get(i) / other.get(i))

    def mulScalInPlace(self, alpha: Scalar[Self.dtype]) where Self.origin.mut:
        for i in range(self.length):
            self.set(i, self.get(i) * alpha)

    def norm(self) -> Scalar[Self.dtype]:
        return self.normL2Square()

    def squaredNorm(self) -> Scalar[Self.dtype]:
        return self.normL2Square()

    def vecLen(self) -> Scalar[Self.dtype]:
        return self.normL2()

    def applyInPlace[
        func: def(Scalar[Self.dtype]) thin -> Scalar[Self.dtype]
    ](self) where Self.origin.mut:
        for i in range(self.length):
            self.set(i, func(self.get(i)))

    def vecPrint(self):
        for i in range(self.length):
            print(self.get(i))
