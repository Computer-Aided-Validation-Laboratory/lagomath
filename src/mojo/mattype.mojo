from std.collections import Span
from vectype import Vec
from std.collections import Array


struct Mat[R: Int, C: Int, dtype: DType](Copyable, Movable):
    var mat: Array[Scalar[Self.dtype], Self.R * Self.C]

    def __init__(out self, fill: Scalar[Self.dtype] = 0):
        self.mat = Array[Scalar[Self.dtype], Self.R * Self.C](fill=fill)

    @staticmethod
    def initZeros() -> Self:
        return Self(0)

    @staticmethod
    def initFill(value: Scalar[Self.dtype]) -> Self:
        return Self(value)

    @staticmethod
    def initIdentity() -> Self:
        var result = Self(0)
        for i in range(min(Self.R, Self.C)):
            result.mat[i * Self.C + i] = 1
        return result^

    def get(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        debug_assert(
            r >= 0 and r < Self.R and c >= 0 and c < Self.C, "matrix index"
        )
        return self.mat[r * Self.C + c]

    def set(mut self, r: Int, c: Int, value: Scalar[Self.dtype]):
        debug_assert(
            r >= 0 and r < Self.R and c >= 0 and c < Self.C, "matrix index"
        )
        self.mat[r * Self.C + c] = value

    def transpose(self) -> Mat[Self.C, Self.R, Self.dtype]:
        var result = Mat[Self.C, Self.R, Self.dtype]()
        for r in range(Self.R):
            for c in range(Self.C):
                result.set(c, r, self.get(r, c))
        return result^

    def trace(self) -> Scalar[Self.dtype]:
        var value: Scalar[Self.dtype] = 0
        for i in range(min(Self.R, Self.C)):
            value += self.get(i, i)
        return value

    def add(self, other: Self) -> Self:
        var result = Self()
        for i in range(Self.R * Self.C):
            result.mat[i] = self.mat[i] + other.mat[i]
        return result^

    def sub(self, other: Self) -> Self:
        var result = Self()
        for i in range(Self.R * Self.C):
            result.mat[i] = self.mat[i] - other.mat[i]
        return result^

    @staticmethod
    def initOnes() -> Self:
        return Self(1)

    @staticmethod
    def initDiag(value: Scalar[Self.dtype]) -> Self:
        var result = Self()
        for i in range(min(Self.R, Self.C)):
            result.set(i, i, value)
        return result^

    def mulScal(self, alpha: Scalar[Self.dtype]) -> Self:
        var result = Self()
        for i in range(Self.R * Self.C):
            result.mat[i] = self.mat[i] * alpha
        return result^

    def getRowVec(self, row: Int) -> Vec[Self.C, Self.dtype]:
        var result = Vec[Self.C, Self.dtype]()
        for c in range(Self.C):
            result.set(c, self.get(row, c))
        return result^

    def getColVec(self, col: Int) -> Vec[Self.R, Self.dtype]:
        var result = Vec[Self.R, Self.dtype]()
        for r in range(Self.R):
            result.set(r, self.get(r, col))
        return result^

    def getSubMat[
        rows: Int, cols: Int
    ](self, row: Int, col: Int) -> Mat[rows, cols, Self.dtype]:
        var result = Mat[rows, cols, Self.dtype]()
        for r in range(rows):
            for c in range(cols):
                result.set(r, c, self.get(row + r, col + c))
        return result^

    def insertRowVec[
        N: Int
    ](mut self, row: Int, col: Int, vector: Vec[N, Self.dtype]):
        for i in range(N):
            self.set(row, col + i, vector.get(i))

    def insertColVec[
        N: Int
    ](mut self, col: Int, row: Int, vector: Vec[N, Self.dtype]):
        for i in range(N):
            self.set(row + i, col, vector.get(i))

    def insertSubMat[
        rows: Int, cols: Int
    ](mut self, row: Int, col: Int, matrix: Mat[rows, cols, Self.dtype]):
        for r in range(rows):
            for c in range(cols):
                self.set(row + r, col + c, matrix.get(r, c))

    def mulVec(
        self, vector: Vec[Self.C, Self.dtype]
    ) -> Vec[Self.R, Self.dtype]:
        var result = Vec[Self.R, Self.dtype]()
        for r in range(Self.R):
            var value: Scalar[Self.dtype] = 0
            for c in range(Self.C):
                value += self.get(r, c) * vector.get(c)
            result.set(r, value)
        return result^

    def mulMatRect[
        P: Int
    ](self, other: Mat[Self.C, P, Self.dtype]) -> Mat[Self.R, P, Self.dtype]:
        var result = Mat[Self.R, P, Self.dtype]()
        for r in range(Self.R):
            for c in range(P):
                var value: Scalar[Self.dtype] = 0
                for k in range(Self.C):
                    value += self.get(r, k) * other.get(k, c)
                result.set(r, c, value)
        return result^

    def mulMat(self, other: Self) -> Self:
        comptime assert Self.R == Self.C
        var result = Self()
        for r in range(Self.R):
            for c in range(Self.C):
                var value: Scalar[Self.dtype] = 0
                for k in range(Self.C):
                    value += self.get(r, k) * other.get(k, c)
                result.set(r, c, value)
        return result^

    @staticmethod
    def initSlice[o: Origin](source: Span[Scalar[Self.dtype], o]) -> Self:
        debug_assert(len(source) >= Self.R * Self.C, "source length")
        var result = Self()
        for i in range(Self.R * Self.C):
            result.mat[i] = source[i]
        return result^

    @staticmethod
    def initRows(
        rows: Array[Array[Scalar[Self.dtype], Self.C], Self.R]
    ) -> Self:
        var result = Self()
        for r in range(Self.R):
            for c in range(Self.C):
                result.set(r, c, rows[r][c])
        return result^

    def matPrint(self):
        for r in range(Self.R):
            for c in range(Self.C):
                print(self.get(r, c))


comptime Mat22T[dtype: DType] = Mat[2, 2, dtype]
comptime Mat22f = Mat[2, 2, DType.float64]
comptime Mat33T[dtype: DType] = Mat[3, 3, dtype]
comptime Mat33f = Mat[3, 3, DType.float64]
comptime Mat44T[dtype: DType] = Mat[4, 4, dtype]
comptime Mat44f = Mat[4, 4, DType.float64]
