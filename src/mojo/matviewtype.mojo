from std.collections import Span


struct MatView[dtype: DType, origin: Origin](Copyable, Movable):
    var data: Span[Scalar[Self.dtype], Self.origin]
    var rows_num: Int
    var cols_num: Int

    def __init__(
        out self,
        data: Span[Scalar[Self.dtype], Self.origin],
        rows_num: Int,
        cols_num: Int,
    ):
        debug_assert(
            rows_num >= 0
            and cols_num >= 0
            and len(data) == rows_num * cols_num,
            "matrix backing length",
        )
        self.data = data
        self.rows_num = rows_num
        self.cols_num = cols_num

    @staticmethod
    def init(
        data: Span[Scalar[Self.dtype], Self.origin],
        rows_num: Int,
        cols_num: Int,
    ) -> Self:
        return Self(data, rows_num, cols_num)

    @staticmethod
    def initAlloc(
        mut storage: List[Scalar[Self.dtype]], rows_num: Int, cols_num: Int
    ) -> MatView[Self.dtype, origin_of(storage)]:
        debug_assert(rows_num >= 0 and cols_num >= 0, "matrix dimensions")
        storage = List[Scalar[Self.dtype]](length=rows_num * cols_num, fill=0)
        return MatView(Span(storage), rows_num, cols_num)

    def get(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        debug_assert(
            r >= 0 and r < self.rows_num and c >= 0 and c < self.cols_num,
            "matrix index",
        )
        return self.data[r * self.cols_num + c]

    def set(
        self, r: Int, c: Int, value: Scalar[Self.dtype]
    ) where Self.origin.mut:
        debug_assert(
            r >= 0 and r < self.rows_num and c >= 0 and c < self.cols_num,
            "matrix index",
        )
        comptime if Self.origin.mut:
            self.data[r * self.cols_num + c : r * self.cols_num + c + 1].fill(
                value
            )

    def fill(self, value: Scalar[Self.dtype]) where Self.origin.mut:
        for i in range(self.rows_num * self.cols_num):
            comptime if Self.origin.mut:
                self.data[i : i + 1].fill(value)

    def trace(self) -> Scalar[Self.dtype]:
        var value: Scalar[Self.dtype] = 0
        for i in range(min(self.rows_num, self.cols_num)):
            value += self.get(i, i)
        return value

    def transpose[
        o_dst: Origin[mut=True]
    ](self, dst: MatView[Self.dtype, o_dst]):
        debug_assert(
            dst.rows_num == self.cols_num and dst.cols_num == self.rows_num,
            "transpose shape",
        )
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                dst.set(c, r, self.get(r, c))

    def addInPlace[
        o_other: Origin
    ](self, other: MatView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(
            self.rows_num == other.rows_num and self.cols_num == other.cols_num,
            "matrix shape",
        )
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, self.get(r, c) + other.get(r, c))

    def subInPlace[
        o_other: Origin
    ](self, other: MatView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(
            self.rows_num == other.rows_num and self.cols_num == other.cols_num,
            "matrix shape",
        )
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, self.get(r, c) - other.get(r, c))

    def mulInPlace[
        o_other: Origin
    ](self, other: MatView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(
            self.rows_num == other.rows_num and self.cols_num == other.cols_num,
            "matrix shape",
        )
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, self.get(r, c) * other.get(r, c))

    def divInPlace[
        o_other: Origin
    ](self, other: MatView[Self.dtype, o_other]) where Self.origin.mut:
        debug_assert(
            self.rows_num == other.rows_num and self.cols_num == other.cols_num,
            "matrix shape",
        )
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, self.get(r, c) / other.get(r, c))

    def mulScalInPlace(self, alpha: Scalar[Self.dtype]) where Self.origin.mut:
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, self.get(r, c) * alpha)

    def identity(self) where Self.origin.mut:
        self.fillDiag(0, 1)

    def fillDiag(
        self, fill: Scalar[Self.dtype], diagonal: Scalar[Self.dtype]
    ) where Self.origin.mut:
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, diagonal if r == c else fill)

    def flatIndex(self, r: Int, c: Int) -> Int:
        debug_assert(
            r >= 0 and r < self.rows_num and c >= 0 and c < self.cols_num,
            "matrix index",
        )
        return r * self.cols_num + c

    def rowBase(self, r: Int) -> Int:
        debug_assert(r >= 0 and r < self.rows_num, "row index")
        return r * self.cols_num

    def getFlat(self, index: Int) -> Scalar[Self.dtype]:
        return self.data[index]

    def setFlat(
        self, index: Int, value: Scalar[Self.dtype]
    ) where Self.origin.mut:
        self.set(index // self.cols_num, index % self.cols_num, value)

    def minByRow(self, fixed_col: Int) -> Scalar[Self.dtype]:
        var value = self.get(0, fixed_col)
        for r in range(1, self.rows_num):
            value = min(value, self.get(r, fixed_col))
        return value

    def maxByRow(self, fixed_col: Int) -> Scalar[Self.dtype]:
        var value = self.get(0, fixed_col)
        for r in range(1, self.rows_num):
            value = max(value, self.get(r, fixed_col))
        return value

    def matPrint(self):
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                print(self.get(r, c))

    def transposeSquare[
        o_buffer: Origin[mut=True]
    ](self, buffer: MatView[Self.dtype, o_buffer]) where Self.origin.mut:
        debug_assert(self.rows_num == self.cols_num, "square transpose")
        self.transpose(buffer)
        for r in range(self.rows_num):
            for c in range(self.cols_num):
                self.set(r, c, buffer.get(r, c))

    def getSlice(self, row: Int) -> Span[Scalar[Self.dtype], Self.origin]:
        debug_assert(row >= 0 and row < self.rows_num, "row index")
        return self.data[row * self.cols_num : (row + 1) * self.cols_num]
