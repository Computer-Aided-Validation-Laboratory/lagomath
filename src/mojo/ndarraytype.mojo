from std.collections import Span
from vecviewtype import VecView


struct NDArray[dtype: DType, origin: Origin[mut=True]](Movable):
    var data: Span[Scalar[Self.dtype], Self.origin]
    var length: Int
    var dims: List[Int]
    var strides: List[Int]

    def __init__(
        out self,
        data: Span[Scalar[Self.dtype], Self.origin],
        dims: List[Int],
    ):
        debug_assert(len(dims) > 0, "positive rank")
        self.data = data
        self.length = len(data)
        var length = self.length
        self.dims = dims.copy()
        self.strides = List[Int]()
        var product = 1
        for i in range(len(dims)):
            debug_assert(dims[i] > 0, "positive dimensions")
            product *= dims[i]
        debug_assert(length >= product, "backing length")
        for i in range(len(dims)):
            var stride = 1
            for j in range(i + 1, len(dims)):
                stride *= dims[j]
            self.strides.append(stride)

    @staticmethod
    def init(
        data: Span[Scalar[Self.dtype], Self.origin], dims: List[Int]
    ) -> Self:
        return Self(data, dims)

    def getFlatIdx(self, indices: List[Int]) -> Int:
        debug_assert(len(indices) == len(self.dims), "index rank")
        var result = 0
        for i in range(len(indices)):
            debug_assert(
                indices[i] >= 0 and indices[i] < self.dims[i], "index bounds"
            )
            result += indices[i] * self.strides[i]
        return result

    def get(self, indices: List[Int]) -> Scalar[Self.dtype]:
        return self.data[self.getFlatIdx(indices)]

    def set(self, indices: List[Int], value: Scalar[Self.dtype]):
        self.data[self.getFlatIdx(indices)] = value

    def getFlat(self, index: Int) -> Scalar[Self.dtype]:
        debug_assert(index >= 0 and index < self.length, "flat index")
        return self.data[index]

    def setFlat(self, index: Int, value: Scalar[Self.dtype]):
        debug_assert(index >= 0 and index < self.length, "flat index")
        self.data[index] = value

    def calcFlatStride(self, dim: Int) -> Int:
        return self.strides[dim]

    def planeBase(self, index: Int) -> Int:
        debug_assert(
            len(self.dims) >= 1 and index >= 0 and index < self.dims[0],
            "plane index",
        )
        return index * self.strides[0]

    def subBase2(self, i: Int, j: Int) -> Int:
        debug_assert(
            len(self.dims) >= 2
            and i >= 0
            and i < self.dims[0]
            and j >= 0
            and j < self.dims[1],
            "prefix index",
        )
        return i * self.strides[0] + j * self.strides[1]

    def offset2(self, i: Int, j: Int) -> Int:
        debug_assert(len(self.dims) == 2, "rank two")
        return self.subBase2(i, j)

    def offset3(self, i: Int, j: Int, k: Int) -> Int:
        debug_assert(len(self.dims) == 3, "rank three")
        debug_assert(k >= 0 and k < self.dims[2], "third index")
        return self.subBase2(i, j) + k

    def offset4(self, i: Int, j: Int, k: Int, l: Int) -> Int:
        debug_assert(len(self.dims) == 4, "rank four")
        debug_assert(
            k >= 0 and k < self.dims[2] and l >= 0 and l < self.dims[3],
            "trailing indices",
        )
        return self.subBase2(i, j) + k * self.strides[2] + l

    def getPlaneSlice(self, channel: Int) -> VecView[Self.dtype, Self.origin]:
        var start = channel * self.strides[0]
        debug_assert(channel >= 0 and start <= self.length, "plane slice")
        return VecView(self.data[start:])

    def getSlice(
        self, indices: List[Int], fixed_dim: Int
    ) -> VecView[Self.dtype, Self.origin]:
        debug_assert(
            fixed_dim >= 0 and fixed_dim + 1 < len(self.dims), "fixed dimension"
        )
        return VecView(
            self.data[
                self.getFlatIdx(indices) : self.getFlatIdx(indices)
                + self.strides[fixed_dim]
            ]
        )

    def fixedPrefixView(self, prefix: List[Int]) -> Self:
        debug_assert(len(prefix) < len(self.dims), "prefix rank")
        var start = 0
        var dims = List[Int]()
        for i in range(len(prefix)):
            debug_assert(
                prefix[i] >= 0 and prefix[i] < self.dims[i], "prefix index"
            )
            start += prefix[i] * self.strides[i]
        for i in range(len(prefix), len(self.dims)):
            dims.append(self.dims[i])
        var length = self.length
        if len(prefix) > 0:
            length = self.strides[len(prefix) - 1]
        return Self(self.data[start : start + length], dims)

    @staticmethod
    def initFlat(
        mut storage: List[Scalar[Self.dtype]], dims: List[Int]
    ) -> NDArray[Self.dtype, origin_of(storage)]:
        debug_assert(len(dims) > 0, "positive rank")
        var count = 1
        for i in range(len(dims)):
            debug_assert(dims[i] > 0, "positive dimensions")
            count *= dims[i]
        storage = List[Scalar[Self.dtype]](capacity=count)
        for i in range(count):
            storage.append(0)
        return NDArray(Span(storage), dims)

    def deinit(mut self):
        # Data remains caller-owned, exactly as for Zig's split ownership contract.
        self.dims = List[Int]()
        self.strides = List[Int]()

    def dupe(
        self, mut storage: List[Scalar[Self.dtype]]
    ) -> NDArray[Self.dtype, origin_of(storage)]:
        storage = List[Scalar[Self.dtype]](capacity=self.length)
        for i in range(self.length):
            storage.append(self.getFlat(i))
        return NDArray(Span(storage), self.dims)

    def fill(self, value: Scalar[Self.dtype]):
        for i in range(self.length):
            self.data[i] = value

    def mulScalInPlace(self, alpha: Scalar[Self.dtype]):
        for i in range(self.length):
            self.data[i] *= alpha

    def addInPlace[
        o_other: Origin[mut=True]
    ](self, other: NDArray[Self.dtype, o_other]):
        debug_assert(self.dims == other.dims, "array shape")
        for i in range(self.length):
            self.setFlat(i, self.getFlat(i) + other.getFlat(i))

    def subInPlace[
        o_other: Origin[mut=True]
    ](self, other: NDArray[Self.dtype, o_other]):
        debug_assert(self.dims == other.dims, "array shape")
        for i in range(self.length):
            self.setFlat(i, self.getFlat(i) - other.getFlat(i))

    def mulInPlace[
        o_other: Origin[mut=True]
    ](self, other: NDArray[Self.dtype, o_other]):
        debug_assert(self.dims == other.dims, "array shape")
        for i in range(self.length):
            self.setFlat(i, self.getFlat(i) * other.getFlat(i))

    def divInPlace[
        o_other: Origin[mut=True]
    ](self, other: NDArray[Self.dtype, o_other]):
        debug_assert(self.dims == other.dims, "array shape")
        for i in range(self.length):
            self.setFlat(i, self.getFlat(i) / other.getFlat(i))


def matchArrayDims[
    dtype: DType, oa: Origin[mut=True], ob: Origin[mut=True]
](a: NDArray[dtype, oa], b: NDArray[dtype, ob]) -> Bool:
    return a.dims == b.dims
