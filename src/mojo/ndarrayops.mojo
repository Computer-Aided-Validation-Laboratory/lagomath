from ndarraytype import NDArray
from matviewtype import MatView


struct NDArrayOps[dtype: DType]:
    @staticmethod
    def add[
        oa: Origin[mut=True], ob: Origin[mut=True], od: Origin[mut=True]
    ](
        a: NDArray[Self.dtype, oa],
        b: NDArray[Self.dtype, ob],
        dst: NDArray[Self.dtype, od],
    ):
        debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
        for i in range(a.length):
            dst.setFlat(i, a.getFlat(i) + b.getFlat(i))

    @staticmethod
    def sub[
        oa: Origin[mut=True], ob: Origin[mut=True], od: Origin[mut=True]
    ](
        a: NDArray[Self.dtype, oa],
        b: NDArray[Self.dtype, ob],
        dst: NDArray[Self.dtype, od],
    ):
        debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
        for i in range(a.length):
            dst.setFlat(i, a.getFlat(i) - b.getFlat(i))

    @staticmethod
    def mulElemWise[
        oa: Origin[mut=True], ob: Origin[mut=True], od: Origin[mut=True]
    ](
        a: NDArray[Self.dtype, oa],
        b: NDArray[Self.dtype, ob],
        dst: NDArray[Self.dtype, od],
    ):
        debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
        for i in range(a.length):
            dst.setFlat(i, a.getFlat(i) * b.getFlat(i))

    @staticmethod
    def divElemWise[
        oa: Origin[mut=True], ob: Origin[mut=True], od: Origin[mut=True]
    ](
        a: NDArray[Self.dtype, oa],
        b: NDArray[Self.dtype, ob],
        dst: NDArray[Self.dtype, od],
    ):
        debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
        for i in range(a.length):
            dst.setFlat(i, a.getFlat(i) / b.getFlat(i))

    @staticmethod
    def extractMat[
        oa: Origin[mut=True], od: Origin[mut=True]
    ](
        a: NDArray[Self.dtype, oa],
        fixed: List[Int],
        row_ext: Int,
        col_ext: Int,
        dst: MatView[Self.dtype, od],
    ):
        var index = fixed.copy()
        for r in range(a.dims[row_ext]):
            for c in range(a.dims[col_ext]):
                index[row_ext] = r
                index[col_ext] = c
                dst.set(r, c, a.get(index))
