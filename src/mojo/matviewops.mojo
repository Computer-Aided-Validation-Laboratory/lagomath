import matops
from vecviewtype import VecView
from matviewtype import MatView


struct MatViewOps[dtype: DType]:
    @staticmethod
    def add[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: MatView[Self.dtype, ob],
        dst: MatView[Self.dtype, od],
    ):
        debug_assert(
            a.rows_num == b.rows_num
            and a.cols_num == b.cols_num
            and dst.rows_num == a.rows_num
            and dst.cols_num == a.cols_num,
            "matrix shape",
        )
        for r in range(a.rows_num):
            for c in range(a.cols_num):
                dst.set(r, c, a.get(r, c) + b.get(r, c))

    @staticmethod
    def sub[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: MatView[Self.dtype, ob],
        dst: MatView[Self.dtype, od],
    ):
        debug_assert(
            a.rows_num == b.rows_num
            and a.cols_num == b.cols_num
            and dst.rows_num == a.rows_num
            and dst.cols_num == a.cols_num,
            "matrix shape",
        )
        for r in range(a.rows_num):
            for c in range(a.cols_num):
                dst.set(r, c, a.get(r, c) - b.get(r, c))

    @staticmethod
    def mulElemWise[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: MatView[Self.dtype, ob],
        dst: MatView[Self.dtype, od],
    ):
        debug_assert(
            a.rows_num == b.rows_num
            and a.cols_num == b.cols_num
            and dst.rows_num == a.rows_num
            and dst.cols_num == a.cols_num,
            "matrix shape",
        )
        for r in range(a.rows_num):
            for c in range(a.cols_num):
                dst.set(r, c, a.get(r, c) * b.get(r, c))

    @staticmethod
    def divElemWise[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: MatView[Self.dtype, ob],
        dst: MatView[Self.dtype, od],
    ):
        debug_assert(
            a.rows_num == b.rows_num
            and a.cols_num == b.cols_num
            and dst.rows_num == a.rows_num
            and dst.cols_num == a.cols_num,
            "matrix shape",
        )
        for r in range(a.rows_num):
            for c in range(a.cols_num):
                dst.set(r, c, a.get(r, c) / b.get(r, c))

    @staticmethod
    def mulScal[
        oa: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        alpha: Scalar[Self.dtype],
        dst: MatView[Self.dtype, od],
    ):
        debug_assert(
            a.rows_num == dst.rows_num and a.cols_num == dst.cols_num,
            "matrix shape",
        )
        for r in range(a.rows_num):
            for c in range(a.cols_num):
                dst.set(r, c, a.get(r, c) * alpha)

    @staticmethod
    def mul[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: MatView[Self.dtype, ob],
        dst: MatView[Self.dtype, od],
    ):
        matops.mulMatMatView(a, b, dst)

    @staticmethod
    def mulSquare[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: MatView[Self.dtype, ob],
        dst: MatView[Self.dtype, od],
    ):
        debug_assert(
            a.rows_num == a.cols_num and b.rows_num == b.cols_num,
            "square matrices",
        )
        matops.mulMatMatView(a, b, dst)

    @staticmethod
    def mulVec[
        oa: Origin, ob: Origin, od: Origin[mut=True]
    ](
        a: MatView[Self.dtype, oa],
        b: VecView[Self.dtype, ob],
        dst: VecView[Self.dtype, od],
    ):
        matops.mulVecMatView(a, b, dst)
