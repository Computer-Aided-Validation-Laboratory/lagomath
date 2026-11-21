"""Opt-in SIMD arithmetic over contiguous NDArray backing storage."""
from ndarraytype import NDArray
import sliceops_simd as core


def addNDArraySIMD[
    dtype: DType,
    oa: Origin[mut=True],
    ob: Origin[mut=True],
    od: Origin[mut=True],
](a: NDArray[dtype, oa], b: NDArray[dtype, ob], dst: NDArray[dtype, od]):
    debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
    core.addSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def subNDArraySIMD[
    dtype: DType,
    oa: Origin[mut=True],
    ob: Origin[mut=True],
    od: Origin[mut=True],
](a: NDArray[dtype, oa], b: NDArray[dtype, ob], dst: NDArray[dtype, od]):
    debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
    core.subSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def mulElemWiseNDArraySIMD[
    dtype: DType,
    oa: Origin[mut=True],
    ob: Origin[mut=True],
    od: Origin[mut=True],
](a: NDArray[dtype, oa], b: NDArray[dtype, ob], dst: NDArray[dtype, od]):
    debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
    core.mulSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def divElemWiseNDArraySIMD[
    dtype: DType,
    oa: Origin[mut=True],
    ob: Origin[mut=True],
    od: Origin[mut=True],
](a: NDArray[dtype, oa], b: NDArray[dtype, ob], dst: NDArray[dtype, od]):
    debug_assert(a.dims == b.dims and dst.dims == a.dims, "array shape")
    core.divSIMD(a.data.as_imm(), b.data.as_imm(), dst.data)


def fillNDArraySIMD[
    dtype: DType, oa: Origin[mut=True]
](a: NDArray[dtype, oa], value: Scalar[dtype]):
    core.fillSIMD(a.data, value)


def mulScalNDArraySIMD[
    dtype: DType, oa: Origin[mut=True]
](a: NDArray[dtype, oa], alpha: Scalar[dtype]):
    core.mulScalInPlaceSIMD(alpha, a.data)
