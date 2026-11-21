from std.collections import Array, Span
from std.math import abs, isfinite, isnan
import lagomath as lm
import sliceops_simd as s
import matops_simd as m
import rowops_simd as rows


def check[dtype: DType]() raises:
    for n in range(64):
        var pa = Array[Scalar[dtype], 66](fill=99)
        var pb = Array[Scalar[dtype], 66](fill=99)
        var po = Array[Scalar[dtype], 66](fill=99)
        for i in range(n):
            pa[i + 1] = Scalar[dtype](i % 9 - 4) / 8
            pb[i + 1] = Scalar[dtype](i % 7 + 1) / 4
        var a = Span(pa)[1 : n + 1]
        var b = Span(pb)[1 : n + 1]
        var dst = Span(po)[1 : n + 1]
        var expected: Scalar[dtype] = 0
        var l1: Scalar[dtype] = 0
        var square: Scalar[dtype] = 0
        for i in range(n):
            expected += a[i] * b[i]
            l1 += abs(a[i])
            square += a[i] * a[i]
        if (
            s.dotSIMD(a.as_imm(), b.as_imm()) != expected
            or s.normL1SIMD(a.as_imm()) != l1
            or s.normL2SquareSIMD(a.as_imm()) != square
        ):
            raise Error("SIMD reduction or tail mismatch")
        s.addSIMD(a.as_imm(), b.as_imm(), dst)
        for i in range(n):
            if dst[i] != a[i] + b[i]:
                raise Error("SIMD add mismatch")
        s.subSIMD(a.as_imm(), b.as_imm(), dst)
        for i in range(n):
            if dst[i] != a[i] - b[i]:
                raise Error("SIMD sub mismatch")
        s.mulSIMD(a.as_imm(), b.as_imm(), dst)
        for i in range(n):
            if dst[i] != a[i] * b[i]:
                raise Error("SIMD multiply mismatch")
        s.divSIMD(a.as_imm(), b.as_imm(), dst)
        for i in range(n):
            if dst[i] != a[i] / b[i]:
                raise Error("SIMD divide mismatch")
        s.axpySIMD(Scalar[dtype](0.5), a.as_imm(), b.as_imm(), dst)
        s.mulScalInPlaceSIMD(Scalar[dtype](2), dst)
        for i in range(n):
            if dst[i] != a[i] + 2 * b[i]:
                raise Error("SIMD in-place mismatch")
        s.fillSIMD(dst, Scalar[dtype](-2))
        for i in range(n):
            if dst[i] != -2:
                raise Error("SIMD fill mismatch")
        if po[0] != 99:
            raise Error("SIMD prefix overwrite")
        for i in range(n + 1, 66):
            if po[i] != 99:
                raise Error("SIMD suffix overwrite")
        if n > 0:
            var av = lm.VecView(a.as_imm())
            var low = s.minSIMD(a.as_imm())
            var high = s.maxSIMD(a.as_imm())
            if (
                low.idx != av.min().idx
                or low.val != av.min().val
                or high.idx != av.max().idx
                or high.val != av.max().val
            ):
                raise Error("SIMD extrema tie mismatch")
    var nonfinite = Array[Scalar[dtype], 17](fill=1)
    for i in range(17):
        nonfinite[i] = Scalar[dtype](0) / Scalar[dtype](0)
        if s.allFiniteSIMD(Span(nonfinite).as_imm()):
            raise Error("SIMD missed NaN")
        var av = lm.VecView(Span(nonfinite).as_imm())
        var actual = s.maxAbsSIMD(Span(nonfinite).as_imm())
        var want = lm.vecops.maxAbsVecView(av)
        if not (isnan(actual) and isnan(want)) and actual != want:
            raise Error("SIMD maxAbs NaN semantics")
        nonfinite[i] = Scalar[dtype](1)
    var a = Array[Scalar[dtype], 5 * 9](fill=0)
    var b = Array[Scalar[dtype], 9 * 11](fill=0)
    var out = Array[Scalar[dtype], 5 * 11 + 2](fill=99)
    for i in range(5 * 9):
        a[i] = Scalar[dtype](i % 11 - 5) / 8
    for i in range(9 * 11):
        b[i] = Scalar[dtype](i % 7 + 1) / 4
    m.mulMatSIMD(5, 9, 11, Span(a).as_imm(), Span(b).as_imm(), Span(out)[1:56])
    for r in range(5):
        for c in range(11):
            var want: Scalar[dtype] = 0
            for k in range(9):
                want += a[r * 9 + k] * b[k * 11 + c]
            if out[1 + r * 11 + c] != want:
                raise Error("SIMD rectangular product mismatch")
    if out[0] != 99 or out[56] != 99:
        raise Error("SIMD product overwrite")
    var source = Array[Scalar[dtype], 9 * 17](fill=0)
    var trans = Array[Scalar[dtype], 9 * 17 + 2](fill=99)
    for i in range(9 * 17):
        source[i] = Scalar[dtype](i)
    m.transposeSIMD(9, 17, Span(source).as_imm(), Span(trans)[1:154])
    for r in range(9):
        for c in range(17):
            if trans[1 + c * 9 + r] != source[r * 17 + c]:
                raise Error("SIMD transpose mismatch")
    if trans[0] != 99 or trans[154] != 99:
        raise Error("SIMD transpose overwrite")
    var matrix = lm.Mat[3, 9, dtype](2)
    rows.addScaledRowMatSIMD(0, 0, Scalar[dtype](-0.5), matrix)
    rows.swapRowsMatSIMD(0, 2, matrix)
    if matrix.get(0, 8) != 2 or matrix.get(2, 8) != 1:
        raise Error("SIMD aliased row update mismatch")

    var aa = Array[Scalar[dtype], 15](fill=2)
    var bb = Array[Scalar[dtype], 15](fill=4)
    var oo = Array[Scalar[dtype], 15](fill=0)
    var shape: List[Int] = [3, 5]
    var na = lm.NDArray(Span(aa), shape)
    var nb = lm.NDArray(Span(bb), shape)
    var nd = lm.NDArray(Span(oo), shape)
    lm.ndarrayops_simd.addNDArraySIMD(na, nb, nd)
    for i in range(15):
        if nd.getFlat(i) != 6:
            raise Error("SIMD NDArray add")
    lm.ndarrayops_simd.subNDArraySIMD(na, nb, nd)
    for i in range(15):
        if nd.getFlat(i) != -2:
            raise Error("SIMD NDArray subtract")
    lm.ndarrayops_simd.mulElemWiseNDArraySIMD(na, nb, nd)
    for i in range(15):
        if nd.getFlat(i) != 8:
            raise Error("SIMD NDArray multiply")
    lm.ndarrayops_simd.divElemWiseNDArraySIMD(na, nb, nd)
    for i in range(15):
        if nd.getFlat(i) != 0.5:
            raise Error("SIMD NDArray divide")
    lm.ndarrayops_simd.fillNDArraySIMD(nd, Scalar[dtype](3))
    lm.ndarrayops_simd.mulScalNDArraySIMD(nd, Scalar[dtype](-2))
    for i in range(15):
        if nd.getFlat(i) != -6:
            raise Error("SIMD NDArray fill and scale")

    var ba = Array[Scalar[dtype], 17 * 3](fill=0)
    var bbv = Array[Scalar[dtype], 17 * 3](fill=0)
    var bo = Array[Scalar[dtype], 17](fill=0)
    for i in range(17 * 3):
        ba[i] = Scalar[dtype](i % 13 - 6) / 8
        bbv[i] = Scalar[dtype](i % 7 + 1) / 4
    lm.vecops_simd.dotBatchSIMD[3](
        Span(ba).as_imm(), Span(bbv).as_imm(), Span(bo)
    )
    for i in range(17):
        var want = (
            ba[i * 3] * bbv[i * 3]
            + ba[i * 3 + 1] * bbv[i * 3 + 1]
            + ba[i * 3 + 2] * bbv[i * 3 + 2]
        )
        if bo[i] != want:
            raise Error("SIMD caller-batch mismatch")
    var perm = Array[Int, 17](fill=0)
    var gathered = Array[Scalar[dtype], 17](fill=0)
    for i in range(17):
        perm[i] = 16 - i
    rows.applyPermutationVecViewSIMD(
        Span(perm), lm.VecView(Span(bo).as_imm()), lm.VecView(Span(gathered))
    )
    for i in range(17):
        if gathered[i] != bo[16 - i]:
            raise Error("SIMD gather mismatch")
    var ea = Array[Scalar[dtype], 5](fill=99)
    var eb = Array[Scalar[dtype], 5](fill=88)
    var eo = Array[Scalar[dtype], 5](fill=77)
    for i in range(4):
        ea[i] = Scalar[dtype](i + 1)
        eb[i] = Scalar[dtype](1 if i == 0 or i == 3 else 0)
    var ma = lm.MatView(Span(ea)[0:4], 2, 2)
    var mb = lm.MatView(Span(eb)[0:4], 2, 2)
    var mo = lm.MatView(Span(eo)[0:4], 2, 2)
    m.mulMatMatViewSIMD(ma, mb, mo)
    for i in range(4):
        if eo[i] != ea[i]:
            raise Error("SIMD oversized product mismatch")
    if eo[4] != 77 or m.maxAbsMatViewSIMD(ma) != 4:
        raise Error("SIMD oversized matrix contract")
    m.fillMatViewSIMD(mo, Scalar[dtype](2))
    if eo[4] != 77:
        raise Error("SIMD matrix fill tail overwrite")


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("SIMD block, tail, sentinel and alias checks passed")
