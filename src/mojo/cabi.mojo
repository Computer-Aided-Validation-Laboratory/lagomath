from std.collections import Span
import lagomath as lm
from std.math import isfinite


@export
def lm_abi_version() abi("C") -> UInt32:
    return 1


def overlap[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutUntrackedOrigin],
    an: Int,
    b: Pointer[Scalar[dtype], MutUntrackedOrigin],
    bn: Int,
) -> Bool:
    var width = 8
    comptime if dtype == DType.float32:
        width = 4
    return Int(a) < Int(b) + bn * width and Int(b) < Int(a) + an * width


def dimensions(n: Int64) -> Bool:
    return n >= 0 and n <= 1000000000


def dot[
    dtype: DType
](
    ap: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    bp: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    n: Int64,
    rp: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
) -> Int32:
    if not dimensions(n) or not rp:
        return 1
    var r = rp.value()
    if n == 0:
        r[unsafe_offset=0] = 0
        return 0
    if not ap or not bp:
        return 1
    var a = lm.VecView(Span(unsafe_ptr=ap.value(), length=Int(n)))
    var b = lm.VecView(Span(unsafe_ptr=bp.value(), length=Int(n)))
    if not lm.vecops.allFiniteVecView(a) or not lm.vecops.allFiniteVecView(b):
        return 5
    r[unsafe_offset=0] = lm.vecops.dotVecView(a, b)
    return 0


def matmul[
    dtype: DType
](
    ap: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    bp: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    op: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    rows: Int64,
    inner: Int64,
    cols: Int64,
) -> Int32:
    if (
        not dimensions(rows)
        or not dimensions(inner)
        or not dimensions(cols)
        or rows == 0
        or inner == 0
        or cols == 0
        or not ap
        or not bp
        or not op
    ):
        return 1
    var a = ap.value()
    var b = bp.value()
    var dst = op.value()
    var m = Int(rows)
    var n = Int(inner)
    var p = Int(cols)
    if overlap(a, m * n, dst, m * p) or overlap(b, n * p, dst, m * p):
        return 6
    var av = lm.MatView(Span(unsafe_ptr=a, length=m * n), m, n)
    var bv = lm.MatView(Span(unsafe_ptr=b, length=n * p), n, p)
    if not lm.matops.allFiniteMatView(av) or not lm.matops.allFiniteMatView(bv):
        return 5
    lm.matops.mulMatMatView(
        av, bv, lm.MatView(Span(unsafe_ptr=dst, length=m * p), m, p)
    )
    return 0


def axpy[
    dtype: DType
](
    alpha: Scalar[dtype],
    ap: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    op: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    n: Int64,
) -> Int32:
    if not dimensions(n):
        return 1
    if n == 0:
        return 0
    if not ap or not op:
        return 1
    var a = ap.value()
    var dst = op.value()
    if overlap(a, Int(n), dst, Int(n)) and Int(a) != Int(dst):
        return 6
    var av = lm.VecView(Span(unsafe_ptr=a, length=Int(n)))
    var dv = lm.VecView(Span(unsafe_ptr=dst, length=Int(n)))
    if (
        not isfinite(alpha)
        or not lm.vecops.allFiniteVecView(av)
        or not lm.vecops.allFiniteVecView(dv)
    ):
        return 5
    lm.vecops.axpyVecView(alpha, av, dv)
    return 0


def solve[
    dtype: DType, chol: Bool
](
    ap: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    bp: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    op: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    n: Int64,
    sp: Optional[Pointer[Scalar[dtype], MutUntrackedOrigin]],
    count: Int64,
) -> Int32:
    if not dimensions(n) or not dimensions(count) or n == 0:
        return 1
    var length = Int(n)
    if Int(count) < length * length + length:
        return 2
    if not ap or not bp or not op or not sp:
        return 1
    var a = ap.value()
    var b = bp.value()
    var dst = op.value()
    var scratch = sp.value()
    var total = length * length + length
    if (
        overlap(a, length * length, dst, length)
        or overlap(b, length, dst, length)
        or overlap(a, length * length, scratch, total)
        or overlap(b, length, scratch, total)
        or overlap(dst, length, scratch, total)
    ):
        return 6
    var av = lm.MatView(
        Span(unsafe_ptr=a, length=length * length), length, length
    )
    var bv = lm.VecView(Span(unsafe_ptr=b, length=length))
    var dv = lm.VecView(Span(unsafe_ptr=dst, length=length))
    var wm = lm.MatView(
        Span(unsafe_ptr=scratch, length=length * length), length, length
    )
    var wv = lm.VecView(
        Span(unsafe_ptr=scratch, length=total)[length * length :]
    )
    try:
        comptime if chol:
            lm.linsolve.solveCholeskyMatViewChecked(av, bv, wm, wv, dv)
        else:
            lm.linsolve.solveGaussMatViewChecked(av, bv, wm, wv, dv)
    except e:
        var message = String(e)
        if message == "SingularMatrix":
            return 3
        if message == "NotPositiveDefinite":
            return 4
        if message == "NonFiniteInput":
            return 5
        return 2
    return 0


@export
def lm_dot_f32(
    ap: Optional[Pointer[Float32, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    n: Int64,
    rp: Optional[Pointer[Float32, MutUntrackedOrigin]],
) abi("C") -> Int32:
    return dot(ap, bp, n, rp)


@export
def lm_matmul_f32(
    ap: Optional[Pointer[Float32, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    op: Optional[Pointer[Float32, MutUntrackedOrigin]],
    m: Int64,
    n: Int64,
    p: Int64,
) abi("C") -> Int32:
    return matmul(ap, bp, op, m, n, p)


@export
def lm_matvec_f32(
    ap: Optional[Pointer[Float32, MutUntrackedOrigin]],
    m: Int64,
    n: Int64,
    bp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    op: Optional[Pointer[Float32, MutUntrackedOrigin]],
) abi("C") -> Int32:
    return matmul(ap, bp, op, m, n, 1)


@export
def lm_axpy_f32(
    alpha: Float32,
    ap: Optional[Pointer[Float32, MutUntrackedOrigin]],
    op: Optional[Pointer[Float32, MutUntrackedOrigin]],
    n: Int64,
) abi("C") -> Int32:
    return axpy(alpha, ap, op, n)


@export
def lm_solve_gauss_f32(
    ap: Optional[Pointer[Float32, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    op: Optional[Pointer[Float32, MutUntrackedOrigin]],
    n: Int64,
    sp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    count: Int64,
) abi("C") -> Int32:
    return solve[DType.float32, False](ap, bp, op, n, sp, count)


@export
def lm_solve_cholesky_f32(
    ap: Optional[Pointer[Float32, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    op: Optional[Pointer[Float32, MutUntrackedOrigin]],
    n: Int64,
    sp: Optional[Pointer[Float32, MutUntrackedOrigin]],
    count: Int64,
) abi("C") -> Int32:
    return solve[DType.float32, True](ap, bp, op, n, sp, count)


@export
def lm_dot_f64(
    ap: Optional[Pointer[Float64, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    n: Int64,
    rp: Optional[Pointer[Float64, MutUntrackedOrigin]],
) abi("C") -> Int32:
    return dot(ap, bp, n, rp)


@export
def lm_matmul_f64(
    ap: Optional[Pointer[Float64, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    op: Optional[Pointer[Float64, MutUntrackedOrigin]],
    m: Int64,
    n: Int64,
    p: Int64,
) abi("C") -> Int32:
    return matmul(ap, bp, op, m, n, p)


@export
def lm_matvec_f64(
    ap: Optional[Pointer[Float64, MutUntrackedOrigin]],
    m: Int64,
    n: Int64,
    bp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    op: Optional[Pointer[Float64, MutUntrackedOrigin]],
) abi("C") -> Int32:
    return matmul(ap, bp, op, m, n, 1)


@export
def lm_axpy_f64(
    alpha: Float64,
    ap: Optional[Pointer[Float64, MutUntrackedOrigin]],
    op: Optional[Pointer[Float64, MutUntrackedOrigin]],
    n: Int64,
) abi("C") -> Int32:
    return axpy(alpha, ap, op, n)


@export
def lm_solve_gauss_f64(
    ap: Optional[Pointer[Float64, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    op: Optional[Pointer[Float64, MutUntrackedOrigin]],
    n: Int64,
    sp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    count: Int64,
) abi("C") -> Int32:
    return solve[DType.float64, False](ap, bp, op, n, sp, count)


@export
def lm_solve_cholesky_f64(
    ap: Optional[Pointer[Float64, MutUntrackedOrigin]],
    bp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    op: Optional[Pointer[Float64, MutUntrackedOrigin]],
    n: Int64,
    sp: Optional[Pointer[Float64, MutUntrackedOrigin]],
    count: Int64,
) abi("C") -> Int32:
    return solve[DType.float64, True](ap, bp, op, n, sp, count)


from std.collections import Array
from std.sys import size_of, align_of


@fieldwise_init
struct Mat22ABI(Copyable, Movable):
    var data: Array[Float64, 4]


@export
def lm_mat22_transpose_f64(
    ap: Optional[Pointer[Mat22ABI, MutUntrackedOrigin]],
    op: Optional[Pointer[Mat22ABI, MutUntrackedOrigin]],
) abi("C") -> Int32:
    if not ap or not op:
        return 1
    var a = ap.value()
    var dst = op.value()
    for i in range(4):
        if not isfinite(a[unsafe_offset=0].data[i]):
            return 5
    var copy = a[unsafe_offset=0].copy()
    dst[unsafe_offset=0].data[0] = copy.data[0]
    dst[unsafe_offset=0].data[1] = copy.data[2]
    dst[unsafe_offset=0].data[2] = copy.data[1]
    dst[unsafe_offset=0].data[3] = copy.data[3]
    return 0


@export
def lm_mat22_size_f64() abi("C") -> UInt32:
    return UInt32(size_of[Mat22ABI]())


@export
def lm_mat22_align_f64() abi("C") -> UInt32:
    return UInt32(align_of[Mat22ABI]())
