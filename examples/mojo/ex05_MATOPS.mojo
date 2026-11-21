import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var a = lm.Mat[2, 2, DType.float64]()
    a.mat[0] = -0.625
    a.mat[1] = -0.5
    a.mat[2] = -0.375
    a.mat[3] = -0.25
    var b = lm.Vec[2, DType.float64]()
    b.set(0, 0.0)
    b.set(1, 0.25)
    var result = lm.matops.mulVecMat(a, b)
    print(result.get(0), result.get(1))
    var rectangular = lm.Mat[2, 3, DType.float64]()
    rectangular.mat[0] = -0.625
    rectangular.mat[1] = -0.5
    rectangular.mat[2] = -0.375
    rectangular.mat[3] = -0.25
    rectangular.mat[4] = -0.125
    rectangular.mat[5] = 0.0
    var rhs = lm.Mat[3, 4, DType.float64]()
    rhs.mat[0] = -0.75
    rhs.mat[1] = -0.5
    rhs.mat[2] = -0.25
    rhs.mat[3] = 0.0
    rhs.mat[4] = 0.25
    rhs.mat[5] = 0.5
    rhs.mat[6] = 0.75
    rhs.mat[7] = -0.75
    rhs.mat[8] = -0.5
    rhs.mat[9] = -0.25
    rhs.mat[10] = 0.0
    rhs.mat[11] = 0.25
    var six = lm.Vec[6, DType.float64]()
    six.set(0, -0.75)
    six.set(1, -0.625)
    six.set(2, -0.5)
    six.set(3, -0.375)
    six.set(4, -0.25)
    six.set(5, -0.125)
    var small = lm.Mat[2, 2, DType.float64]()
    small.mat[0] = 3.0267459962185654
    small.mat[1] = -2.302533430200883
    small.mat[2] = 0.8434609434185725
    small.mat[3] = 1.9640895896025747
    var product_rect = lm.matops.mulMatMat(rectangular, rhs)
    var hessian = lm.matops.outerAddMat(
        Float64(1), six, six, lm.Mat[6, 6, DType.float64]()
    )
    print(product_rect.get(0, 0), hessian.get(0, 0), lm.Mat22Ops.det(small))
