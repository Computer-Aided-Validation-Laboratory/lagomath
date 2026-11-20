import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var a = lm.Mat[2, 2, DType.float64]()
    a.mat[0] = 12.419444258027395
    a.mat[1] = -0.18022820979080997
    a.mat[2] = -0.18022820979080997
    a.mat[3] = 3.769247759293302
    var b = lm.Vec[2, DType.float64]()
    b.set(0, 1.0)
    b.set(1, 2.0)
    var factor = lm.linsolve.factorLUMat(a)
    var result = lm.linsolve.solveLUFactoredMat(factor, b)
    print(result.get(0), result.get(1))
    var second = lm.linsolve.solveLUFactoredMat(factor, b.mulScal(2))
    print(second.get(0), second.get(1))
