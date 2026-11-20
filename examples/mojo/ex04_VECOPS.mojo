import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var v = lm.Vec[3, DType.float64]()
    v.set(0, -0.75)
    v.set(1, -0.625)
    v.set(2, -0.5)
    print(v.normL1(), v.normL2(), v.normL2Square())
    var b = lm.Vec[3, DType.float64]()
    b.set(0, 0.25)
    b.set(1, 0.5)
    b.set(2, 0.75)
    var result = lm.vecops.axpyVec(Float64(-0.5), v, b)
    print(
        lm.vecops.dotVec(v, b),
        v.sum(),
        result.get(0),
        result.get(1),
        result.get(2),
    )
