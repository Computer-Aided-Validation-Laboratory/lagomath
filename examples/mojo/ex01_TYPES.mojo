import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var v = lm.Vec[3, DType.float64]()
    v.set(0, -0.75)
    v.set(1, -0.625)
    v.set(2, -0.5)
    print(v.normL1(), v.normL2(), v.normL2Square())
