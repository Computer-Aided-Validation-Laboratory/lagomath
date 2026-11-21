from std.collections import Span
import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var data = Array[Float64, 4](fill=0)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    var matrix = lm.MatView(Span(data), 2, 2)
    lm.rowops.swapRowsMatView(0, 1, matrix)
    print(data[0], data[1], data[2], data[3])
