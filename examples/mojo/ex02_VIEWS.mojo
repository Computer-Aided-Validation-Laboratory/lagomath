from std.collections import Span
import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var data = Array[Float64, 3](fill=0)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    var view = lm.VecView(Span(data))
    view.set(1, 7)
    print(data[1])
