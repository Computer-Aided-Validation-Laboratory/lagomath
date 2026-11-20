from std.collections import Span
import lagomath as lm

# Inputs sourced from shared numerical fixtures.
from std.collections import Array


def main() raises:
    var data = Array[Float64, 24](fill=0)
    var dims = List[Int]()
    dims.append(2)
    dims.append(3)
    dims.append(4)
    var array = lm.NDArray(Span(data), dims)
    array.setFlat(23, 7)
    print(array.offset3(1, 2, 3), array.getFlat(23))
