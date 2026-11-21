import lagomath as lm


def main() raises:
    var a = lm.Mat[2, 2, DType.float64].initIdentity()
    var b = lm.Vec[2, DType.float64](1)
    var result = lm.linsolve.solveGaussMatChecked(a, b)
    print(result.get(0), result.get(1))
