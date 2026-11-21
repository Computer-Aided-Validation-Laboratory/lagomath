import lagomath as lm


def main() raises:
    var a = lm.Mat[3, 3, DType.float64]()
    a.mat[0] = 0.0
    a.mat[1] = 2.0
    a.mat[2] = 1.0
    a.mat[3] = 3.0
    a.mat[4] = -1.0
    a.mat[5] = 2.0
    a.mat[6] = 4.0
    a.mat[7] = -1.0
    a.mat[8] = 5.0
    var b = lm.Vec[3, DType.float64]()
    b.set(0, 5.0)
    b.set(1, 7.0)
    b.set(2, 15.0)
    var result = lm.linsolve.solveGaussMat(a, b)
    print(result.get(0), result.get(1), result.get(2))
