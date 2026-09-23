"""
Generate reference test cases for linear algebra operations.

Computes reference values for 2x2, 3x3, and 6x6 linear systems,
including LU, Cholesky, and Gaussian elimination reference solutions.
"""

import numpy as np
import scipy.linalg


def print_matrix(name: str, mat: np.ndarray) -> None:
    print(f"// {name}")
    rows, cols = mat.shape
    for rr in range(rows):
        entries = [f"{mat[rr, cc]:.17e}" for cc in range(cols)]
        joined = ", ".join(entries)
        print(f"    .{{{joined}}},")


def print_vector(name: str, vec: np.ndarray) -> None:
    print(f"// {name}")
    entries = [f"{val:.17e}" for val in vec]
    joined = ", ".join(entries)
    print(f"    .{{{joined}}},")


def main() -> None:
    # 2x2 system
    a2 = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    b2 = np.array([5.0, 5.0], dtype=np.float64)
    x2 = np.linalg.solve(a2, b2)
    l2 = np.linalg.cholesky(a2)

    print("// --- 2x2 System ---")
    print_matrix("A2", a2)
    print_vector("b2", b2)
    print_vector("x2", x2)
    print_matrix("L2 (Cholesky)", l2)

    # 3x3 system requiring pivoting
    a3 = np.array(
        [[0.0, 2.0, 1.0], [3.0, -1.0, 2.0], [4.0, -1.0, 5.0]],
        dtype=np.float64,
    )
    b3 = np.array([5.0, 7.0, 15.0], dtype=np.float64)
    x3 = np.linalg.solve(a3, b3)

    print("// --- 3x3 System ---")
    print_matrix("A3", a3)
    print_vector("b3", b3)
    print_vector("x3", x3)

    # 6x6 DIC-like SPD system
    # Hessian: J^T J + lambda I
    np.random.seed(42)
    j = np.random.randn(20, 6)
    a6 = j.T @ j + 0.1 * np.eye(6)
    b6 = np.array([1.2, -0.5, 3.4, -2.1, 0.8, 1.5], dtype=np.float64)
    x6 = np.linalg.solve(a6, b6)
    l6 = np.linalg.cholesky(a6)

    print("// --- 6x6 DIC SPD System ---")
    print_matrix("A6 (SPD)", a6)
    print_vector("b6", b6)
    print_vector("x6", x6)
    print_matrix("L6 (Cholesky)", l6)


if __name__ == "__main__":
    main()
