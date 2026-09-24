// Lagomath: Lightweight Matrix and Vector Mathematics for Zig
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)

const std = @import("std");
const assert = std.debug.assert;
const MatStack = @import("matstack.zig").MatStack;
const VecStack = @import("vecstack.zig").VecStack;
const MatSlice = @import("matslice.zig").MatSlice;
const VecSlice = @import("vecslice.zig").VecSlice;

/// Multiplies an M x N fixed-size matrix by an N-vector, returning an M-vector.
///
/// Dimensions M and N are known at compile time.
/// Arguments are passed by value and are not mutated.
/// Performs no heap allocation.
pub fn mulVecStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: MatStack(M, N, T),
    vec: VecStack(N, T),
) VecStack(M, T) {
    var result: VecStack(M, T) = undefined;
    inline for (0..M) |rr| {
        var sum: T = 0;
        inline for (0..N) |cc| {
            sum += mat.get(rr, cc) * vec.get(cc);
        }
        result.set(rr, sum);
    }
    return result;
}

/// Multiplies a borrowed slice matrix `mat` by `vec`, writing into `out`.
///
/// Asserts that `mat.cols_num == vec.slice.len` and `mat.rows_num == out.slice.len`.
/// Storage is borrowed; no memory is allocated.
pub fn mulVecSlice(
    comptime T: type,
    mat: MatSlice(T),
    vec: VecSlice(T),
    out: VecSlice(T),
) void {
    assert(mat.cols_num == vec.slice.len);
    assert(mat.rows_num == out.slice.len);
    for (0..mat.rows_num) |rr| {
        var sum: T = 0;
        for (0..mat.cols_num) |cc| {
            sum += mat.get(rr, cc) * vec.get(cc);
        }
        out.set(rr, sum);
    }
}

/// Multiplies an M x N matrix by an N x P matrix, returning an M x P matrix.
///
/// General rectangular matrix multiplication with dimensions known at compile time.
/// Passed by value; performs no heap allocation.
pub fn mulMatStack(
    comptime M: usize,
    comptime N: usize,
    comptime P: usize,
    comptime T: type,
    lhs: MatStack(M, N, T),
    rhs: MatStack(N, P, T),
) MatStack(M, P, T) {
    var result = MatStack(M, P, T).initZeros();
    inline for (0..M) |rr| {
        inline for (0..P) |cc| {
            var sum: T = 0;
            inline for (0..N) |kk| {
                sum += lhs.get(rr, kk) * rhs.get(kk, cc);
            }
            result.set(rr, cc, sum);
        }
    }
    return result;
}

/// Multiplies borrowed slice matrix `lhs` by `rhs`, writing to `out`.
///
/// Asserts matching inner dimensions and output shape.
/// Storage is borrowed; no heap allocation.
pub fn mulMatSlice(
    comptime T: type,
    lhs: MatSlice(T),
    rhs: MatSlice(T),
    out: MatSlice(T),
) void {
    assert(lhs.cols_num == rhs.rows_num);
    assert(lhs.rows_num == out.rows_num);
    assert(rhs.cols_num == out.cols_num);

    for (0..lhs.rows_num) |rr| {
        for (0..rhs.cols_num) |cc| {
            var sum: T = 0;
            for (0..lhs.cols_num) |kk| {
                sum += lhs.get(rr, kk) * rhs.get(kk, cc);
            }
            out.set(rr, cc, sum);
        }
    }
}

/// Computes the outer product `A = lhs * rhs^T` of two fixed-size vectors.
///
/// Dimensions M and N are known at compile time.
/// Result is returned by value as an M x N matrix.
/// Performs no heap allocation.
pub fn outerStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    lhs: VecStack(M, T),
    rhs: VecStack(N, T),
) MatStack(M, N, T) {
    var result: MatStack(M, N, T) = undefined;
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            result.set(rr, cc, lhs.get(rr) * rhs.get(cc));
        }
    }
    return result;
}

/// Computes the outer product `out = lhs * rhs^T` for borrowed slices.
///
/// Asserts that `out.rows_num == lhs.slice.len` and `out.cols_num == rhs.slice.len`.
/// Writes directly to caller-provided `out`. Performs no heap allocation.
pub fn outerSlice(
    comptime T: type,
    lhs: VecSlice(T),
    rhs: VecSlice(T),
    out: MatSlice(T),
) void {
    assert(out.rows_num == lhs.slice.len);
    assert(out.cols_num == rhs.slice.len);
    for (0..lhs.slice.len) |rr| {
        for (0..rhs.slice.len) |cc| {
            out.set(rr, cc, lhs.get(rr) * rhs.get(cc));
        }
    }
}

/// Accumulates scaled outer product `mat += alpha * lhs * rhs^T` for fixed-size types.
///
/// Dimensions M and N are compile-time constants.
/// Does not construct a temporary matrix; accumulates directly into returned value.
/// Performs no heap allocation.
pub fn outerAddStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    alpha: T,
    lhs: VecStack(M, T),
    rhs: VecStack(N, T),
    mat: MatStack(M, N, T),
) MatStack(M, N, T) {
    var result = mat;
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            const current = result.get(rr, cc);
            result.set(rr, cc, current + alpha * lhs.get(rr) * rhs.get(cc));
        }
    }
    return result;
}

/// Accumulates scaled outer product `mat += alpha * lhs * rhs^T` in place for slices.
///
/// Asserts matching dimensions between vectors and target matrix.
/// Directly mutates `mat` without temporary storage. Performs no heap allocation.
pub fn outerAddSlice(
    comptime T: type,
    alpha: T,
    lhs: VecSlice(T),
    rhs: VecSlice(T),
    mat: MatSlice(T),
) void {
    assert(mat.rows_num == lhs.slice.len);
    assert(mat.cols_num == rhs.slice.len);
    for (0..lhs.slice.len) |rr| {
        for (0..rhs.slice.len) |cc| {
            const current = mat.get(rr, cc);
            mat.set(rr, cc, current + alpha * lhs.get(rr) * rhs.get(cc));
        }
    }
}

/// Computes the Frobenius norm of fixed-size matrix `mat`.
///
/// Reuses flat contiguous array elements directly.
/// Value semantics; performs no heap allocation.
pub fn frobeniusNormStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: MatStack(M, N, T),
) T {
    var sum: T = 0;
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            sum += mat.mat[rr][cc] * mat.mat[rr][cc];
        }
    }
    return @sqrt(sum);
}

/// Computes the Frobenius norm of borrowed slice matrix `mat`.
///
/// Reuses flat slice storage directly. Performs no heap allocation.
pub fn frobeniusNormSlice(
    comptime T: type,
    mat: MatSlice(T),
) T {
    var sum: T = 0;
    for (0..mat.slice.len) |ii| {
        sum += mat.slice[ii] * mat.slice[ii];
    }
    return @sqrt(sum);
}

/// Computes the maximum absolute value across all elements of fixed-size matrix `mat`.
///
/// Dimensions are known at compile time. Requires M > 0 and N > 0.
/// Value semantics; performs no heap allocation.
pub fn maxAbsStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: MatStack(M, N, T),
) T {
    comptime assert(M > 0 and N > 0);
    var max_val: T = @abs(mat.mat[0][0]);
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            const abs_val = @abs(mat.mat[rr][cc]);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
    }
    return max_val;
}

/// Computes the maximum absolute value across all elements of borrowed slice matrix `mat`.
///
/// Asserts `mat.slice.len > 0`.
/// Performs no heap allocation.
pub fn maxAbsSlice(
    comptime T: type,
    mat: MatSlice(T),
) T {
    assert(mat.slice.len > 0);
    var max_val: T = @abs(mat.slice[0]);
    for (1..mat.slice.len) |ii| {
        const abs_val = @abs(mat.slice[ii]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

/// Checks whether all elements of fixed-size matrix `mat` are finite.
///
/// Value semantics; performs no heap allocation.
pub fn allFiniteStack(
    comptime M: usize,
    comptime N: usize,
    comptime T: type,
    mat: MatStack(M, N, T),
) bool {
    inline for (0..M) |rr| {
        inline for (0..N) |cc| {
            if (!std.math.isFinite(mat.mat[rr][cc])) {
                return false;
            }
        }
    }
    return true;
}

/// Checks whether all elements of borrowed slice matrix `mat` are finite.
///
/// Performs no heap allocation.
pub fn allFiniteSlice(
    comptime T: type,
    mat: MatSlice(T),
) bool {
    for (0..mat.slice.len) |ii| {
        if (!std.math.isFinite(mat.slice[ii])) {
            return false;
        }
    }
    return true;
}

test "matops.mulVecStack and mulVecSlice" {
    const mat = MatStack(2, 3, f64).initRows(.{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    const vec = VecStack(3, f64).initSlice(&.{ 1.0, 1.0, 2.0 });
    const result = mulVecStack(2, 3, f64, mat, vec);

    try std.testing.expectEqual(@as(f64, 9.0), result.get(0));
    try std.testing.expectEqual(@as(f64, 21.0), result.get(1));

    var mat_raw = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var vec_raw = [_]f64{ 1.0, 1.0, 2.0 };
    var out_raw = [_]f64{ 0.0, 0.0 };

    mulVecSlice(
        f64,
        MatSlice(f64).init(&mat_raw, 2, 3),
        VecSlice(f64).init(&vec_raw),
        VecSlice(f64).init(&out_raw),
    );
    try std.testing.expectEqual(@as(f64, 9.0), out_raw[0]);
    try std.testing.expectEqual(@as(f64, 21.0), out_raw[1]);
}

test "matops.mulMatStack rectangular and mulMatSlice" {
    // 2x3 times 3x2 -> 2x2
    const m_a = MatStack(2, 3, f64).initRows(.{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });
    const m_b = MatStack(3, 2, f64).initRows(.{
        .{ 7.0, 8.0 },
        .{ 9.0, 1.0 },
        .{ 2.0, 3.0 },
    });

    const m_c = mulMatStack(2, 3, 2, f64, m_a, m_b);

    // Row 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31; 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // Row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85; 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    try std.testing.expectEqual(@as(f64, 31.0), m_c.get(0, 0));
    try std.testing.expectEqual(@as(f64, 19.0), m_c.get(0, 1));
    try std.testing.expectEqual(@as(f64, 85.0), m_c.get(1, 0));
    try std.testing.expectEqual(@as(f64, 55.0), m_c.get(1, 1));

    var a_raw = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var b_raw = [_]f64{ 7.0, 8.0, 9.0, 1.0, 2.0, 3.0 };
    var out_raw = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    mulMatSlice(
        f64,
        MatSlice(f64).init(&a_raw, 2, 3),
        MatSlice(f64).init(&b_raw, 3, 2),
        MatSlice(f64).init(&out_raw, 2, 2),
    );
    try std.testing.expectEqual(@as(f64, 31.0), out_raw[0]);
    try std.testing.expectEqual(@as(f64, 19.0), out_raw[1]);
    try std.testing.expectEqual(@as(f64, 85.0), out_raw[2]);
    try std.testing.expectEqual(@as(f64, 55.0), out_raw[3]);
}

test "matops.outer and outerAdd" {
    const x = VecStack(2, f64).initSlice(&.{ 2.0, 3.0 });
    const y = VecStack(3, f64).initSlice(&.{ 4.0, 5.0, 6.0 });

    const out_stack = outerStack(2, 3, f64, x, y);
    try std.testing.expectEqual(@as(f64, 8.0), out_stack.get(0, 0));
    try std.testing.expectEqual(@as(f64, 10.0), out_stack.get(0, 1));
    try std.testing.expectEqual(@as(f64, 12.0), out_stack.get(0, 2));
    try std.testing.expectEqual(@as(f64, 12.0), out_stack.get(1, 0));
    try std.testing.expectEqual(@as(f64, 15.0), out_stack.get(1, 1));
    try std.testing.expectEqual(@as(f64, 18.0), out_stack.get(1, 2));

    const init_mat = MatStack(2, 3, f64).initFill(1.0);
    const add_stack = outerAddStack(2, 3, f64, 0.5, x, y, init_mat);
    try std.testing.expectEqual(@as(f64, 5.0), add_stack.get(0, 0));
    try std.testing.expectEqual(@as(f64, 6.0), add_stack.get(0, 1));
    try std.testing.expectEqual(@as(f64, 7.0), add_stack.get(0, 2));
    try std.testing.expectEqual(@as(f64, 7.0), add_stack.get(1, 0));
    try std.testing.expectEqual(@as(f64, 8.5), add_stack.get(1, 1));
    try std.testing.expectEqual(@as(f64, 10.0), add_stack.get(1, 2));

    var raw_mat = [_]f64{1.0} ** 6;
    var raw_x = [_]f64{ 2.0, 3.0 };
    var raw_y = [_]f64{ 4.0, 5.0, 6.0 };
    outerAddSlice(
        f64,
        0.5,
        VecSlice(f64).init(&raw_x),
        VecSlice(f64).init(&raw_y),
        MatSlice(f64).init(&raw_mat, 2, 3),
    );
    try std.testing.expectEqual(@as(f64, 5.0), raw_mat[0]);
    try std.testing.expectEqual(@as(f64, 6.0), raw_mat[1]);
    try std.testing.expectEqual(@as(f64, 7.0), raw_mat[2]);
    try std.testing.expectEqual(@as(f64, 7.0), raw_mat[3]);
    try std.testing.expectEqual(@as(f64, 8.5), raw_mat[4]);
    try std.testing.expectEqual(@as(f64, 10.0), raw_mat[5]);
}

test "matops.frobeniusNorm and maxAbs" {
    const mat = MatStack(2, 2, f64).initRows(.{
        .{ 1.0, -2.0 },
        .{ 3.0, -4.0 },
    });
    // 1 + 4 + 9 + 16 = 30; sqrt(30)
    const expected_fnorm = @sqrt(@as(f64, 30.0));
    try std.testing.expectEqual(expected_fnorm, frobeniusNormStack(2, 2, f64, mat));
    try std.testing.expectEqual(@as(f64, 4.0), maxAbsStack(2, 2, f64, mat));

    var raw = [_]f64{ 1.0, -2.0, 3.0, -4.0 };
    const s_mat = MatSlice(f64).init(&raw, 2, 2);
    try std.testing.expectEqual(expected_fnorm, frobeniusNormSlice(f64, s_mat));
    try std.testing.expectEqual(@as(f64, 4.0), maxAbsSlice(f64, s_mat));
}

test "matops.allFinite" {
    const m_good = MatStack(2, 2, f64).initFill(1.0);
    try std.testing.expect(allFiniteStack(2, 2, f64, m_good));

    var m_nan = m_good;
    m_nan.set(0, 1, std.math.nan(f64));
    try std.testing.expect(!allFiniteStack(2, 2, f64, m_nan));

    var raw_good = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var raw_inf = [_]f64{ 1.0, std.math.inf(f64), 3.0, 4.0 };
    try std.testing.expect(allFiniteSlice(f64, MatSlice(f64).init(&raw_good, 2, 2)));
    try std.testing.expect(!allFiniteSlice(f64, MatSlice(f64).init(&raw_inf, 2, 2)));
}
