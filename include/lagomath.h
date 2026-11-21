#ifndef LAGOMATH_H
#define LAGOMATH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define LM_ABI_VERSION 1
#define LM_OK 0
#define LM_INVALID_ARGUMENT 1
#define LM_SHAPE_MISMATCH 2
#define LM_SINGULAR 3
#define LM_NOT_POSITIVE_DEFINITE 4
#define LM_NONFINITE 5
#define LM_INVALID_ALIASING 6
typedef struct { double data[4]; } lm_mat22_f64;
int32_t lm_mat22_transpose_f64(const lm_mat22_f64 *a, lm_mat22_f64 *out);
uint32_t lm_mat22_size_f64(void);
uint32_t lm_mat22_align_f64(void);
uint32_t lm_abi_version(void);
int32_t lm_dot_f32(const float *a, const float *b, int64_t length, float *result);
int32_t lm_matmul_f32(const float *a, const float *b, float *out, int64_t rows, int64_t inner, int64_t cols);
int32_t lm_matvec_f32(const float *a, int64_t rows, int64_t cols, const float *b, float *out);
int32_t lm_axpy_f32(float alpha, const float *a, float *out, int64_t length);
int32_t lm_solve_gauss_f32(const float *a, const float *b, float *out, int64_t n, float *scratch, int64_t scratch_count);
int32_t lm_solve_cholesky_f32(const float *a, const float *b, float *out, int64_t n, float *scratch, int64_t scratch_count);
int32_t lm_dot_f64(const double *a, const double *b, int64_t length, double *result);
int32_t lm_matmul_f64(const double *a, const double *b, double *out, int64_t rows, int64_t inner, int64_t cols);
int32_t lm_matvec_f64(const double *a, int64_t rows, int64_t cols, const double *b, double *out);
int32_t lm_axpy_f64(double alpha, const double *a, double *out, int64_t length);
int32_t lm_solve_gauss_f64(const double *a, const double *b, double *out, int64_t n, double *scratch, int64_t scratch_count);
int32_t lm_solve_cholesky_f64(const double *a, const double *b, double *out, int64_t n, double *scratch, int64_t scratch_count);
#ifdef __cplusplus
}
#endif
#endif
