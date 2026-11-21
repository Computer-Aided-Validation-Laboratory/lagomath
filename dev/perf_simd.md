# Manual SIMD performance summary

Source: fresh `simd-rerun-open` benchmark run on 20 November 2026. These are machine-specific observations, not portable performance guarantees.

## Method and grouping

- AMD Ryzen 7 8845HS; Zig 0.16.0 ReleaseFast and Mojo 1.2.0 `-O3`, native CPU target, floating-point contraction disabled. Automatic compiler vectorisation remained enabled.
- Manual kernels used 256-bit vectors: eight f32 or four f64 lanes, not full-width AVX-512.
- 1,114 matched workloads per language, each with ten samples and identical repetitions for automatic and manual implementations (1–3,361,541 calls per sample). No suspend was detected.
- Timings include harness dispatch/barriers and operation-specific resets/copies; allocation, JSON parsing and setup are outside the timed region.
- **Faster:** manual maximum < automatic minimum. **Slower:** manual minimum > automatic maximum. **Same / overlapping:** all remaining cases, including bounds that touch. “Same” means inconclusive under this overlap rule, not proven equivalent.
- Bounds describe observed batch-average per-call sample timings, not individual-call latency, confidence intervals or a formal significance test. Speedup is automatic median / manual median; values above 1 favour manual SIMD.

## Overall classification

| Language | Slower | Same / overlapping | Faster | Total |
|---|---:|---:|---:|---:|
| Zig | 150 | 530 | 434 | 1114 |
| Mojo | 78 | 290 | 746 | 1114 |

Counts weight each workload equally; they are not an overall application speedup.

## Matrix multiplication interpretation

The main comparison measures the existing implementation against the manual SIMD implementation, which also changes traversal and blocking. Its large matrix gains cannot be attributed solely to SIMD. A separate scalar-source control uses the matching traversal and allows compiler vectorisation:

| f64 256×256 multiplication | Zig median | Mojo median |
|---|---:|---:|
| Existing automatic implementation | 29.495 ms | 25.713 ms |
| Matching-traversal scalar-source control | 1.271 ms | 1.026 ms |
| Manual SIMD | 1.311 ms | 1.107 ms |

The control and manual ranges overlap at this size in both languages. There is no demonstrated manual SIMD advantage over the matching traversal control here. The classifications below compare against the existing automatic implementation.

## Representative timings

Each cell is median [minimum–maximum]. Exact benchmark IDs identify type, shape and input variant.

### Zig

| Case | Classification | Automatic | Manual SIMD | Speedup |
|---|---|---|---|---:|
| `bench.dot.Vec.3.f64` | Same / overlapping | 3.654 ns [3.615 ns–4.248 ns] | 3.870 ns [3.827 ns–4.467 ns] | 0.944× |
| `bench.dot.VecView.257.f64` | Faster | 139.329 ns [136.965 ns–142.052 ns] | 28.064 ns [27.799 ns–32.849 ns] | 4.965× |
| `bench.dot.VecView.1048576.f64` | Faster | 674.422 µs [662.411 µs–744.385 µs] | 228.835 µs [217.136 µs–344.247 µs] | 2.947× |
| `bench.matmul.MatView.256.f64` | Faster | 29.495 ms [29.328 ms–29.880 ms] | 1.311 ms [1.269 ms–1.537 ms] | 22.502× |
| `bench.lu.MatView.256.f64` | Faster | 1.488 ms [1.474 ms–1.602 ms] | 704.260 µs [691.206 µs–816.544 µs] | 2.112× |
| `bench.choleskyReuse.Mat.6.f64` | Slower | 23.289 ns [23.113 ns–29.139 ns] | 70.501 ns [69.828 ns–76.608 ns] | 0.330× |
| `bench.matmul.MatView.512.f32` | Faster | 332.483 ms [321.210 ms–344.156 ms] | 5.245 ms [5.147 ms–6.559 ms] | 63.396× |
| `bench.matvec.Mat.3x3.pivot_required.f32` | Same / overlapping | 7.929 ns [7.744 ns–9.392 ns] | 7.928 ns [7.672 ns–9.181 ns] | 1.000× |

### Mojo

| Case | Classification | Automatic | Manual SIMD | Speedup |
|---|---|---|---|---:|
| `bench.dot.Vec.3.f64` | Faster | 11.076 ns [11.000 ns–11.716 ns] | 9.386 ns [9.295 ns–9.937 ns] | 1.180× |
| `bench.dot.VecView.257.f64` | Faster | 148.733 ns [145.578 ns–151.729 ns] | 29.352 ns [29.322 ns–29.548 ns] | 5.067× |
| `bench.dot.VecView.1048576.f64` | Faster | 679.316 µs [669.475 µs–745.740 µs] | 224.738 µs [214.401 µs–308.597 µs] | 3.023× |
| `bench.matmul.MatView.256.f64` | Faster | 25.713 ms [24.806 ms–26.173 ms] | 1.107 ms [1.017 ms–1.162 ms] | 23.221× |
| `bench.lu.MatView.256.f64` | Faster | 1.549 ms [1.534 ms–1.672 ms] | 671.304 µs [660.117 µs–680.270 µs] | 2.308× |
| `bench.inv.Mat.3x3.f32` | Slower | 36.582 ns [36.062 ns–40.293 ns] | 63.235 ns [62.811 ns–63.515 ns] | 0.579× |
| `bench.matmul.MatView.512.f32` | Faster | 270.778 ms [265.698 ms–273.152 ms] | 4.871 ms [4.776 ms–6.001 ms] | 55.591× |
| `bench.matmulRect.Mat.2x3x4.f64` | Same / overlapping | 26.286 ns [26.198 ns–27.097 ns] | 26.264 ns [25.485 ns–26.726 ns] | 1.001× |

## All cases grouped by observed outcome

Every workload appears once per language below. Operation, container and dtype are grouped to keep the summary compact. Case variants retain the size/shape and any input-pattern suffix from the benchmark ID: `bench.<operation>.<container>.<variant>.<dtype>`. Speedup ranges are the minimum and maximum median speedup across that row’s cases.

### Zig

#### Slower (150 cases)

| Operation | Container | Dtype | Case variants | Median speedup range |
|---|---|---|---|---:|
| add | slice | f32 | `1` | 0.923–0.923× |
| cholesky | Mat | f32 | `3x3.signed_alternating` | 0.887–0.887× |
| cholesky | Mat | f64 | `6`, `8`, `12`, `6x6.signed_alternating`, `12x12.signed_alternating` | 0.600–0.838× |
| cholesky | MatView | f32 | `4` | 0.897–0.897× |
| cholesky | MatView | f64 | `2`, `4`, `12`, `3x3.signed_alternating` | 0.832–0.962× |
| choleskyBatch16 | Mat | f32 | `12x12` | 0.604–0.604× |
| choleskyBatch16 | Mat | f64 | `6x6`, `8x8`, `12x12` | 0.340–0.518× |
| choleskyBatch16 | MatView | f32 | `12x12` | 0.895–0.895× |
| choleskyBatch16 | MatView | f64 | `2x2`, `6x6`, `8x8`, `12x12` | 0.716–0.947× |
| choleskyReuse | Mat | f32 | `12` | 0.606–0.606× |
| choleskyReuse | Mat | f64 | `6`, `8`, `12` | 0.330–0.507× |
| choleskyReuse | MatView | f32 | `12` | 0.906–0.906× |
| choleskyReuse | MatView | f64 | `12`, `16` | 0.815–0.880× |
| det | Mat | f32 | `3x3` | 0.923–0.923× |
| det | Mat | f64 | `2x2` | 0.981–0.981× |
| div | slice | f32 | `3` | 0.910–0.910× |
| dotBatch3 | slice | f32 | `4096`, `65537` | 0.726–0.729× |
| dotBatch3 | slice | f64 | `4096`, `65537` | 0.715–0.782× |
| factorCholesky | Mat | f32 | `4x4` | 0.951–0.951× |
| factorCholesky | Mat | f64 | `3x3`, `8x8` | 0.893–0.906× |
| factorCholesky | MatView | f32 | `4x4` | 0.903–0.903× |
| factorCholesky | MatView | f64 | `2x2`, `3x3` | 0.869–0.963× |
| factorLU | Mat | f64 | `3x3` | 0.914–0.914× |
| fill | slice | f32 | `3` | 0.947–0.947× |
| fillMatrix | MatView | f64 | `128` | 0.979–0.979× |
| gauss | Mat | f32 | `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.dense_random` | 0.805–0.972× |
| gauss | Mat | f64 | `6x6.signed_alternating`, `6x6.dense_random` | 0.919–0.922× |
| gauss | MatView | f32 | `3x3.signed_alternating`, `3x3.dense_random` | 0.976–0.981× |
| gauss | MatView | f64 | `3x3.pivot_required` | 0.918–0.918× |
| inv | Mat | f32 | `3x3` | 0.963–0.963× |
| inv | Mat | f64 | `4x4` | 0.775–0.775× |
| lu | Mat | f64 | `6x6.signed_alternating`, `6x6.dense_random` | 0.946–0.952× |
| lu | MatView | f32 | `3x3.dense_random` | 0.983–0.983× |
| luBatch16 | Mat | f32 | `12x12` | 0.821–0.821× |
| luBatch16 | Mat | f64 | `6x6`, `8x8`, `12x12` | 0.665–0.763× |
| luBatch16 | MatView | f32 | `12x12` | 0.937–0.937× |
| luBatch16 | MatView | f64 | `8x8` | 0.972–0.972× |
| luReuse | Mat | f32 | `12` | 0.825–0.825× |
| luReuse | Mat | f64 | `6`, `8`, `12` | 0.683–0.809× |
| luReuse | MatView | f32 | `12` | 0.938–0.938× |
| luReuse | MatView | f64 | `12` | 0.899–0.899× |
| matmul | Mat | f32 | `6`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 0.352–0.532× |
| matmul | Mat | f64 | `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 0.441–0.800× |
| matmul | MatView | f32 | `3`, `4`, `6`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random` | 0.590–0.748× |
| matmul | MatView | f64 | `2`, `3`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 0.771–0.852× |
| matmulRect | Mat | f64 | `6x12x8` | 0.431–0.431× |
| matmulRect | MatView | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `12x6x3` | 0.545–0.793× |
| matmulRect | MatView | f64 | `3x2x3`, `4x8x3`, `12x6x3` | 0.654–0.905× |
| matrixMaxAbs | MatView | f32 | `3` | 0.912–0.912× |
| matvec | Mat | f32 | `8`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random` | 0.556–0.779× |
| max | slice | f32 | `7` | 0.937–0.937× |
| max | slice | f64 | `1`, `31` | 0.940–0.946× |
| maxAbs | slice | f32 | `3` | 0.975–0.975× |
| mul | slice | f32 | `3`, `7` | 0.853–0.889× |
| normL2 | Vec | f32 | `4` | 0.958–0.958× |
| outerAdd | Mat | f64 | `4` | 0.743–0.743× |
| permute | slice | f32 | `3` | 0.922–0.922× |
| rowops | MatView | f32 | `4` | 0.953–0.953× |
| rowopsRect | MatView | f64 | `2x3x4`, `4x8x3` | 0.796–0.823× |
| scale | slice | f32 | `3` | 0.934–0.934× |
| scaleRow | MatView | f32 | `3` | 0.950–0.950× |
| sub | slice | f32 | `3` | 0.924–0.924× |
| sum | slice | f32 | `7` | 0.919–0.919× |
| transpose | Mat | f32 | `8`, `12` | 0.448–0.566× |
| transpose | Mat | f64 | `4`, `6` | 0.583–0.719× |
| transpose | MatView | f32 | `8`, `12`, `16`, `32`, `64`, `127` | 0.663–0.944× |
| transpose | MatView | f64 | `8` | 0.872–0.872× |
| transposeRect | Mat | f32 | `2x3x4` | 0.977–0.977× |
| transposeRect | Mat | f64 | `4x8x3`, `6x12x8`, `12x6x3` | 0.540–0.929× |
| transposeRect | MatView | f64 | `4x8x3`, `6x12x8`, `12x6x3` | 0.462–0.738× |

#### Same / overlapping (530 cases)

| Operation | Container | Dtype | Case variants | Median speedup range |
|---|---|---|---|---:|
| add | slice | f32 | `3`, `16`, `31` | 0.957–1.047× |
| add | slice | f64 | `1`, `3`, `7`, `16`, `31` | 0.872–0.972× |
| addScaledRow | MatView | f32 | `3`, `33`, `128`, `512` | 0.963–1.124× |
| addScaledRow | MatView | f64 | `3`, `128`, `512` | 0.982–1.015× |
| allFinite | slice | f32 | `3` | 0.992–0.992× |
| allFinite | slice | f64 | `3` | 0.972–0.972× |
| axpy | Vec | f32 | `3`, `4`, `6`, `8`, `12`, `16` | 0.954–1.064× |
| axpy | Vec | f64 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 0.992–1.075× |
| axpy | slice | f32 | `3`, `33` | 0.978–1.038× |
| axpy | slice | f64 | `3`, `33` | 1.001–1.041× |
| cholesky | Mat | f32 | `2`, `3`, `4`, `6`, `8`, `12`, `6x6.signed_alternating`, `12x12.signed_alternating` | 0.880–1.076× |
| cholesky | Mat | f64 | `2`, `3`, `4` | 0.918–1.058× |
| cholesky | MatView | f32 | `2`, `3`, `6`, `8`, `12`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `16`, `64`, `128`, `256`, `512` | 0.881–1.148× |
| cholesky | MatView | f64 | `3`, `6`, `8`, `6x6.signed_alternating`, `12x12.signed_alternating`, `16`, `32`, `64`, `256`, `512` | 0.918–1.108× |
| choleskyBatch16 | Mat | f32 | `2x2`, `3x3`, `4x4`, `6x6`, `8x8` | 0.996–1.002× |
| choleskyBatch16 | Mat | f64 | `2x2`, `3x3`, `4x4` | 1.003–1.007× |
| choleskyBatch16 | MatView | f32 | `2x2`, `3x3`, `4x4`, `6x6`, `8x8` | 0.965–1.012× |
| choleskyBatch16 | MatView | f64 | `3x3`, `4x4` | 0.966–1.000× |
| choleskyReuse | Mat | f32 | `2`, `3`, `4`, `6`, `8` | 0.989–1.006× |
| choleskyReuse | Mat | f64 | `2`, `3`, `4` | 0.994–0.998× |
| choleskyReuse | MatView | f32 | `2`, `3`, `4`, `6`, `8`, `16`, `32`, `64`, `128`, `256`, `512` | 0.895–1.024× |
| choleskyReuse | MatView | f64 | `2`, `3`, `4`, `6`, `8`, `32`, `64`, `128`, `256`, `512` | 0.878–1.006× |
| det | Mat | f32 | `2x2`, `4x4` | 0.988–0.998× |
| det | Mat | f64 | `3x3`, `4x4` | 0.924–1.007× |
| div | slice | f64 | `3` | 0.920–0.920× |
| divMatrix | MatView | f32 | `3` | 1.085–1.085× |
| divMatrix | MatView | f64 | `3` | 1.121–1.121× |
| dot | Vec | f32 | `2`, `3`, `4`, `6`, `8`, `12` | 0.944–1.110× |
| dot | Vec | f64 | `2`, `3`, `4`, `6` | 0.944–1.100× |
| dot | VecView | f32 | `1`, `3`, `7` | 0.968–1.001× |
| dot | VecView | f64 | `1`, `3`, `7` | 1.052–1.130× |
| dotBatch3 | slice | f32 | `3`, `33` | 0.846–0.987× |
| dotBatch3 | slice | f64 | `3`, `33` | 0.846–1.028× |
| factorCholesky | Mat | f32 | `2x2`, `3x3`, `6x6`, `12x12` | 0.871–1.059× |
| factorCholesky | Mat | f64 | `2x2`, `4x4`, `6x6`, `12x12` | 0.841–1.014× |
| factorCholesky | MatView | f32 | `2x2`, `3x3`, `6x6`, `8x8`, `12x12`, `16`, `64`, `256`, `512` | 0.958–1.158× |
| factorCholesky | MatView | f64 | `4x4`, `6x6`, `8x8`, `12x12`, `16`, `32`, `64`, `256`, `512` | 0.945–1.118× |
| factorLU | Mat | f32 | `2x2`, `3x3`, `4x4`, `6x6` | 0.968–1.125× |
| factorLU | Mat | f64 | `2x2`, `4x4`, `6x6`, `8x8`, `12x12` | 0.922–1.154× |
| factorLU | MatView | f32 | `2x2`, `3x3`, `4x4` | 0.950–1.120× |
| factorLU | MatView | f64 | `2x2`, `3x3` | 0.996–0.998× |
| fill | slice | f32 | `33` | 0.990–0.990× |
| fill | slice | f64 | `3`, `33` | 1.012–1.075× |
| fillMatrix | MatView | f32 | `3`, `128`, `512` | 0.981–1.004× |
| fillMatrix | MatView | f64 | `3`, `512` | 0.965–1.006× |
| gauss | Mat | f32 | `2`, `3`, `4`, `6`, `8`, `12`, `6x6.signed_alternating`, `3x3.dense_random`, `12x12.dense_random` | 0.883–1.133× |
| gauss | Mat | f64 | `2`, `3`, `4`, `6`, `8`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `12x12.dense_random` | 0.915–1.079× |
| gauss | MatView | f32 | `2`, `3`, `6`, `8`, `3x3.pivot_required`, `6x6.pivot_required`, `6x6.signed_alternating`, `6x6.dense_random` | 0.898–1.122× |
| gauss | MatView | f64 | `3`, `4`, `6`, `6x6.pivot_required`, `3x3.dense_random` | 1.049–1.190× |
| inv | Mat | f32 | `2x2` | 0.988–0.988× |
| inv | Mat | f64 | `2x2`, `3x3` | 1.002–1.078× |
| lu | Mat | f32 | `2`, `3`, `4`, `6`, `12`, `3x3.pivot_required`, `6x6.pivot_required`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 0.884–1.131× |
| lu | Mat | f64 | `2`, `3`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `12x12.signed_alternating`, `3x3.dense_random`, `12x12.dense_random` | 0.873–1.190× |
| lu | MatView | f32 | `2`, `3`, `6`, `8`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `6x6.dense_random` | 0.879–1.143× |
| lu | MatView | f64 | `2`, `3`, `4`, `6`, `8`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random` | 0.983–1.119× |
| luBatch16 | Mat | f32 | `2x2`, `3x3`, `4x4`, `6x6`, `8x8` | 0.975–1.003× |
| luBatch16 | Mat | f64 | `2x2`, `3x3`, `4x4` | 0.988–1.009× |
| luBatch16 | MatView | f32 | `2x2`, `3x3`, `4x4`, `8x8` | 1.006–1.052× |
| luBatch16 | MatView | f64 | `2x2`, `3x3`, `4x4`, `6x6`, `12x12` | 0.884–1.079× |
| luReuse | Mat | f32 | `2`, `3`, `4`, `6`, `8` | 0.990–1.002× |
| luReuse | Mat | f64 | `2`, `3`, `4` | 0.983–1.005× |
| luReuse | MatView | f32 | `2`, `3`, `4`, `6`, `8`, `16`, `32`, `64`, `128`, `256`, `512` | 0.941–1.052× |
| luReuse | MatView | f64 | `2`, `3`, `4`, `6`, `8`, `16`, `32`, `64`, `256`, `512` | 0.894–1.186× |
| matmul | Mat | f32 | `2`, `3`, `4`, `8`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 0.990–1.047× |
| matmul | Mat | f64 | `2`, `3`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 0.986–1.014× |
| matmul | MatView | f32 | `2` | 0.883–0.883× |
| matmulRect | Mat | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 1.002–1.029× |
| matmulRect | Mat | f64 | `2x3x4`, `3x2x3`, `4x8x3`, `12x6x3` | 0.942–1.123× |
| matrixAllFinite | MatView | f32 | `3` | 1.114–1.114× |
| matrixAllFinite | MatView | f64 | `3` | 1.118–1.118× |
| matrixMaxAbs | MatView | f64 | `3` | 0.942–0.942× |
| matrixNorm | MatView | f32 | `3` | 1.007–1.007× |
| matrixNorm | MatView | f64 | `3` | 1.003–1.003× |
| matvec | Mat | f32 | `2`, `3`, `4`, `6`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random` | 0.997–1.020× |
| matvec | Mat | f64 | `2`, `3`, `4`, `6`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random` | 0.993–1.078× |
| matvec | MatView | f32 | `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random` | 0.922–1.157× |
| matvec | MatView | f64 | `8` | 1.008–1.008× |
| matvecRect | Mat | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 0.829–1.079× |
| matvecRect | Mat | f64 | `2x3x4`, `3x2x3` | 0.988–1.027× |
| matvecRect | MatView | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 1.005–1.100× |
| matvecRect | MatView | f64 | `2x3x4`, `3x2x3` | 0.991–1.005× |
| max | slice | f32 | `1`, `3`, `16`, `31` | 0.942–1.033× |
| max | slice | f64 | `3`, `7`, `16` | 0.994–1.003× |
| maxAbs | slice | f32 | `33` | 1.134–1.134× |
| maxAbs | slice | f64 | `3` | 0.942–0.942× |
| mean | slice | f32 | `3` | 0.995–0.995× |
| mean | slice | f64 | `3` | 1.032–1.032× |
| min | slice | f64 | `33` | 1.115–1.115× |
| mul | slice | f32 | `1`, `16`, `31` | 0.887–0.996× |
| mul | slice | f64 | `1`, `3`, `7`, `16`, `31` | 0.907–1.028× |
| mulMatrix | MatView | f32 | `3` | 1.076–1.076× |
| mulMatrix | MatView | f64 | `3` | 1.112–1.112× |
| normL1 | slice | f32 | `3` | 0.973–0.973× |
| normL1 | slice | f64 | `3` | 1.002–1.002× |
| normL2 | Vec | f32 | `2`, `3`, `6`, `8`, `12` | 0.981–1.084× |
| normL2 | Vec | f64 | `2`, `3`, `4`, `6` | 0.980–1.040× |
| normL2 | slice | f32 | `3` | 0.977–0.977× |
| normL2 | slice | f64 | `3` | 0.940–0.940× |
| normL2Square | slice | f64 | `3` | 0.946–0.946× |
| outer | Mat | f32 | `3x3`, `6x6`, `12x12` | 1.034–1.068× |
| outer | Mat | f64 | `6x6`, `12x12` | 1.033–1.034× |
| outer | MatView | f32 | `3x3`, `3` | 1.104–1.120× |
| outer | MatView | f64 | `6x6`, `3` | 1.028–1.126× |
| outerAdd | Mat | f32 | `3`, `4`, `6`, `8`, `12` | 0.931–1.020× |
| outerAdd | Mat | f64 | `2`, `6`, `8` | 0.960–1.066× |
| outerAdd | MatView | f32 | `2`, `3`, `4` | 0.976–1.166× |
| outerAdd | MatView | f64 | `2`, `3`, `4` | 1.031–1.168× |
| permute | slice | f32 | `33`, `65537` | 0.939–1.002× |
| permute | slice | f64 | `3`, `33`, `65537` | 0.957–1.006× |
| rowops | Mat | f32 | `3`, `4`, `6`, `12` | 0.989–1.066× |
| rowops | Mat | f64 | `2`, `6`, `8` | 0.997–1.210× |
| rowops | MatView | f32 | `2`, `3`, `6`, `8`, `12` | 0.946–0.992× |
| rowops | MatView | f64 | `3`, `4`, `6`, `8`, `12` | 1.002–1.063× |
| rowopsRect | Mat | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 0.947–1.007× |
| rowopsRect | Mat | f64 | `2x3x4`, `6x12x8`, `12x6x3` | 0.931–1.093× |
| rowopsRect | MatView | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 1.005–1.090× |
| rowopsRect | MatView | f64 | `3x2x3`, `6x12x8`, `12x6x3` | 1.009–1.111× |
| scale | slice | f32 | `33` | 0.999–0.999× |
| scale | slice | f64 | `3` | 0.968–0.968× |
| scaleMatrix | MatView | f32 | `3` | 1.013–1.013× |
| scaleMatrix | MatView | f64 | `3` | 0.984–0.984× |
| scaleRow | MatView | f32 | `33`, `128`, `512` | 0.990–1.167× |
| scaleRow | MatView | f64 | `3`, `33`, `128`, `512` | 0.947–1.080× |
| sub | slice | f32 | `33` | 1.121–1.121× |
| sub | slice | f64 | `3` | 0.879–0.879× |
| subMatrix | MatView | f32 | `3` | 1.084–1.084× |
| subMatrix | MatView | f64 | `3` | 1.126–1.126× |
| sum | slice | f32 | `1`, `3`, `16` | 0.994–0.999× |
| sum | slice | f64 | `1`, `3`, `7`, `16` | 0.946–1.003× |
| trace | MatView | f32 | `3`, `33` | 1.004–1.005× |
| trace | MatView | f64 | `3`, `33` | 0.959–1.049× |
| transpose | Mat | f32 | `3`, `4` | 0.995–1.019× |
| transpose | Mat | f64 | `2`, `3`, `8` | 1.004–1.085× |
| transpose | MatView | f32 | `2`, `3`, `4`, `6` | 0.965–1.140× |
| transpose | MatView | f64 | `2`, `3`, `6`, `12`, `16`, `32`, `127` | 0.924–1.132× |
| transposeRect | Mat | f32 | `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 0.962–1.058× |
| transposeRect | Mat | f64 | `2x3x4`, `3x2x3` | 0.972–1.049× |
| transposeRect | MatView | f32 | `2x3x4`, `6x12x8`, `12x6x3` | 0.973–1.027× |
| transposeRect | MatView | f64 | `2x3x4`, `3x2x3` | 1.048–1.072× |

#### Faster (434 cases)

| Operation | Container | Dtype | Case variants | Median speedup range |
|---|---|---|---|---:|
| add | slice | f32 | `7`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.009–4.695× |
| add | slice | f64 | `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.347–3.574× |
| addMatrix | MatView | f32 | `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 2.099–7.000× |
| addMatrix | MatView | f64 | `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.345–3.698× |
| addScaledRow | MatView | f64 | `33` | 1.090–1.090× |
| allFinite | slice | f32 | `33`, `4096`, `65537` | 1.884–7.955× |
| allFinite | slice | f64 | `33`, `4096`, `65537` | 1.666–3.934× |
| axpy | Vec | f32 | `2` | 1.081–1.081× |
| axpy | slice | f32 | `4096`, `65537` | 2.912–3.442× |
| axpy | slice | f64 | `4096`, `65537` | 1.697–1.891× |
| cholesky | Mat | f64 | `3x3.signed_alternating` | 1.058–1.058× |
| cholesky | MatView | f32 | `32` | 1.063–1.063× |
| cholesky | MatView | f64 | `128` | 1.104–1.104× |
| div | slice | f32 | `33`, `4096`, `65537` | 1.518–7.702× |
| div | slice | f64 | `33`, `4096`, `65537` | 2.058–4.017× |
| divMatrix | MatView | f32 | `33`, `128`, `512` | 6.816–8.043× |
| divMatrix | MatView | f64 | `33`, `128`, `512` | 3.960–4.033× |
| dot | Vec | f32 | `16` | 1.411–1.411× |
| dot | Vec | f64 | `8`, `12`, `16` | 1.154–1.403× |
| dot | VecView | f32 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.354–10.337× |
| dot | VecView | f64 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.529–4.965× |
| factorCholesky | Mat | f32 | `8x8` | 1.131–1.131× |
| factorCholesky | MatView | f32 | `32`, `128` | 1.094–1.162× |
| factorCholesky | MatView | f64 | `128` | 1.105–1.105× |
| factorLU | Mat | f32 | `8x8`, `12x12` | 1.035–1.324× |
| factorLU | MatView | f32 | `6x6`, `8x8`, `12x12`, `16`, `32`, `64`, `128`, `256`, `512` | 1.117–4.065× |
| factorLU | MatView | f64 | `4x4`, `6x6`, `8x8`, `12x12`, `16`, `32`, `64`, `128`, `256`, `512` | 1.193–2.192× |
| fill | slice | f32 | `4096`, `65537` | 1.126–1.586× |
| fill | slice | f64 | `4096`, `65537` | 1.061–1.378× |
| fillMatrix | MatView | f32 | `33` | 1.355–1.355× |
| fillMatrix | MatView | f64 | `33` | 1.269–1.269× |
| gauss | Mat | f32 | `12x12.pivot_required`, `12x12.signed_alternating` | 1.130–1.183× |
| gauss | Mat | f64 | `12`, `12x12.pivot_required` | 1.071–1.172× |
| gauss | MatView | f32 | `4`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random` | 1.067–1.243× |
| gauss | MatView | f64 | `2`, `8`, `12`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 1.007–1.256× |
| inv | Mat | f32 | `4x4` | 1.230–1.230× |
| lu | Mat | f32 | `8`, `12x12.signed_alternating` | 1.041–1.093× |
| lu | Mat | f64 | `4`, `3x3.pivot_required`, `3x3.signed_alternating` | 1.194–1.237× |
| lu | MatView | f32 | `4`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random`, `16`, `32`, `64`, `128`, `256`, `512` | 1.085–3.707× |
| lu | MatView | f64 | `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random`, `16`, `32`, `64`, `128`, `256`, `512` | 1.173–2.112× |
| luBatch16 | MatView | f32 | `6x6` | 1.123–1.123× |
| luReuse | MatView | f64 | `128` | 1.139–1.139× |
| matmul | MatView | f32 | `8`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512` | 2.219–63.396× |
| matmul | MatView | f64 | `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512` | 1.625–27.934× |
| matmulRect | MatView | f32 | `6x12x8`, `32x128x16`, `128x32x64` | 5.549–15.455× |
| matmulRect | MatView | f64 | `2x3x4`, `6x12x8`, `32x128x16`, `128x32x64` | 1.305–8.836× |
| matrixAllFinite | MatView | f32 | `33`, `128`, `512` | 4.569–5.174× |
| matrixAllFinite | MatView | f64 | `33`, `128`, `512` | 2.879–3.029× |
| matrixMaxAbs | MatView | f32 | `33`, `128`, `512` | 3.199–3.371× |
| matrixMaxAbs | MatView | f64 | `33`, `128`, `512` | 1.599–1.627× |
| matrixNorm | MatView | f32 | `33`, `128`, `512` | 7.974–8.835× |
| matrixNorm | MatView | f64 | `33`, `128`, `512` | 3.987–4.185× |
| matvec | Mat | f64 | `8`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random` | 1.264–2.180× |
| matvec | MatView | f32 | `2`, `3`, `4`, `6`, `3x3.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.255–11.156× |
| matvec | MatView | f64 | `2`, `3`, `4`, `6`, `12`, `3x3.pivot_required`, `6x6.pivot_required`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.173–5.211× |
| matvecRect | Mat | f64 | `4x8x3`, `6x12x8`, `12x6x3` | 1.087–2.029× |
| matvecRect | MatView | f64 | `4x8x3`, `6x12x8`, `12x6x3` | 1.075–2.173× |
| max | slice | f32 | `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.549–9.101× |
| max | slice | f64 | `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.263–3.900× |
| maxAbs | slice | f32 | `4096`, `65537` | 3.208–3.294× |
| maxAbs | slice | f64 | `33`, `4096`, `65537` | 1.047–1.612× |
| mean | slice | f32 | `33`, `4096`, `65537` | 1.633–8.678× |
| mean | slice | f64 | `33`, `4096`, `65537` | 1.784–4.158× |
| min | slice | f32 | `3`, `33`, `4096`, `65537` | 1.105–9.419× |
| min | slice | f64 | `3`, `4096`, `65537` | 1.104–4.446× |
| mul | slice | f32 | `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.479–5.267× |
| mul | slice | f64 | `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.320–3.257× |
| mulMatrix | MatView | f32 | `33`, `128`, `512` | 2.509–6.427× |
| mulMatrix | MatView | f64 | `33`, `128`, `512` | 1.378–3.387× |
| normL1 | slice | f32 | `33`, `4096`, `65537` | 1.529–8.762× |
| normL1 | slice | f64 | `33`, `4096`, `65537` | 1.565–4.188× |
| normL2 | Vec | f32 | `16` | 1.403–1.403× |
| normL2 | Vec | f64 | `8`, `12`, `16` | 1.154–1.444× |
| normL2 | slice | f32 | `33`, `4096`, `65537` | 1.647–8.393× |
| normL2 | slice | f64 | `33`, `4096`, `65537` | 1.701–4.093× |
| normL2Square | slice | f32 | `3`, `33`, `4096`, `65537` | 1.040–8.440× |
| normL2Square | slice | f64 | `33`, `4096`, `65537` | 1.380–4.184× |
| outer | Mat | f64 | `3x3` | 1.048–1.048× |
| outer | MatView | f32 | `6x6`, `12x12`, `33`, `128`, `512` | 1.208–7.981× |
| outer | MatView | f64 | `3x3`, `12x12`, `33`, `128`, `512` | 1.082–4.162× |
| outerAdd | Mat | f32 | `2` | 1.065–1.065× |
| outerAdd | Mat | f64 | `3`, `12` | 1.237–1.263× |
| outerAdd | MatView | f32 | `6`, `8`, `12`, `33`, `128`, `512` | 1.277–4.332× |
| outerAdd | MatView | f64 | `6`, `8`, `12`, `33`, `128`, `512` | 1.252–2.494× |
| permute | slice | f32 | `4096` | 1.072–1.072× |
| permute | slice | f64 | `4096` | 1.078–1.078× |
| rowops | Mat | f32 | `2`, `8` | 1.040–1.089× |
| rowops | Mat | f64 | `3`, `4`, `12` | 1.064–1.190× |
| rowops | MatView | f64 | `2` | 1.015–1.015× |
| rowopsRect | Mat | f64 | `3x2x3`, `4x8x3` | 1.176–1.191× |
| scale | slice | f32 | `4096`, `65537` | 3.944–6.701× |
| scale | slice | f64 | `33`, `4096`, `65537` | 1.031–1.971× |
| scaleMatrix | MatView | f32 | `33`, `128`, `512` | 3.346–7.330× |
| scaleMatrix | MatView | f64 | `33`, `128`, `512` | 1.677–3.775× |
| stream | NDArray | f32 | `256`, `32x32`, `8x8x8`, `4x8x8x8`, `2x4x8x8x8`, `8x8x16`, `4x8x8x4` | 6.152–11.473× |
| stream | NDArray | f64 | `256`, `32x32`, `8x8x8`, `4x8x8x8`, `2x4x8x8x8`, `8x8x16`, `4x8x8x4` | 4.160–5.581× |
| sub | slice | f32 | `4096`, `65537` | 2.882–2.966× |
| sub | slice | f64 | `33`, `4096`, `65537` | 1.174–1.557× |
| subMatrix | MatView | f32 | `33`, `128`, `512` | 2.524–6.691× |
| subMatrix | MatView | f64 | `33`, `128`, `512` | 1.398–3.473× |
| sum | slice | f32 | `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.320–11.301× |
| sum | slice | f64 | `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.436–6.811× |
| trace | MatView | f32 | `128`, `512` | 1.496–1.799× |
| trace | MatView | f64 | `128`, `512` | 1.683–2.307× |
| transpose | Mat | f32 | `2`, `6` | 1.056–1.345× |
| transpose | Mat | f64 | `12` | 1.144–1.144× |
| transpose | MatView | f32 | `128`, `256`, `512`, `1024` | 2.893–18.667× |
| transpose | MatView | f64 | `4`, `64`, `128`, `256`, `512`, `1024` | 1.176–5.781× |
| transposeRect | MatView | f32 | `3x2x3`, `4x8x3` | 1.041–1.079× |

### Mojo

#### Slower (78 cases)

| Operation | Container | Dtype | Case variants | Median speedup range |
|---|---|---|---|---:|
| addScaledRow | MatView | f32 | `33`, `128` | 0.809–0.960× |
| choleskyBatch16 | Mat | f32 | `2x2`, `3x3`, `4x4`, `12x12` | 0.639–0.889× |
| choleskyBatch16 | Mat | f64 | `2x2`, `6x6`, `8x8`, `12x12` | 0.613–0.852× |
| choleskyBatch16 | MatView | f32 | `6x6` | 0.979–0.979× |
| choleskyBatch16 | MatView | f64 | `6x6`, `8x8`, `12x12` | 0.670–0.807× |
| choleskyReuse | Mat | f32 | `2`, `3`, `12` | 0.833–0.936× |
| choleskyReuse | Mat | f64 | `2`, `6`, `8`, `12` | 0.819–0.902× |
| choleskyReuse | MatView | f64 | `16` | 0.888–0.888× |
| det | Mat | f32 | `3x3` | 0.732–0.732× |
| dot | VecView | f32 | `1`, `3`, `7` | 0.738–0.790× |
| dot | VecView | f64 | `1` | 0.827–0.827× |
| inv | Mat | f32 | `3x3` | 0.579–0.579× |
| inv | Mat | f64 | `3x3` | 0.633–0.633× |
| luBatch16 | Mat | f32 | `2x2`, `3x3`, `4x4` | 0.854–0.921× |
| luBatch16 | Mat | f64 | `2x2` | 0.907–0.907× |
| luBatch16 | MatView | f64 | `12x12` | 0.902–0.902× |
| luReuse | Mat | f64 | `12` | 0.902–0.902× |
| matmul | Mat | f32 | `4` | 0.861–0.861× |
| matmul | MatView | f32 | `2`, `3`, `4`, `6`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random` | 0.625–0.802× |
| matmul | MatView | f64 | `2`, `3`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 0.823–0.856× |
| matmulRect | Mat | f32 | `3x2x3`, `12x6x3` | 0.708–0.843× |
| matmulRect | MatView | f32 | `2x3x4`, `3x2x3`, `4x8x3`, `12x6x3` | 0.625–0.726× |
| matmulRect | MatView | f64 | `3x2x3`, `4x8x3`, `12x6x3` | 0.680–0.835× |
| matvec | MatView | f32 | `2`, `6x6.pivot_required`, `3x3.signed_alternating` | 0.951–0.978× |
| matvecRect | Mat | f32 | `12x6x3` | 0.718–0.718× |
| matvecRect | MatView | f64 | `3x2x3` | 0.942–0.942× |
| min | slice | f32 | `3` | 0.949–0.949× |
| outerAdd | Mat | f64 | `8` | 0.843–0.843× |
| rowops | Mat | f64 | `8` | 0.882–0.882× |
| rowopsRect | Mat | f64 | `3x2x3`, `12x6x3` | 0.646–0.648× |
| rowopsRect | MatView | f64 | `3x2x3`, `12x6x3` | 0.646–0.665× |
| scaleRow | MatView | f32 | `33` | 0.887–0.887× |
| transposeRect | Mat | f32 | `6x12x8` | 0.858–0.858× |
| transposeRect | MatView | f32 | `2x3x4`, `3x2x3`, `4x8x3` | 0.776–0.904× |
| transposeRect | MatView | f64 | `3x2x3` | 0.981–0.981× |

#### Same / overlapping (290 cases)

| Operation | Container | Dtype | Case variants | Median speedup range |
|---|---|---|---|---:|
| add | slice | f32 | `1`, `3`, `7` | 1.047–1.133× |
| add | slice | f64 | `1`, `3`, `1048576` | 1.071–1.332× |
| addScaledRow | MatView | f32 | `3`, `512` | 0.991–1.086× |
| addScaledRow | MatView | f64 | `33`, `128`, `512` | 0.998–1.062× |
| allFinite | slice | f64 | `3` | 1.075–1.075× |
| axpy | slice | f32 | `3` | 1.177–1.177× |
| cholesky | Mat | f32 | `2`, `3`, `6`, `3x3.signed_alternating` | 1.023–1.106× |
| cholesky | Mat | f64 | `3`, `6`, `8`, `12`, `6x6.signed_alternating`, `12x12.signed_alternating` | 0.992–1.046× |
| cholesky | MatView | f32 | `2`, `3`, `3x3.signed_alternating`, `12x12.signed_alternating`, `16`, `512` | 1.084–1.168× |
| cholesky | MatView | f64 | `3`, `6`, `8`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `16`, `512` | 1.009–1.192× |
| choleskyBatch16 | Mat | f32 | `6x6`, `8x8` | 1.016–1.021× |
| choleskyBatch16 | Mat | f64 | `3x3`, `4x4` | 1.002–1.015× |
| choleskyBatch16 | MatView | f32 | `2x2`, `3x3`, `4x4`, `8x8`, `12x12` | 0.903–1.002× |
| choleskyBatch16 | MatView | f64 | `2x2`, `3x3`, `4x4` | 0.945–0.980× |
| choleskyReuse | Mat | f32 | `4`, `6`, `8` | 0.972–1.008× |
| choleskyReuse | Mat | f64 | `4` | 0.985–0.985× |
| choleskyReuse | MatView | f32 | `4`, `6`, `8`, `12`, `16`, `32`, `64`, `128` | 0.939–1.121× |
| choleskyReuse | MatView | f64 | `6`, `8`, `12`, `32`, `64`, `128`, `512` | 0.903–1.187× |
| det | Mat | f64 | `3x3` | 1.006–1.006× |
| div | slice | f64 | `3` | 1.049–1.049× |
| dot | VecView | f64 | `3` | 0.945–0.945× |
| dotBatch3 | slice | f32 | `3`, `33`, `4096`, `65537` | 1.001–1.158× |
| dotBatch3 | slice | f64 | `3`, `65537` | 1.079–1.097× |
| factorCholesky | Mat | f32 | `2x2`, `3x3` | 1.016–1.059× |
| factorCholesky | Mat | f64 | `3x3`, `6x6`, `8x8` | 0.991–1.147× |
| factorCholesky | MatView | f32 | `2x2`, `3x3`, `512` | 1.050–1.122× |
| factorCholesky | MatView | f64 | `2x2`, `3x3`, `4x4`, `6x6`, `12x12`, `16` | 1.043–1.157× |
| factorLU | Mat | f32 | `2x2`, `3x3` | 1.037–1.037× |
| factorLU | Mat | f64 | `2x2`, `3x3` | 1.099–1.167× |
| factorLU | MatView | f32 | `2x2`, `3x3` | 1.090–1.218× |
| fillMatrix | MatView | f64 | `3` | 1.194–1.194× |
| gauss | Mat | f32 | `2`, `3`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 0.958–1.099× |
| gauss | Mat | f64 | `2`, `3`, `3x3.pivot_required`, `3x3.dense_random` | 0.988–1.051× |
| gauss | MatView | f32 | `2`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 1.112–1.273× |
| gauss | MatView | f64 | `3`, `3x3.pivot_required` | 1.116–1.180× |
| inv | Mat | f32 | `2x2` | 1.019–1.019× |
| inv | Mat | f64 | `2x2` | 1.067–1.067× |
| lu | Mat | f32 | `2`, `3`, `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 1.023–1.050× |
| lu | Mat | f64 | `2`, `3`, `3x3.pivot_required`, `6x6.pivot_required` | 1.016–1.199× |
| lu | MatView | f32 | `2`, `3`, `3x3.pivot_required`, `6x6.pivot_required` | 1.094–1.206× |
| lu | MatView | f64 | `3`, `3x3.pivot_required`, `3x3.signed_alternating` | 1.058–1.149× |
| luBatch16 | Mat | f32 | `6x6`, `8x8`, `12x12` | 0.967–1.010× |
| luBatch16 | Mat | f64 | `3x3`, `4x4`, `6x6`, `8x8`, `12x12` | 0.892–1.010× |
| luBatch16 | MatView | f32 | `2x2`, `3x3`, `4x4`, `6x6`, `8x8`, `12x12` | 0.920–1.012× |
| luBatch16 | MatView | f64 | `2x2`, `3x3`, `4x4`, `6x6`, `8x8` | 0.896–0.967× |
| luReuse | Mat | f32 | `2`, `3`, `4`, `6`, `8`, `12` | 0.943–1.004× |
| luReuse | Mat | f64 | `2`, `3`, `4`, `6`, `8` | 0.901–1.021× |
| luReuse | MatView | f32 | `4`, `6`, `8`, `12`, `16`, `32`, `64`, `128`, `512` | 0.959–1.119× |
| luReuse | MatView | f64 | `3`, `6`, `8`, `12`, `16`, `32`, `64`, `128`, `256`, `512` | 0.937–1.182× |
| matmul | Mat | f32 | `3x3.pivot_required`, `3x3.signed_alternating`, `3x3.dense_random` | 1.083–1.087× |
| matmul | Mat | f64 | `3`, `3x3.pivot_required`, `3x3.signed_alternating` | 1.024–1.030× |
| matmulRect | Mat | f32 | `2x3x4` | 1.095–1.095× |
| matmulRect | Mat | f64 | `2x3x4`, `3x2x3`, `12x6x3` | 1.001–1.057× |
| matrixAllFinite | MatView | f64 | `3` | 1.105–1.105× |
| matvec | MatView | f32 | `3`, `4`, `6`, `3x3.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 0.955–1.184× |
| matvec | MatView | f64 | `2`, `3`, `4`, `3x3.pivot_required`, `6x6.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `6x6.dense_random` | 1.020–1.175× |
| matvecRect | Mat | f32 | `2x3x4`, `3x2x3` | 1.007–1.015× |
| matvecRect | Mat | f64 | `3x2x3` | 0.963–0.963× |
| matvecRect | MatView | f32 | `2x3x4`, `3x2x3` | 0.958–0.992× |
| matvecRect | MatView | f64 | `2x3x4`, `4x8x3` | 0.915–1.129× |
| max | slice | f32 | `1`, `3`, `7` | 0.950–0.966× |
| max | slice | f64 | `1`, `3`, `7` | 0.961–1.031× |
| maxAbs | slice | f32 | `3` | 1.037–1.037× |
| maxAbs | slice | f64 | `3` | 1.045–1.045× |
| mean | slice | f32 | `3` | 0.993–0.993× |
| mean | slice | f64 | `3` | 1.046–1.046× |
| min | slice | f64 | `3` | 0.994–0.994× |
| mul | slice | f32 | `1`, `3`, `7` | 1.085–1.119× |
| mul | slice | f64 | `1`, `3`, `7`, `16`, `1048576` | 1.055–1.255× |
| normL1 | slice | f32 | `3` | 1.107–1.107× |
| normL1 | slice | f64 | `3` | 1.015–1.015× |
| normL2 | slice | f32 | `3` | 0.950–0.950× |
| normL2 | slice | f64 | `3` | 1.071–1.071× |
| normL2Square | slice | f32 | `3` | 1.062–1.062× |
| normL2Square | slice | f64 | `3` | 1.075–1.075× |
| outer | Mat | f32 | `3x3` | 1.013–1.013× |
| outer | Mat | f64 | `3x3`, `6x6` | 0.973–1.026× |
| outer | MatView | f64 | `3x3` | 1.145–1.145× |
| outerAdd | Mat | f32 | `2`, `3`, `4` | 1.017–1.185× |
| outerAdd | Mat | f64 | `2`, `3` | 1.022–1.143× |
| outerAdd | MatView | f32 | `2` | 1.129–1.129× |
| permute | slice | f32 | `65537` | 1.153–1.153× |
| permute | slice | f64 | `3`, `65537` | 1.119–1.134× |
| rowops | Mat | f32 | `2`, `12` | 1.131–1.143× |
| rowops | Mat | f64 | `6` | 1.226–1.226× |
| rowopsRect | Mat | f32 | `2x3x4`, `3x2x3`, `6x12x8` | 0.839–1.009× |
| rowopsRect | Mat | f64 | `2x3x4` | 1.029–1.029× |
| rowopsRect | MatView | f32 | `2x3x4`, `3x2x3`, `6x12x8`, `12x6x3` | 0.967–1.163× |
| scaleRow | MatView | f32 | `128`, `512` | 0.987–1.006× |
| scaleRow | MatView | f64 | `33`, `128`, `512` | 1.006–1.020× |
| sub | slice | f64 | `3` | 1.068–1.068× |
| subMatrix | MatView | f64 | `3` | 1.140–1.140× |
| sum | slice | f32 | `1`, `3`, `7` | 0.881–0.941× |
| sum | slice | f64 | `1`, `3`, `7` | 0.938–1.186× |
| trace | MatView | f64 | `3` | 1.113–1.113× |
| transpose | Mat | f32 | `2`, `3` | 0.970–1.053× |
| transpose | Mat | f64 | `6` | 1.062–1.062× |
| transpose | MatView | f32 | `2`, `3`, `4` | 1.096–1.134× |
| transpose | MatView | f64 | `2`, `3` | 1.083–1.112× |
| transposeRect | Mat | f32 | `12x6x3` | 0.962–0.962× |
| transposeRect | MatView | f32 | `6x12x8`, `12x6x3` | 0.926–0.940× |
| transposeRect | MatView | f64 | `2x3x4` | 1.077–1.077× |

#### Faster (746 cases)

| Operation | Container | Dtype | Case variants | Median speedup range |
|---|---|---|---|---:|
| add | slice | f32 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.235–4.475× |
| add | slice | f64 | `7`, `16`, `31`, `64`, `257`, `1003`, `4096`, `65537` | 1.115–2.669× |
| addMatrix | MatView | f32 | `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 3.800–10.997× |
| addMatrix | MatView | f64 | `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.397–6.001× |
| addScaledRow | MatView | f64 | `3` | 1.110–1.110× |
| allFinite | slice | f32 | `3`, `33`, `4096`, `65537` | 1.083–8.059× |
| allFinite | slice | f64 | `33`, `4096`, `65537` | 1.341–3.446× |
| axpy | Vec | f32 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 1.119–3.290× |
| axpy | Vec | f64 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 1.101–1.672× |
| axpy | slice | f32 | `33`, `4096`, `65537` | 1.203–1.673× |
| axpy | slice | f64 | `3`, `33`, `4096`, `65537` | 1.160–1.616× |
| cholesky | Mat | f32 | `4`, `8`, `12`, `6x6.signed_alternating`, `12x12.signed_alternating` | 1.107–1.221× |
| cholesky | Mat | f64 | `2`, `4`, `3x3.signed_alternating` | 1.047–1.209× |
| cholesky | MatView | f32 | `4`, `6`, `8`, `12`, `6x6.signed_alternating`, `32`, `64`, `128`, `256` | 1.165–1.382× |
| cholesky | MatView | f64 | `2`, `4`, `12`, `32`, `64`, `128`, `256` | 1.015–1.355× |
| choleskyReuse | Mat | f64 | `3` | 1.031–1.031× |
| choleskyReuse | MatView | f32 | `2`, `3`, `256`, `512` | 1.023–1.328× |
| choleskyReuse | MatView | f64 | `2`, `3`, `4`, `256` | 1.119–1.229× |
| det | Mat | f32 | `2x2`, `4x4` | 1.162–1.481× |
| det | Mat | f64 | `2x2`, `4x4` | 1.213–1.585× |
| div | slice | f32 | `3`, `33`, `4096`, `65537` | 1.089–7.528× |
| div | slice | f64 | `33`, `4096`, `65537` | 1.720–3.968× |
| divMatrix | MatView | f32 | `3`, `33`, `128`, `512` | 1.233–7.814× |
| divMatrix | MatView | f64 | `3`, `33`, `128`, `512` | 1.227–4.229× |
| dot | Vec | f32 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 1.185–3.118× |
| dot | Vec | f64 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 1.180–1.506× |
| dot | VecView | f32 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.334–9.100× |
| dot | VecView | f64 | `7`, `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.251–5.067× |
| dotBatch3 | slice | f64 | `33`, `4096` | 1.034–1.175× |
| factorCholesky | Mat | f32 | `4x4`, `6x6`, `8x8`, `12x12` | 1.128–1.334× |
| factorCholesky | Mat | f64 | `2x2`, `4x4`, `12x12` | 1.054–1.167× |
| factorCholesky | MatView | f32 | `4x4`, `6x6`, `8x8`, `12x12`, `16`, `32`, `64`, `128`, `256` | 1.197–1.576× |
| factorCholesky | MatView | f64 | `8x8`, `32`, `64`, `128`, `256`, `512` | 1.068–1.414× |
| factorLU | Mat | f32 | `4x4`, `6x6`, `8x8`, `12x12` | 1.089–2.216× |
| factorLU | Mat | f64 | `4x4`, `6x6`, `8x8`, `12x12` | 1.489–1.906× |
| factorLU | MatView | f32 | `4x4`, `6x6`, `8x8`, `12x12`, `16`, `32`, `64`, `128`, `256`, `512` | 1.338–4.249× |
| factorLU | MatView | f64 | `2x2`, `3x3`, `4x4`, `6x6`, `8x8`, `12x12`, `16`, `32`, `64`, `128`, `256`, `512` | 1.225–2.573× |
| fill | slice | f32 | `3`, `33`, `4096`, `65537` | 1.219–4.010× |
| fill | slice | f64 | `3`, `33`, `4096`, `65537` | 1.175–1.978× |
| fillMatrix | MatView | f32 | `3`, `33`, `128`, `512` | 1.223–5.865× |
| fillMatrix | MatView | f64 | `33`, `128`, `512` | 2.896–2.926× |
| gauss | Mat | f32 | `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 1.142–1.763× |
| gauss | Mat | f64 | `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 1.054–1.594× |
| gauss | MatView | f32 | `3`, `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 1.163–1.764× |
| gauss | MatView | f64 | `2`, `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 1.160–1.686× |
| inv | Mat | f32 | `4x4` | 1.609–1.609× |
| inv | Mat | f64 | `4x4` | 1.228–1.228× |
| lu | Mat | f32 | `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 1.160–1.545× |
| lu | Mat | f64 | `4`, `6`, `8`, `12`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 1.034–1.466× |
| lu | MatView | f32 | `4`, `6`, `8`, `12`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random`, `16`, `32`, `64`, `128`, `256`, `512` | 1.164–3.857× |
| lu | MatView | f64 | `2`, `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random`, `16`, `32`, `64`, `128`, `256`, `512` | 1.143–2.308× |
| luReuse | MatView | f32 | `2`, `3`, `256` | 1.033–1.158× |
| luReuse | MatView | f64 | `2`, `4` | 1.116–1.178× |
| matmul | Mat | f32 | `2`, `3`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random` | 1.071–8.357× |
| matmul | Mat | f64 | `2`, `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 1.024–4.776× |
| matmul | MatView | f32 | `8`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `12x12.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512` | 1.598–55.591× |
| matmul | MatView | f64 | `4`, `6`, `8`, `12`, `6x6.pivot_required`, `12x12.pivot_required`, `6x6.signed_alternating`, `12x12.signed_alternating`, `6x6.dense_random`, `12x12.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512` | 1.429–28.242× |
| matmulRect | Mat | f32 | `4x8x3`, `6x12x8` | 1.513–4.855× |
| matmulRect | Mat | f64 | `4x8x3`, `6x12x8` | 1.650–2.940× |
| matmulRect | MatView | f32 | `6x12x8`, `32x128x16`, `128x32x64` | 6.485–23.350× |
| matmulRect | MatView | f64 | `2x3x4`, `6x12x8`, `32x128x16`, `128x32x64` | 1.389–11.809× |
| matrixAllFinite | MatView | f32 | `3`, `33`, `128`, `512` | 1.212–4.092× |
| matrixAllFinite | MatView | f64 | `33`, `128`, `512` | 3.391–3.949× |
| matrixMaxAbs | MatView | f32 | `3`, `33`, `128`, `512` | 1.226–16.011× |
| matrixMaxAbs | MatView | f64 | `3`, `33`, `128`, `512` | 1.212–8.029× |
| matrixNorm | MatView | f32 | `3`, `33`, `128`, `512` | 1.232–8.065× |
| matrixNorm | MatView | f64 | `3`, `33`, `128`, `512` | 1.231–3.986× |
| matvec | Mat | f32 | `2`, `3`, `4`, `6`, `8`, `12`, `3x3.pivot_required`, `6x6.pivot_required`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 1.115–4.160× |
| matvec | Mat | f64 | `2`, `3`, `4`, `6`, `8`, `12`, `3x3.pivot_required`, `6x6.pivot_required`, `12x12.pivot_required`, `3x3.signed_alternating`, `6x6.signed_alternating`, `12x12.signed_alternating`, `3x3.dense_random`, `6x6.dense_random`, `12x12.dense_random` | 1.139–2.472× |
| matvec | MatView | f32 | `8`, `12`, `12x12.signed_alternating`, `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.170–9.380× |
| matvec | MatView | f64 | `6`, `8`, `12`, `12x12.pivot_required`, `12x12.signed_alternating`, `3x3.dense_random`, `12x12.dense_random`, `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.018–4.522× |
| matvecRect | Mat | f32 | `4x8x3`, `6x12x8` | 2.031–2.726× |
| matvecRect | Mat | f64 | `2x3x4`, `4x8x3`, `6x12x8`, `12x6x3` | 1.044–2.477× |
| matvecRect | MatView | f32 | `4x8x3`, `6x12x8`, `12x6x3` | 1.216–2.017× |
| matvecRect | MatView | f64 | `6x12x8`, `12x6x3` | 1.049–1.427× |
| max | slice | f32 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.025–6.044× |
| max | slice | f64 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.172–3.049× |
| maxAbs | slice | f32 | `33`, `4096`, `65537` | 1.752–16.195× |
| maxAbs | slice | f64 | `33`, `4096`, `65537` | 1.658–7.932× |
| mean | slice | f32 | `33`, `4096`, `65537` | 1.830–8.420× |
| mean | slice | f64 | `33`, `4096`, `65537` | 2.230–4.086× |
| min | slice | f32 | `33`, `4096`, `65537` | 1.267–6.034× |
| min | slice | f64 | `33`, `4096`, `65537` | 1.531–2.971× |
| mul | slice | f32 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.298–5.735× |
| mul | slice | f64 | `31`, `64`, `257`, `1003`, `4096`, `65537` | 1.165–1.843× |
| mulMatrix | MatView | f32 | `3`, `33`, `128`, `512` | 1.213–9.959× |
| mulMatrix | MatView | f64 | `3`, `33`, `128`, `512` | 1.219–5.686× |
| normL1 | slice | f32 | `33`, `4096`, `65537` | 1.881–8.148× |
| normL1 | slice | f64 | `33`, `4096`, `65537` | 1.408–4.013× |
| normL2 | Vec | f32 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 1.111–2.949× |
| normL2 | Vec | f64 | `2`, `3`, `4`, `6`, `8`, `12`, `16` | 1.243–1.473× |
| normL2 | slice | f32 | `33`, `4096`, `65537` | 1.452–8.049× |
| normL2 | slice | f64 | `33`, `4096`, `65537` | 1.634–3.993× |
| normL2Square | slice | f32 | `33`, `4096`, `65537` | 1.393–8.049× |
| normL2Square | slice | f64 | `33`, `4096`, `65537` | 1.611–3.997× |
| outer | Mat | f32 | `6x6`, `12x12` | 1.233–1.444× |
| outer | Mat | f64 | `12x12` | 1.121–1.121× |
| outer | MatView | f32 | `3x3`, `6x6`, `12x12`, `3`, `33`, `128`, `512` | 1.155–7.977× |
| outer | MatView | f64 | `6x6`, `12x12`, `3`, `33`, `128`, `512` | 1.129–6.070× |
| outerAdd | Mat | f32 | `6`, `8`, `12` | 1.312–1.618× |
| outerAdd | Mat | f64 | `4`, `6`, `12` | 1.167–1.241× |
| outerAdd | MatView | f32 | `3`, `4`, `6`, `8`, `12`, `33`, `128`, `512` | 1.148–1.778× |
| outerAdd | MatView | f64 | `2`, `3`, `4`, `6`, `8`, `12`, `33`, `128`, `512` | 1.157–1.651× |
| permute | slice | f32 | `3`, `33`, `4096` | 1.158–1.738× |
| permute | slice | f64 | `33`, `4096` | 1.371–1.610× |
| rowops | Mat | f32 | `3`, `4`, `6`, `8` | 1.211–2.870× |
| rowops | Mat | f64 | `2`, `3`, `4`, `12` | 1.174–1.585× |
| rowops | MatView | f32 | `2`, `3`, `4`, `6`, `8`, `12` | 1.097–1.322× |
| rowops | MatView | f64 | `2`, `3`, `4`, `6`, `8`, `12` | 1.127–1.277× |
| rowopsRect | Mat | f32 | `4x8x3`, `12x6x3` | 1.297–1.516× |
| rowopsRect | Mat | f64 | `4x8x3`, `6x12x8` | 1.277–2.256× |
| rowopsRect | MatView | f32 | `4x8x3` | 1.525–1.525× |
| rowopsRect | MatView | f64 | `2x3x4`, `4x8x3`, `6x12x8` | 1.072–2.326× |
| scale | slice | f32 | `3`, `33`, `4096`, `65537` | 1.185–4.132× |
| scale | slice | f64 | `3`, `33`, `4096`, `65537` | 1.172–3.824× |
| scaleMatrix | MatView | f32 | `3`, `33`, `128`, `512` | 1.323–11.363× |
| scaleMatrix | MatView | f64 | `3`, `33`, `128`, `512` | 1.307–7.559× |
| scaleRow | MatView | f32 | `3` | 1.208–1.208× |
| scaleRow | MatView | f64 | `3` | 1.144–1.144× |
| stream | NDArray | f32 | `256`, `32x32`, `8x8x8`, `4x8x8x8`, `2x4x8x8x8`, `8x8x16`, `4x8x8x4` | 5.498–7.931× |
| stream | NDArray | f64 | `256`, `32x32`, `8x8x8`, `4x8x8x8`, `2x4x8x8x8`, `8x8x16`, `4x8x8x4` | 3.569–4.050× |
| sub | slice | f32 | `3`, `33`, `4096`, `65537` | 1.097–3.007× |
| sub | slice | f64 | `33`, `4096`, `65537` | 1.260–1.556× |
| subMatrix | MatView | f32 | `3`, `33`, `128`, `512` | 1.149–3.952× |
| subMatrix | MatView | f64 | `33`, `128`, `512` | 1.452–3.141× |
| sum | slice | f32 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.287–10.148× |
| sum | slice | f64 | `16`, `31`, `64`, `257`, `1003`, `4096`, `65537`, `1048576` | 1.236–4.221× |
| trace | MatView | f32 | `3`, `33`, `128`, `512` | 1.156–7.014× |
| trace | MatView | f64 | `33`, `128`, `512` | 3.407–6.379× |
| transpose | Mat | f32 | `4`, `6`, `8`, `12` | 1.183–3.584× |
| transpose | Mat | f64 | `2`, `3`, `4`, `8`, `12` | 1.150–1.556× |
| transpose | MatView | f32 | `6`, `8`, `12`, `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.211–8.082× |
| transpose | MatView | f64 | `4`, `6`, `8`, `12`, `16`, `32`, `64`, `127`, `128`, `256`, `512`, `1024` | 1.226–5.075× |
| transposeRect | Mat | f32 | `2x3x4`, `3x2x3`, `4x8x3` | 1.148–1.552× |
| transposeRect | Mat | f64 | `2x3x4`, `3x2x3`, `4x8x3`, `6x12x8`, `12x6x3` | 1.366–2.324× |
| transposeRect | MatView | f64 | `4x8x3`, `6x12x8`, `12x6x3` | 1.185–1.684× |

## Local source data and reproduction

- [Per-case CSV with exact medians and bounds](../benchmarks/results/simd-rerun-open.comparison.csv)
- [Full comparison JSON, including traversal controls](../benchmarks/results/simd-rerun-open.comparison.json)
- Benchmark command: `.venv/bin/python tools/run_simd_benchmarks.py --label-prefix simd-rerun-open --samples 10 --target-ms 10 --include-slow`

The raw benchmark outputs remain ignored by Git. This summary was explicitly requested for `dev/perf_simd.md`; no ignore rules or implementation defaults were changed.
