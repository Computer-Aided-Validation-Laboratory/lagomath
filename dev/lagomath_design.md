# Lagomath Numerical Toolbox Design

## 1. Purpose

Lagomath is a lightweight numerical toolbox for scientific and engineering software written in Zig.

It provides the common mathematical types, numerical kernels and low-level algorithms needed by projects such as:

* Pyvale;
* Riley;
* Felix;
* solid mechanics codes;
* experimental mechanics tools;
* sensor simulation;
* validation and uncertainty analysis;
* image-based mechanics.

Lagomath is **not** intended to implement complete domain workflows.

For example:

```text
Lagomath
    matrix/vector algebra
    numerical differentiation
    numerical integration
    polynomial fitting
    filtering and smoothing
    random sampling
    statistics
    mechanics tensor operations

Felix / Pyvale
    sensor simulation workflows

Riley
    rasterisation and synthetic imaging

DIC implementation
    image matching
    warp optimisation
    strain calculation workflow

Validation code
    validation methodology
    uncertainty propagation
    decision logic
```

The boundary is:

> Lagomath provides the reusable numerical machinery. Applications decide what that machinery means.

---

# 2. Design principles

Lagomath should remain:

```text
small
explicit
dependency-light
data-oriented
allocation-aware
performance-oriented
deterministic where required
usable without framework machinery
```

The normal implementation sequence for every substantial numerical feature is:

```text
correct scalar implementation
        ↓
verification
        ↓
representative scalar benchmark
        ↓
optional SIMD investigation
        ↓
scalar/SIMD correctness comparison
        ↓
representative A/B benchmark
        ↓
retain SIMD only where benefit is meaningful
        ↓
record result in dev/simd_perf_log.md
```

The **scalar implementation remains the default public implementation**.

SIMD is never silently substituted for scalar behaviour.

Fixed-size Stack types preserve compile-time dimensions and use value semantics.

Runtime-sized Slice types represent borrowed memory and use explicit output and scratch storage.

Do not convert Stack types into Slice types merely to reuse implementation in hot numerical paths.

---

# 3. SIMD policy

SIMD in Lagomath is **explicitly opt-in**.

Where a SIMD implementation is retained, expose it with a clear `SIMD` suffix.

For example:

```zig
solveCholeskyStack(...)
solveCholeskyStackSIMD(...)

outerAddStack(...)
outerAddStackSIMD(...)

dotStack(...)
dotStackSIMD(...)
```

The ordinary function without the suffix is always the scalar implementation.

Do not make:

```zig
solveCholeskyStack(...)
```

silently dispatch to SIMD.

Do not automatically select SIMD based on CPU features, matrix dimensions or build mode unless a future explicit API is designed for that behaviour.

The caller chooses.

---

# 4. Why SIMD must be opt-in

Lagomath is intended to be composed into larger numerical kernels.

A caller may already be vectorising over:

```text
DIC subsets
pixels
sensors
time samples
Monte Carlo samples
finite elements
integration points
experimental points
```

For example, an application may deliberately process several independent 6×6 scalar systems across SIMD lanes.

In that situation the desired hierarchy may be:

```text
outer application SIMD
        ↓
scalar Lagomath solve per lane/work item
```

rather than:

```text
outer application SIMD
        ↓
inner Lagomath SIMD
```

Automatically introducing SIMD inside Lagomath would remove this control and may generate a poorer execution strategy.

Therefore:

> **Lagomath provides scalar kernels by default and SIMD kernels as explicit alternatives.**

---

# 5. SIMD must earn its existence

Do not implement SIMD merely because an operation appears vectorisable.

For each candidate:

1. establish the scalar implementation;
2. verify scalar correctness;
3. create a representative benchmark;
4. implement the SIMD candidate;
5. verify numerical equivalence against scalar;
6. benchmark scalar versus SIMD;
7. use enough repetitions and samples to exceed measurement noise;
8. determine whether the improvement is meaningful;
9. retain or reject the SIMD implementation;
10. record the result.

It is completely acceptable for the result to be:

> SIMD was slower. Do not use this approach.

That is useful engineering information.

---

# 6. SIMD performance record

Create:

```text
dev/
└── simd_perf_log.md
```

This file is the permanent record of SIMD investigations in Lagomath.

Every attempted SIMD optimisation should receive a section, including unsuccessful ones.

The purpose is to prevent:

```text
repeating failed experiments
forgetting why scalar was chosen
making performance choices from intuition
losing benchmark context
```

---

# 7. SIMD performance-log format

Each feature should have a short section such as:

````markdown
## Cholesky Stack 6×6

### Feature

`solveCholeskyStackSIMD`

Scalar reference:

`solveCholeskyStack`

### Benchmark

`src/bench/bench_cholesky.zig`

Run with:

```bash
zig build bench-cholesky -Doptimize=ReleaseFast
````

Representative cases:

* `N = 3`, `f64`
* `N = 6`, `f64`
* `N = 8`, `f64`

Configuration:

* 10,000 warmup iterations
* 20 samples
* 100,000 solves per sample

### Result

For the representative 6×6 DIC case:

* scalar median: ...
* SIMD median: ...
* speedup: ...

### Decision

Retained / rejected.

### Notes

SIMD performed poorly because ...

````

Keep these records short and factual.

They are engineering notes, not papers.

---

# 8. Record poor SIMD results

Negative results must also be retained.

For example:

```markdown
## VecStack 3-element dot product

SIMD attempt was slower than scalar for all representative cases.

Likely causes:

- vector setup overhead;
- horizontal reduction cost;
- unused SIMD lanes.

The SIMD implementation was not retained.

Do not repeat this implementation without new evidence or a materially different approach.
````

This is important.

A rejected optimisation should not simply disappear from Git history and then be rediscovered six months later.

---

# 9. High-level package structure

The intended source tree can grow toward:

```text
src/
├── root.zig
│
├── ndarray.zig
├── matslice.zig
├── matstack.zig
├── vecslice.zig
├── vecstack.zig
│
├── arrayops.zig
├── vecops.zig
├── matops.zig
├── rowops.zig
├── tensorops.zig
│
├── linsolve.zig
├── gauss.zig
├── lu.zig
├── cholesky.zig
├── qr.zig
│
├── random.zig
├── distributions.zig
├── sampling.zig
├── stats.zig
│
├── differentiate.zig
├── integrate.zig
│
├── kernels.zig
├── filter.zig
│
├── polynomial.zig
├── polyfit.zig
│
├── interp.zig
├── geometry.zig
├── mechanics.zig
│
└── bench/
    ├── bench_cholesky.zig
    ├── bench_lu.zig
    ├── bench_polyfit.zig
    ├── bench_filter.zig
    └── ...
```

Development records live separately:

```text
dev/
└── simd_perf_log.md
```

Do not create every source file immediately.

Modules should appear when concrete cross-project functionality requires them.

---

# 10. Core data types

The core representations remain:

```zig
NDArray(T)

MatSlice(T)
MatStack(M, N, T)

VecSlice(T)
VecStack(N, T)
```

## Stack types

Use for fixed-size hot numerical work:

```text
compile-time dimensions
inline contiguous storage
pass by value
return by value
local fixed-size scratch
```

Typical examples:

```text
2D/3D vectors
3×3 deformation gradients
3×3 stress tensors
6×6 DIC Hessians
small coordinate transforms
```

## Slice types

Use for runtime-sized problems:

```text
borrowed memory
runtime dimensions
explicit caller-owned output
explicit scratch buffers
no hidden allocation
```

## NDArray

Use for general multidimensional contiguous scientific data:

```text
images
fields
sensor ensembles
time histories
Monte Carlo samples
spatial grids
```

Do not attempt to turn NDArray into a complete NumPy clone.

---

# 11. Vector operations

`vecops.zig` should own reusable vector mathematics.

Required operations include:

```text
dot product
squared norm
norm
min
max
arg min
arg max
max absolute value
sum
mean
all finite

add
subtract
multiply elementwise
divide elementwise
scale

AXPY
normalisation
distance
squared distance
projection
```

Where representation materially matters, expose:

```text
...Stack()
...Slice()
```

Stack implementations retain compile-time dimensions.

## SIMD stage

Once each scalar operation is correct and benchmarked, investigate SIMD for operations likely to benefit:

```text
dot
squared norm
AXPY
elementwise arithmetic
larger reductions
```

Retained implementations use explicit names such as:

```zig
dotStackSIMD()
axpyStackSIMD()
```

Do not retain SIMD implementations with no meaningful advantage.

Record all investigations in:

```text
dev/simd_perf_log.md
```

---

# 12. Matrix operations

`matops.zig` should provide:

```text
matrix × vector
matrix × matrix
transpose

outer product
accumulating outer product

trace
determinant
small explicit inverse

Frobenius norm
max absolute value
all finite

symmetric part
skew-symmetric part
```

Fixed-size multiplication must support:

```text
M×N × N×P → M×P
```

not only square matrices.

## SIMD stage

After scalar correctness, investigate SIMD for:

```text
matrix × vector
matrix × matrix
outer product
outerAdd
reductions
```

Pay particular attention to representative small engineering sizes:

```text
2×2
3×3
4×4
6×6
```

Do not assume that explicit SIMD helps these cases.

Retained alternatives use names such as:

```zig
mulMatStackSIMD()
outerAddStackSIMD()
```

The scalar forms remain the default.

---

# 13. Linear equation solving

Lagomath should initially provide:

```text
Gaussian elimination with partial pivoting
LU with partial pivoting
Cholesky for SPD systems
```

with separate Stack and Slice implementations.

Public scalar API:

```text
solveGaussStack()
solveGaussSlice()

solveLUStack()
solveLUSlice()

solveCholeskyStack()
solveCholeskySlice()
```

Factorisations should also be reusable:

```text
factorLU...
solveLUFactored...

factorCholesky...
solveCholeskyFactored...
```

Stack interfaces favour values in and values out.

Slice interfaces use explicit scratch/output memory.

## Solver SIMD stage

Only after each solver's scalar implementation is complete and independently verified should SIMD be investigated.

Possible SIMD alternatives:

```zig
solveGaussStackSIMD()
solveGaussSliceSIMD()

solveLUStackSIMD()
solveLUSliceSIMD()

solveCholeskyStackSIMD()
solveCholeskySliceSIMD()
```

The solver control flow may remain scalar while SIMD is used only for arithmetic sub-kernels.

For example:

```text
pivot selection         scalar
control flow            scalar
row update              SIMD candidate
dot product             SIMD candidate
triangular inner loop   SIMD candidate
```

This is preferable to forcing the entire algorithm into vector form.

---

# 14. Solver benchmarking

Benchmark solver SIMD against scalar using representative engineering cases.

At minimum:

```text
2×2
3×3
6×6
8×8
12×12
```

The 6×6 `f64` case is especially important for 2D DIC.

Benchmarks live in:

```text
src/bench/
```

using:

```text
bench_CASE.zig
```

for example:

```text
bench_gauss.zig
bench_lu.zig
bench_cholesky.zig
```

File-scope constants control:

```text
matrix size
numeric type
warmup iterations
number of samples
operations per sample
enabled cases
```

Example:

```zig
const N: usize = 6;
const T = f64;

const NUM_WARMUP: usize = 10_000;
const NUM_SAMPLES: usize = 20;
const NUM_RUNS_PER_SAMPLE: usize = 100_000;
```

Do not include allocations or input construction in the timed region.

---

# 15. QR factorisation

Add QR when polynomial fitting is implemented.

Prefer Householder QR.

Provide scalar:

```text
factorQRStack()
factorQRSlice()

solveLeastSquaresQRStack()
solveLeastSquaresQRSlice()
```

QR should be the robust default for general least-squares polynomial fitting.

## SIMD stage

Once scalar QR is verified, investigate SIMD only in arithmetic kernels where it is plausible:

```text
Householder dot products
vector updates
matrix column updates
```

Retain SIMD only if representative polynomial-fit cases show meaningful benefit.

Any retained API is explicitly suffixed:

```zig
solveLeastSquaresQRStackSIMD()
```

Record the result in `dev/simd_perf_log.md`.

---

# 16. Random number and distribution primitives

Create:

```text
random.zig
distributions.zig
```

The RNG must always be passed explicitly.

Never use hidden global random state.

Initial distributions:

```text
uniform
normal
log-normal
exponential
```

Potential later additions should be driven by actual application need.

---

# 17. Array population

Provide direct population of:

```text
[]T
VecSlice
MatSlice
NDArray
```

Operations include:

```text
fillUniform
fillNormal
fillLogNormal
fillExponential
```

Allocation remains outside these routines.

## SIMD stage

Random generation is not automatically a SIMD target.

After the scalar implementation is stable, SIMD may be investigated where:

```text
distribution transform is expensive
large arrays are routinely populated
RNG structure allows safe independent lane generation
```

Reproducibility semantics must be documented.

Do not accept changed random streams casually in exchange for speed.

Record successful and unsuccessful experiments.

---

# 18. Sampling methods

Create:

```text
sampling.zig
```

Initial algorithms:

```text
random permutation
shuffle
sample with replacement
sample without replacement

stratified sampling
Latin hypercube sampling
```

Higher-level uncertainty workflows remain outside Lagomath.

## SIMD stage

Most sampling-control algorithms are unlikely to benefit from SIMD.

Do not vectorise them mechanically.

Investigate only where a clear arithmetic kernel exists.

Record negative results where an attempted SIMD approach proves ineffective.

---

# 19. Statistics

Create:

```text
stats.zig
```

Initial statistics:

```text
sum
mean
variance
standard deviation

minimum
maximum
arg minimum
arg maximum

root mean square

median
quantile
percentile

covariance
correlation

mean absolute error
mean squared error
root mean squared error

empirical CDF primitives
histogram counts
```

## SIMD stage

Reduction-heavy operations are plausible SIMD candidates:

```text
sum
mean
variance
RMS
covariance
error metrics
```

But scalar remains the baseline and default.

Retained alternatives are explicit:

```zig
meanSIMD()
varianceSIMD()
covarianceSIMD()
```

Numerical changes caused by reduction ordering must be measured and documented alongside runtime performance.

---

# 20. Streaming statistics

Eventually provide:

```zig
MeanAccumulator(T)
VarianceAccumulator(T)
CovarianceAccumulator(T)
```

Use stable online algorithms where appropriate.

SIMD should only be investigated if concrete batching use cases justify it.

---

# 21. Numerical differentiation

Create:

```text
differentiate.zig
```

Provide:

```text
forward difference
backward difference
central difference
higher-order stencils where justified
```

Support:

```text
uniform spacing
explicit coordinate arrays
```

Also provide scalar-field gradients in:

```text
1D
2D
3D
```

and component-wise gradients of vector fields.

This enables construction of displacement gradients and deformation gradients.

## SIMD stage

After scalar finite-difference kernels are verified, SIMD should be investigated for large contiguous fields.

Likely good cases:

```text
long 1D signals
large image rows
large regular spatial fields
```

Likely poor cases may include:

```text
tiny fixed vectors
irregular coordinates
edge handling
```

Expose SIMD explicitly:

```zig
differentiateCentralSIMD()
```

where proven useful.

---

# 22. Numerical integration

Create:

```text
integrate.zig
```

Initial methods:

```text
trapezoidal rule
cumulative trapezoidal rule
Simpson rule
```

Support uniform and explicit coordinates.

Later add simple structured multidimensional integration when concrete needs appear.

## SIMD stage

Long contiguous integrations/reductions may benefit from SIMD.

Benchmark realistic sensor histories and field dimensions before retaining SIMD variants.

Be careful about floating-point reduction-order changes.

---

# 23. Smoothing kernels

Create:

```text
kernels.zig
```

Initial kernels:

```text
box
Gaussian
```

Potential later:

```text
triangular
```

Support:

```text
standard deviation
support/radius
normalisation
```

Prefer separable kernels where appropriate.

Kernel construction itself is generally not a high-priority SIMD target.

---

# 24. Filtering and smoothing

Create:

```text
filter.zig
```

Operations:

```text
convolve1D
convolve2D
```

Boundary policies should be explicit:

```text
truncate
clamp
mirror
wrap
```

The same operations should support either spatial or temporal interpretation.

## SIMD stage

Convolution is a strong SIMD candidate for suitable contiguous data.

After scalar correctness, benchmark SIMD for representative:

```text
time-series lengths
image widths/heights
kernel widths
Gaussian radii
```

Do not assume that tiny smoothing kernels always benefit.

Retained versions use:

```zig
convolve1DSIMD()
convolve2DSIMD()
```

---

# 25. Savitzky–Golay filtering

Consider Savitzky–Golay after polynomial fitting exists.

It should reuse:

```text
polynomial basis
least-squares machinery
analytical derivatives
```

Do not implement an unrelated second polynomial engine.

SIMD investigation comes only after the scalar implementation is stable.

---

# 26. Polynomial representation

Create:

```text
polynomial.zig
```

Provide:

```text
1D polynomial bases
2D polynomial surfaces
3D polynomial bases where needed
```

For 2D support terms such as:

```text
1

x
y

x²
xy
y²

x³
x²y
xy²
y³
...
```

Provide deterministic documented term ordering.

Expose:

```text
coefficient count
basis evaluation
basis derivatives
```

---

# 27. Polynomial evaluation

Provide:

```text
evaluatePolynomial1D
evaluatePolynomial2D

evaluatePolynomialDerivative1D

evaluatePolynomialDx2D
evaluatePolynomialDy2D

evaluatePolynomialDxx2D
evaluatePolynomialDxy2D
evaluatePolynomialDyy2D
```

Derivatives of fitted surfaces should be analytical.

## SIMD stage

Polynomial evaluation over many independent coordinates is a plausible outer-SIMD use case.

Because callers may choose to SIMD across points themselves, scalar must remain the default.

Any internal SIMD version is explicitly opt-in and benchmarked against that usage pattern.

---

# 28. Multidimensional polynomial fitting

Create:

```text
polyfit.zig
```

Support fitting:

```text
(xᵢ, yᵢ, fᵢ)
        ↓
polynomial coefficients
```

with:

```text
arbitrary sample coordinates
chosen degree
overdetermined systems
weighted least squares
```

QR is the robust default solver.

Coordinate centring/scaling should be supported to improve conditioning.

---

# 29. Polynomial fit result

Provide a fitted model containing:

```text
coefficients
coordinate origin
coordinate scale
```

and methods for:

```text
value
dx
dy
dxx
dxy
dyy
```

without refitting.

Fixed-degree versions should take advantage of compile-time coefficient counts.

Runtime-degree Slice-backed forms may be provided when needed.

## SIMD stage

SIMD should be investigated separately for:

```text
basis construction
least-squares solve
evaluation at many points
derivative evaluation
```

Do not treat “polyfit SIMD” as one monolithic optimisation.

Record each useful experiment separately.

---

# 30. Tensor operations

Create:

```text
tensorops.zig
```

Provide:

```text
trace
determinant
inverse

transpose
symmetric part
skew part

double contraction

deviatoric part
hydrostatic part

principal invariants

basis transformation
rotation
```

Use matrix storage rather than introducing a large tensor hierarchy prematurely.

## SIMD stage

Small 3×3 mechanics tensors are likely cases where explicit SIMD may offer little benefit.

That should be tested rather than assumed.

If SIMD performs poorly, record the result so the same idea is not repeatedly revisited.

---

# 31. Mechanics primitives

Create:

```text
mechanics.zig
```

Initial useful operations:

```text
deformation gradient
small-strain tensor
Green–Lagrange strain
Cauchy–Green tensors
principal invariants
von Mises equivalent quantities
```

Do not make this a constitutive-model library.

---

# 32. Voigt notation

Provide explicit stress and strain conversions.

For example:

```zig
stressToVoigt()
voigtToStress()

strainToVoigtEngineering()
voigtEngineeringToStrain()

strainToVoigtTensor()
voigtTensorToStrain()
```

Never hide the engineering-shear factor-of-two convention.

SIMD is unlikely to be necessary for isolated 3D Voigt transformations, but benchmark if they become a demonstrated hot path.

---

# 33. Geometry utilities

Create:

```text
geometry.zig
```

for generic primitives such as:

```text
distance
squared distance
normalise vector
cross product
projection
orthogonal projection
coordinate rotation
basis transformation
```

Application-specific mesh/raster geometry remains outside Lagomath.

---

# 34. Array operations

Create:

```text
arrayops.zig
```

for:

```text
fill
zero
copy

add
subtract
multiply
divide

square
sqrt
abs
exp
log

clamp

min
max
sum

allFinite
```

NDArray should reuse flat contiguous kernels where appropriate.

## SIMD stage

Large contiguous array operations are obvious SIMD candidates.

Benchmark realistic field and image sizes.

Explicit alternatives use names such as:

```zig
addSIMD()
sqrtSIMD()
```

Scalar remains available even if SIMD wins overwhelmingly.

---

# 35. Public API philosophy

The top-level API should remain shallow:

```zig
const lm = @import("lagomath");

lm.vecops
lm.matops
lm.tensorops

lm.linsolve

lm.random
lm.sampling
lm.stats

lm.differentiate
lm.integrate

lm.kernels
lm.filter

lm.polynomial
lm.polyfit

lm.interp
lm.mechanics
```

Within those modules:

```text
foo()
```

means scalar.

```text
fooSIMD()
```

means explicit SIMD.

Do not use:

```text
fooScalar()
```

for the normal implementation unless disambiguation is genuinely necessary.

Scalar is the baseline; it does not need a suffix.

---

# 36. Benchmark structure

All performance experiments live under:

```text
src/bench/
```

using:

```text
bench_CASE.zig
```

Examples:

```text
bench_cholesky.zig
bench_outer.zig
bench_stats.zig
bench_filter.zig
bench_polyfit.zig
```

Each benchmark contains controllable constants at file scope:

```zig
const NUM_WARMUP: usize = 10_000;
const NUM_SAMPLES: usize = 20;
const NUM_RUNS: usize = 100_000;

const N: usize = 6;
const T = f64;
```

Do not hide important benchmark configuration inside `main`.

---

# 37. Benchmark timing rules

Do not time:

```text
allocation
input generation
random seeding
file I/O
console output
setup of matrices
creation of kernels
```

unless that operation itself is explicitly the subject of the benchmark.

For mathematical-kernel benchmarks, all memory and inputs are prepared before timing.

Use enough repeated operations that timer granularity is insignificant.

Use multiple samples to establish run-to-run variation.

Report at least:

```text
minimum
mean
median where practical
maximum
```

alongside:

```text
scalar/SIMD ratio
speedup
```

---

# 38. Meaningful performance benefit

Do not define one arbitrary universal percentage threshold.

A performance improvement must be:

```text
larger than observed benchmark noise
repeatable
present in representative workloads
worth the implementation complexity
```

For example, a 1–2% improvement in a noisy benchmark is not evidence that SIMD should be retained.

A small improvement may still matter for an extremely dominant kernel, while even a larger improvement may be irrelevant for rarely executed setup code.

Use engineering judgement and record the rationale.

---

# 39. Domain boundary examples

## Sensor simulation

Lagomath provides:

```text
random sampling
covariance
interpolation
integration
filtering
statistics
```

Felix/Pyvale decide how these form a sensor model.

## DIC

Lagomath provides:

```text
matrix solve
outer products
polynomial fit
derivatives
smoothing
```

The DIC engine owns:

```text
subset matching
warps
objective functions
Gauss–Newton/LM workflow
```

## Experimental mechanics

Lagomath provides:

```text
2D polynomial fitting
analytical derivatives
deformation gradients
strain tensors
filtering
statistics
```

The experiment code decides:

```text
point neighbourhoods
window size
smoothing policy
physical interpretation
```

## Validation

Lagomath provides:

```text
statistics
sampling
covariance
quantiles
numerical integration
basic norms/errors
```

Validation methodology remains outside Lagomath.

---

# 40. Feature development lifecycle

Every substantial numerical feature should follow the same lifecycle.

## Stage 1 — Scalar design

Define:

```text
storage representation
ownership
allocation behaviour
Stack/Slice API
failure semantics
numerical assumptions
```

Implement the simplest plausible performant scalar architecture.

## Stage 2 — Correctness

Verify with:

```text
analytic solutions
known reference values
mathematical invariants
independent oracle calculations
cross-implementation checks
```

Add local tests and regression tests.

## Stage 3 — Scalar benchmark

Create:

```text
src/bench/bench_CASE.zig
```

where the operation is performance-relevant.

Record representative scalar behaviour.

## Stage 4 — SIMD investigation

Only now attempt SIMD.

Use `@Vector` or other explicit vector mechanisms only where data layout and computation are suitable.

## Stage 5 — SIMD correctness

Verify SIMD against:

```text
scalar result
independent expected result
```

Document tolerances where floating-point operation ordering differs.

## Stage 6 — SIMD benchmark

A/B test scalar and SIMD using representative cases and enough samples/runs to measure reliably.

Do not benchmark allocation.

## Stage 7 — Decision

Choose:

```text
retain SIMD
reject SIMD
retain SIMD only for particular dimensions/workloads
```

The ordinary scalar implementation always remains available.

## Stage 8 — Record

Add the result to:

```text
dev/simd_perf_log.md
```

including negative results.

Only then is the feature considered performance-reviewed.

---

# 41. Recommended development order

## Phase A — linear algebra

Implement:

```text
vecops
matops
rowops

Gaussian solve
LU
Cholesky
```

Then perform the SIMD lifecycle for worthwhile candidates.

---

## Phase B — random and statistics

Extract/build:

```text
random
distributions
sampling
stats
```

Then separately investigate SIMD for high-volume kernels.

---

## Phase C — polynomial and least-squares infrastructure

Implement:

```text
QR
polynomial basis
1D polynomial evaluation
2D polynomial basis
least-squares polynomial fit
weighted polynomial fit
analytical derivatives
```

Then investigate SIMD component-by-component.

---

## Phase D — differentiation and integration

Implement scalar:

```text
finite differences
spatial gradients
trapz
cumulative trapz
Simpson
```

Verify first.

Then investigate SIMD for large regular datasets.

---

## Phase E — smoothing and filtering

Implement:

```text
kernel generation
1D convolution
2D convolution
Gaussian smoothing
boundary policies
```

Then benchmark SIMD.

Later consider Savitzky–Golay using the common polynomial infrastructure.

---

## Phase F — mechanics

Implement:

```text
tensor operations
Voigt conversion
deformation gradients
strain measures
basic invariants
```

Then test SIMD only where actual workloads justify investigation.

---

# 42. Verification philosophy

Every public numerical operation should have independent correctness evidence.

Prefer:

```text
analytic results
known matrices
known polynomials
known derivatives
known integrals
independent external reference values
mathematical invariants
```

Examples:

```text
differentiate x³ analytically

integrate x² over a known interval

fit a known 2D polynomial and recover its coefficients

differentiate a fitted polynomial analytically

smooth a constant field and preserve the constant

verify Gaussian-kernel normalisation

verify tensor rotation invariants

verify F = I for zero displacement gradient
```

Optimised implementations must additionally be checked against scalar references.

---

# 43. What Lagomath should deliberately not become

Do not add domain workflows such as:

```text
complete DIC
sensor models
constitutive material models
finite elements
mesh handling
rasterisation
validation methodologies
Bayesian inference frameworks
optimisation frameworks
plotting
file formats
```

Do not add functionality merely because another numerical library has it.

The test remains:

> Is this a reusable numerical primitive or small algorithm that multiple engineering applications can reasonably build upon?

If yes, it belongs in Lagomath.

If it encodes a scientific workflow or domain decision, it probably belongs above Lagomath.

---

# 44. Long-term conceptual architecture

```text
                         LAGOMATH
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
     storage              algebra              numerics
        │                    │                    │
   NDArray              vecops/matops       differentiation
   MatStack             tensorops           integration
   MatSlice             linsolve            filtering
   VecStack             QR                  polynomial fit
   VecSlice                                 interpolation
        │                    │                    │
        └─────────────┬──────┴──────┬─────────────┘
                      │             │
                  statistics      mechanics
                      │             │
                random/sampling  strain/tensors
                      │             │
                      └──────┬──────┘
                             │
               reusable engineering substrate
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
        Riley              Felix             Pyvale
```

Across the entire architecture:

```text
scalar API
    = default, composable implementation

SIMD API
    = explicit opt-in alternative
    = retained only after benchmark evidence
    = documented in dev/simd_perf_log.md
```

The goal is not that every project uses every Lagomath module.

The goal is that when one of these projects needs a general numerical building block, it should not need to reinvent it — and when it needs control over how parallelism is composed, Lagomath should not take that control away.
