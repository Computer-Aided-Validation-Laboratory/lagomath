# Lagomath — Agent Instructions

Lagomath is a lightweight, performance-oriented numerical mathematics library for vectors, matrices and multidimensional arrays. Its Zig implementation is the current codebase; a corresponding Mojo implementation and C ABI interoperability are planned. Prioritise **numerical correctness, performance, explicit memory semantics and simplicity for users** together.

## 1. Development workflow — mandatory stage gates

**Default to Stage 1. Do not advance stages without explicit user instruction.** A passed test suite, completed implementation or agreed plan does not itself authorise propagation to another language or interface.

### Stage 1 — Zig implementation (default)

- Develop and refactor the Zig mathematical types, scalar operations, algorithms and focused Zig tests.
- Use the current Zig code as the starting point; follow approved plans for API renames and structural changes.
- Keep the implementation scalar by default. Do not add SIMD variants or benchmark-driven optimisations unless explicitly requested.
- Do not propagate changes into Mojo, C ABI, Python integration or unrelated consumers.
- Run focused tests first; use the existing quiet runner rather than verbose output.

```bash
zig fmt --check .
zig build test -Doptimize=Debug
```

### Stage 2 — Zig consolidation and verification

- Enter only when the user requests consolidation, comprehensive verification or a release-readiness check.
- Complete approved refactoring and exercise the full relevant numerical test cases, including dimensions, layout, views, mutation, ownership, errors and edge cases within the supported contract.
- Run Debug and ReleaseFast tests, the example and package-boundary consumer check.
- Investigate discrepancies; do not silently adjust reference answers or loosen tolerances.
- Remain within Zig. Do not implement Mojo, C ABI or Python just because Zig checks pass.

```bash
zig fmt --check .
zig build test -Doptimize=Debug
zig build test -Doptimize=ReleaseFast
zig build run-example
(cd integration/consumer && zig build run)
```

### Stage 3 — Mojo implementation and cross-language parity

- Enter **only** after explicit approval of the Zig API/behaviour and instruction to implement or update Mojo.
- Use the approved shared mathematical definitions, semantics and test cases, not ad hoc translations of implementation details.
- Maintain matching public type names and equivalent numerical semantics where the languages permit, while retaining idiomatic Zig and Mojo ownership/lifetime models.
- Run the relevant Mojo tests and the shared Zig/Mojo numerical cases. Discover actual Mojo commands in the repository; do not invent targets.
- Do not add the C ABI or Python tests without a separate explicit instruction.

### Stage 4 — C ABI and interoperability

- Enter **only** when explicitly directed to expose an approved feature through a C ABI.
- Export minimal, stable, C-compatible representations and functions. Do not expose compiler-specific Zig or Mojo type layouts as a public ABI.
- Make allocation, deallocation, ownership, dimensionality, scalar types, status codes and error handling explicit across the boundary.
- Add ABI tests, including Python `pytest`/`cffi` tests against the **built** Zig or Mojo library when requested by the feature plan. Test exported behaviour and failure paths, not just symbol presence.
- Keep bindings restricted to approved functionality; do not expand the native API as part of binding work.

### Optional SIMD evaluation — separate approval gate

- **Scalar implementations are the baseline and must remain available.** SIMD is opt-in through explicitly named functions with a `SIMD` suffix (for example, `solveSIMD`); never silently replace or force scalar kernels to use an explicit SIMD implementation.
- Add SIMD only when authorised for a particular operation and when benchmark evidence shows a meaningful improvement for representative use cases.
- Benchmark the scalar and SIMD versions with the same inputs, compiler modes, hardware and numerical accuracy criteria; include cases where outer-loop SIMD by the caller may be preferable.
- Record **both successful and unsuccessful** evaluations in `dev/simd_perf_log.md`: operation, benchmark location and reproduction command, inputs and sizes, environment, results, correctness checks and decision.
- Do not keep a new SIMD implementation solely because it looks faster at source level.

## 2. Documentation and codebase exploration

- **Existing repository:** `README.md` documents the current Zig public API and build/test entry points. Inspect relevant material in `dev/` before relying on additional architecture or testing conventions.
- **Organisation guides:** Follow the [CAVL Performance Oriented Software Guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/performance_oriented_software_guide.md) and [CAVL Zig Style Guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/zig_style_guide.md). For substantive design changes read the relevant guide sections, performance guide first.
- **Focused inspection:** Start with user-named modules, test cases and examples; otherwise inspect the narrowest relevant implementation. Broaden only for a concrete dependency.
- **Minimise context:** Use targeted searches, avoid scanning the full repository without need and do not reread unchanged code or documents already in context.
- **Plan versus implementation:** Do not confuse intended APIs, files or commands with those already present. Verify paths and commands before using them.

## 3. Lagomath-specific engineering rules

1. **Numerical library contracts matter.** Document and test the supported scalar types, shapes, layout, indexing, aliasing and floating-point behaviour. Do not silently change those contracts.
2. **Consistent mathematical types.** The approved target names across Zig and Mojo are `Vec`, `Mat`, `VecView`, `MatView`, `NDArray` and `NDView`. The repository currently uses older names such as `VecStack`, `MatStack`, `VecSlice` and `MatSlice`; rename only within an authorised task. Do not add compatibility shims unless asked.
3. **Distinct ownership semantics.** `Vec` and `Mat` are fixed-size inline values; `VecView` and `MatView` are borrowed views; `NDArray` owns multidimensional storage; `NDView` borrows multidimensional storage. Ensure APIs, copying, mutation, deinitialisation and lifetimes reflect that distinction.
4. **Explicit memory layout.** Establish row/column ordering, strides, contiguity and indexing conventions before implementing kernels. Support strided access only where the declared view contract requires it; avoid hidden conversions and copies.
5. **Allocation is explicit.** Make allocating operations and allocator responsibilities obvious. Keep fixed-size operations and ordinary borrowed-view kernels allocation-free; never return dangling views or scratch-backed storage.
6. **Separate external validation from inner kernels.** Validate user-derived dimensions, buffers and indices at appropriate boundaries. Use assertions for established programmer invariants in lean kernels; provide checked entry points where justified. Do not repeatedly validate every element in a hot loop.
7. **Handle numerical edge cases deliberately.** Define behaviour for zero dimensions (if supported), singular and ill-conditioned systems, division by zero, overflows, NaNs and infinities within the operation's contract. Do not bury genuine errors behind fabricated defaults or `catch unreachable`.
8. **Keep APIs cohesive and predictable.** Prefer mathematical terminology, explicit shape information, clear mutation/return semantics, descriptive names and consistent argument ordering. Avoid speculative abstractions and overlapping aliases.
9. **Specialise selectively.** Compile-time shape/type specialisation is useful for small fixed-size kernels but may increase code size and compilation cost. Use runtime dimensions for dynamically shaped arrays; choose based on actual use cases.
10. **Performance is measured, not assumed.** Consider data locality, copies, branches, loop order, temporaries and call overhead. Establish scalar baselines before optimisation; compare representative small and large vectors, matrices and multidimensional shapes.
11. **Keep numeric comparisons meaningful.** Use exact equality where mathematically warranted, and justified absolute/relative tolerances for floating-point results. Do not hide discrepancies behind excessively loose tolerances.
12. **Allow clean API evolution.** Lagomath is actively developed. Prefer a coherent future API over deprecated names, wrapper layers and backwards-compatibility scaffolding, unless compatibility is an explicit requirement.
13. **Maintain language parity without artificial uniformity.** Share type names, operations, case data and expected outcomes across Zig and Mojo; implementation techniques may differ to respect each language's memory and performance model.
14. **Report evidence.** Distinguish tested behaviour from expectations. State the commands run, failures and untested paths, especially for benchmark and cross-language claims.

## 4. Shared numerical testing and performance cases

- Follow the approved plan for common JSON test fixtures and performance-case definitions when they exist. **Do not invent or move fixture files without instruction.**
- Use one language-neutral definition of cases and expected mathematical behaviour for Zig and Mojo. JSON is the interchange format, not the preferred storage format for internal hot-loop data.
- Include fixed-size and dynamically sized cases, vector/matrix/ND dimensions, non-square shapes, very small and representative larger workloads, views, aliasing, mutability, boundaries and supported errors.
- For algorithms, compare to independent mathematical references or analytically tractable cases where practical; ensure reference values and tolerances are explicit.
- Separate correctness fixtures from performance cases. Do not time JSON parsing, setup, allocations or fixture conversion unless these are intentionally part of the measured operation.
- Run repeated benchmarks with adequate warm-up and a clearly stated methodology. Record both absolute timings and relative comparisons; avoid claiming speedups based on a single noisy run.
- Do not rewrite approved reference outputs or tolerances to make failing tests pass without explicit approval.

## 5. Agent execution efficiency

- Follow the supplied specification and approved architectural decisions; reopen settled choices only when a concrete defect or conflict requires it.
- Work in coherent batches: **build → refactor → add → refactor → verify**. Consolidate documentation only at the end and only on explicit instruction.
- Use deterministic scripts for substantial mechanical renames, followed by targeted inspection.
- Keep routine test output quiet. Redirect verbose compiler, test and benchmark output to local logs; summarise relevant failures instead of dumping full output.
- Prefer targeted tests during development; reserve full configurations and consumer integration for the authorised stage or when specifically requested.
- Avoid broad search, redundant recompilation, repeated checks, unrelated cleanup and unsolicited API expansion.
- **Do not update documentation, unrelated examples or docstrings unless explicitly requested.** `AGENTS.md`, architecture and performance logs are not permission to perform wider documentation edits.
- Communicate concisely: changes made, commands/tests run, measured results and outstanding issues. Stop when the requested scope is complete.

## 6. Zig style summary

These rules summarise the [CAVL Zig Style Guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/zig_style_guide.md). Apply them to Zig code; consult the full guide when needed.

### Types and interfaces

1. Prefer enums, optionals and tagged unions to stringly typed states, redundant flags and loosely coupled tags.
2. Use `union(enum)` for closed runtime variants, `?T` for optional values and compile-time dispatch when the implementation is known statically.
3. Express borrowed/owned and mutable/constant data clearly using `[]T`, `[]const T`, pointers and dedicated value/view types.
4. Avoid unnecessary runtime interfaces, indirect dispatch and general-purpose generic machinery.

### Allocation and errors

5. Pass allocators explicitly to allocating operations; do not silently hide allocation in ordinary mathematical kernels.
6. Distinguish returned allocations (`outer_alloc`) from temporary storage (`local_alloc`) when both are needed; never return local scratch memory.
7. Use `defer`/`errdefer` close to resource acquisition; make transfers of ownership unambiguous.
8. Use errors for legitimate runtime failures, `std.debug.assert` for invariants and `unreachable` only for genuinely impossible paths.
9. Do not silence errors with fabricated defaults, broad catches or `catch unreachable`.

### Performance and organisation

10. Keep inner loops simple, allocation-free where feasible, and conscious of memory order and aliasing.
11. Use `comptime` where it genuinely improves interface or code generation; avoid gratuitous specialisation and `inline`.
12. Use explicit opt-in SIMD kernels only under the performance evaluation gate above.
13. Prefer direct imports and clearly owned declarations; avoid accidental public re-export chains.
14. Use descriptive names: functions are verbs and types are nouns. Keep expressions readable and avoid unnecessary casts.
15. Run `zig fmt` on edited Zig files and add deterministic tests for meaningful behaviours and errors.

## 7. Current repository commands and scope

The commands below are **verified from the current repository README/build script**; future Mojo, benchmark and ABI commands must be checked once implemented.

```bash
zig build test                      # Quiet unit suite
zig build test -Doptimize=ReleaseFast
zig build test-verbose              # Only when per-test output is needed
zig build run-example
(cd integration/consumer && zig build run)
(cd integration/consumer && zig build run -Doptimize=ReleaseFast)
zig fmt --check .
```

The current repository is a Zig source-only library targeting Zig 0.16.x, with `src/`, `examples/basic/`, `integration/consumer/`, `dev/` and `scripts/`. Do not assume a separately built native library, functioning Mojo package, C ABI target or Python test suite already exists.
