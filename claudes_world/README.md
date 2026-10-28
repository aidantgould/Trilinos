# claudes_world

Working notes for the `aidantgould/teko-auto*` branches: plans, scratch studies and
parity checks that support the Teko Krylov surrogate work but are not part of Trilinos.
Nothing here is built or installed, and the whole directory is stripped before any
upstream PR.

- `plans/`: design and execution plans, newest work first. A plan still here may be
  unfinished. `teko_auto_mvp_plan.md` is the current one.
- `teko_auto_parity/` (when it exists): comparisons against the research hack
  (`aidantgould/teko-reconfig-request`) and pyautoteko. Code is tracked, run data is not.

The shipped code lives in `packages/teko/src/Teko_KrylovSurrogate*` and its tests in
`packages/teko/tests/unit_tests/tKrylovSurrogate*.cpp`.
