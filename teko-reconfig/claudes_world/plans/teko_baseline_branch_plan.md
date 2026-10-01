# Plan: a stock-Trilinos baseline branch

**Decisions, 2026-10-01.** Name `aidantgould/teko-baseline`, matching the
existing work branch's prefix rather than the `atgould/` first typed. Base is
`upstream/develop`'s tip. The fork's own stale `develop` and `master` are left
alone, so the fast-forward commands below are recorded but NOT run.

A branch of the fork carrying upstream Trilinos and nothing of ours, to build
the app against as a control. Motivated 2026-10-01: the app hangs on its first
Teko solve, and `strings` showed its binary contains none of this branch's 16
`[TekoAdaptive]` literals, so the hang is between the app and stock Teko. A
baseline makes that testable instead of assumed.

## Which base, and why it barely matters

Measured today, not assumed:

- `upstream/develop` is at `ff79a76fed6` (2026-10-01).
- This work branch sits on `fbcdcf73530` (2026-09-24), 70 upstream commits back.
- Of those 70, exactly **2** touch `packages/teko` or `packages/belos`, both in
  Belos (an `override` annotation in the PCPG solver manager, and a build fix in
  experimental code). **None touch Teko at all.**

So a baseline at today's tip and a baseline at this branch's own base have
identical Teko and near-identical Belos. Taking the tip costs nothing in
comparability and gives a baseline worth keeping.

## What gets done

1. `git fetch upstream` (done).
2. `git branch aidantgould/teko-baseline upstream/develop` — created WITHOUT
   checking it out. The current checkout must not move: `trilinos-build/` is
   compiled against it, and switching the working tree would silently
   invalidate the built `libteko.so` the pyfront extension links to.
3. `git push -u origin aidantgould/teko-baseline`.

Nothing is merged, nothing is rebased, and the branch contains no `teko-reconfig/`
directory, no `claudes_world/`, and none of the hook: it is upstream, exactly.

## Update master instead, then branch from it?

No. Trilinos develops on `develop`; `master` is the release branch, and today it
lags develop by five days (`884e832a427`, 2026-09-26). Branching from `master`
would give a baseline that differs from this work branch by both our changes and
a release-boundary gap, which is a worse control. Separately, the fork's own
`develop` and `master` are stale at 2026-04, so branching from either without
updating it first would give a six-month-old baseline, which is the opposite of
what is wanted.

Syncing the fork's `develop` and `master` to upstream would be independent
hygiene, not a step toward the baseline, and was declined for now. Both are pure
fast-forwards whenever it is wanted:

    git push origin upstream/develop:develop
    git push origin upstream/master:master

## Verification

- `git log aidantgould/teko-baseline ^upstream/develop` is empty (no extra commits).
- `git diff aidantgould/teko-baseline upstream/develop -- packages/teko` is empty.
- `git log --oneline aidantgould/teko-baseline -- teko-reconfig` is empty (none of
  our tooling rode along).
- `git rev-list --count aidantgould/teko-baseline..aidantgould/teko-reconfig-request`
  reports the 28 commits that are ours alone.
- `git branch --show-current` still reports the work branch, and
  `git status --porcelain` is unchanged, i.e. the checkout never moved.

## Building it without disturbing this checkout

Not part of this plan, but the reason step 2 avoids a checkout. When the
baseline needs compiling, use a second working tree rather than switching
branches in place:

    git worktree add ../Trilinos-baseline aidantgould/teko-baseline

That gives a separate source directory at the baseline commit, leaving this
checkout and `trilinos-build/` untouched, and needs its own configure and build
directory.

## Non-goals

- No build of the baseline, and no change to `trilinos-build/`.
- No merge, rebase, or cherry-pick between the branches.
- No change to the work branch, and no checkout switch.
- No attempt to fix the app's hang; this only makes the control available.
