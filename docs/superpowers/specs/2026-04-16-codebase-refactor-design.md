# SQuADDS Codebase Refactor Design

**Goal**

Create a pure refactor roadmap for SQuADDS that improves modularity, testability, readability, and maintainability without changing public behavior, dataset semantics, or user-facing APIs.

**Non-Goals**

- No schema changes to `SQuADDS/SQuADDS_DB`
- No behavioral changes to search, interpolation, simulation, or contribution flows
- No forced migration of existing user code
- No tutorial/notebook rewrites as part of the first tracked refactor wave

## Context

The current codebase is useful and feature-rich, but it has several structural pressures:

- large modules with mixed responsibilities (`squadds/core/db.py`, `squadds/core/analysis.py`, `squadds/simulations/objects.py`)
- integration-style tests that currently behave more like scripts than a maintainable test suite
- duplicated concepts across database access, search, simulation, and contribution flows
- heavy use of broad imports and implicit state
- weak boundaries between pure data transforms, I/O, plotting, and simulator orchestration

The right cleanup strategy is not a rewrite. It is a characterization-tests-first, module-by-module extraction that preserves the current public entrypoints while making internals easier to reason about.

## Approaches Considered

### Approach 1: Big-Bang Rewrite

Replace the current package structure with a new architecture in one pass.

Why not:

- too risky for a research-heavy package with simulator and dataset integration
- difficult to prove behavioral compatibility
- invites accidental API and schema drift

### Approach 2: Thin Compatibility Facades Over New Internal Modules

Keep existing public modules and symbols, but progressively extract smaller internal modules and route the old entrypoints through them.

Why this is recommended:

- preserves compatibility while improving structure
- allows characterization tests around existing behavior before moving code
- supports medium-sized, meaningful commits with verification after each pass
- makes it easy to stop after any phase with a cleaner codebase than before

### Approach 3: Test-Only Cleanup First, Architecture Later

Focus only on test harness improvements and defer real module cleanup.

Why not as the full strategy:

- useful as a first phase, but insufficient on its own
- does not address the monolithic module boundaries that slow future feature work

## Recommended Design

Use **Approach 2**, staged through six refactor phases:

1. **Foundation and Characterization**
   Turn script-like tests into real pytest tests, add integration gates, and remove obvious repo hygiene friction so future refactors have a reliable baseline.

2. **Database Layer Decomposition**
   Split dataset catalog discovery, selection state, loading/flattening, and multi-component dataframe assembly into focused internal modules while keeping `SQuADDS_DB` as the compatibility facade.

3. **Analysis Layer Decomposition**
   Split Hamiltonian-parameter enrichment, metric dispatch, closest-design search, result extraction, and plotting into separate modules while keeping `Analyzer` stable.

4. **Simulation Layer Decomposition**
   Separate sweep expansion, geometry construction, simulator execution, result normalization, and checkpoint-friendly orchestration from the current mixed simulation modules.

5. **Contribution Workflow Cleanup**
   Isolate Hugging Face dataset interactions, GitHub PR creation, contributor metadata handling, and schema validation into clearer service-style modules.

6. **UI and Docs Alignment**
   Keep the Streamlit UI thin by moving query/search orchestration behind testable helpers, and align docs/tests/tutorials with the refactored internals.

## Compatibility Rules

Every tracked refactor phase must obey these rules:

- preserve exported names from `squadds/__init__.py`
- preserve existing dataset config names and on-disk JSON structure
- preserve current default search and interpolation behavior
- preserve existing simulation object names and their public call surfaces unless only moved internally
- keep deprecations explicit instead of silently removing behavior

## Wishlist Tie-Ins

These wishlist items can be partially or fully advanced during a pure refactor, without changing functionality:

- **Addressing TODOs in the code**
  Refactor work can isolate TODO-heavy areas into smaller modules where follow-up fixes are safer.

- **Robustly handling caching and environment variables setup for all OS**
  Refactoring env/config access into one place is a pure-structure improvement.

- **Check front-end UI code thoroughly against API calls**
  A thinner UI adapter layer makes that auditing easier.

- **Improve system design for both the SQuADDS package and SQuADDS_DB**
  This is the core purpose of the refactor.

- **Refactor code to implement faster methods with lower memory usage for handling DataFrame operations**
  The first step is isolating pure transforms and merge/search logic.

- **Implement an automated build check with comprehensive unit tests**
  Converting script-like tests into reliable pytest coverage is part of Phase 1.

- **Create unit tests for each feature/file**
  The refactor should move the project toward file-scoped characterization tests.

- **Letting users choose the `.env` file location OR telling them where to find it**
  Centralizing env/config resolution is a pure refactor target.

- **Standardize the handling of units for simulated results and implement necessary backend changes**
  The standardization scaffolding can begin with normalization helpers and compatibility adapters.

## Codex Recs

- Formalize a small internal architecture map under `squadds/` with focused modules for catalog, loading, normalization, plotting, and orchestration.
- Introduce characterization tests before every large extraction so refactors are evidence-backed instead of intuition-driven.
- Move all schema and payload normalization into dedicated helpers; do not let simulator, DB, and contribution layers each invent their own record shapes.
- Add a lightweight compatibility policy for public symbols so internal cleanup can move quickly without surprising downstream users.
- Treat checkpoint/resume as a first-class pattern for long-running simulation workflows, even when the initial implementation lives in scripts.
- Keep notebooks and tutorial scripts as consumers of the package, not the place where package architecture lives.
