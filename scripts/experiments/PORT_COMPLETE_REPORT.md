# Port-complete QMetal GDS: representation and transfer study

Generated from the artifacts at `SQuADDS-port-gds-artifacts` using the scripts in
`SQuADDS-port-gds/scripts/experiments/`. Nothing was uploaded anywhere.

## 1. Dataset, splits, and provenance

| item | value |
| --- | --- |
| layout source | port-complete QMetal regeneration, `codex/transmon-port-gds` @ `6b2788a` |
| simulation source | `SQuADDS/SQuADDS_DB` @ `0e25705f54c343fb96571ff15b6fd8375ca899aa` |
| published embeddings | `SQuADDS/SQuADDS_Layout_Embeddings` @ `6cd6295` (resolved at run time) |
| rows | 2,828 = 1,934 `TransmonCross` + 894 `CapNInterdigitalTee` |
| primary targets | `cross_to_claw` (fF) and `top_to_bottom` (fF) |
| split policy | repeated grouped holdout, groups = `design_id`, 30% test, 12 repeats |
| seed | 24 (split seed `24 + 101 * repeat`), projector seed 24 |
| model | random-Fourier features (128) on the compact embedding, multi-output ridge, alpha 0.3 |
| metrics | RMSE (fF), median absolute error (fF), MAE (fF), R² |

**Leakage control.** Splits are grouped on `design_id`, so the one duplicated
TransmonCross design (`transmon_cross_0007` and `transmon_cross_0621`, identical
`design_id` *and* `layout_id`) can never straddle train and test. 2,827 unique
identities across 2,828 rows.

**Integrity.** All 2,828 manifest rows resolve to a GDS file. A full-sweep check
(`terminal_ordering.json`) confirms **2,828/2,828 expose exactly two terminals and
two port markers, with zero failures**.

**Join keys.** `layout_id` shares **zero** rows with the published manifest for
both families, because adding port layers changes the content hash. All old-versus-new
comparisons therefore join on `design_id` (1,933/1,934 Transmon, 894/894 CapN).

## 2. Representations

| name | what it is |
| --- | --- |
| `old/v0`, `old/v2` | the published tables, built from the portless GDS |
| `new/v0` | `static-shape-v0` **as published**, rebuilt on the new GDS. Its role map ignores ports for these families, so this isolates geometry |
| `new/v0-etch` | **local variant**: conductor + etch, port-blind |
| `new/v0-ports` | **local variant**: conductor + etch + ordered ports |
| `new/v1-local` | **local** v1 fit on this two-family cohort. Not comparable to published v1, which is Generalized-only and fit on write |
| `new/v2` | `universal-geometry-v2` as published; the only shipped standard whose role map already consumes ports for every family |

## 3. What actually changed in the geometry

| family | conductor area (new/old) | perimeter (new/old) | v0 bitmap |
| --- | --- | --- | --- |
| TransmonCross | **1.0000 exactly** | **1.0000 exactly** | identical |
| CapNInterdigitalTee | ×2.12 median (1.38–4.65) | ×1.74 median | all differ |

Terminal ordering under v2, port-based versus the old area fallback, across the
whole sweep:

| family | rows | ordering differs |
| --- | ---: | ---: |
| TransmonCross | 1,934 | **0 (0.0%)** |
| CapNInterdigitalTee | 894 | **87 (9.7%)** |

## 4. Within-family results, full training pool

**TransmonCross — `cross_to_claw`** (1,354 train / 580 test per repeat)

| representation | RMSE fF | med abs fF | MAE fF | R² |
| --- | ---: | ---: | ---: | ---: |
| old/v0 | 0.3262 | 0.1644 | 0.2307 | 0.9831 |
| new/v0 | 0.3920 | 0.1970 | 0.2732 | 0.9757 |
| new/v0-etch | 0.2766 | 0.1519 | 0.2019 | 0.9879 |
| new/v0-ports | 0.2656 | 0.1462 | 0.1932 | 0.9888 |
| new/v1-local | 0.1561 | 0.0783 | 0.1067 | 0.9961 |
| old/v2 | 0.0409 | 0.0234 | 0.0300 | 0.9997 |
| **new/v2** | **0.0408** | **0.0234** | **0.0299** | **0.9997** |

**CapNInterdigitalTee — `top_to_bottom`** (626 train / 268 test per repeat)

| representation | RMSE fF | med abs fF | MAE fF | R² |
| --- | ---: | ---: | ---: | ---: |
| old/v0 | 2.1655 | 0.9554 | 1.3584 | 0.9597 |
| new/v0 | 2.0081 | 0.9062 | 1.3198 | 0.9662 |
| new/v0-etch | 1.9307 | 0.8426 | 1.2574 | 0.9688 |
| new/v0-ports | 1.6270 | 0.7733 | 1.0695 | 0.9778 |
| new/v1-local | 1.8200 | 0.8196 | 1.1872 | 0.9713 |
| old/v2 | 0.7423 | 0.3172 | 0.4627 | 0.9953 |
| **new/v2** | **0.7081** | **0.2349** | **0.3887** | **0.9956** |

Standard deviations across the 12 repeats are in `within_family.csv`; they are
roughly 0.002 fF (Transmon v2) to 0.4 fF (CapN old/v0).

## 5. Effect isolation

| effect | contrast | RMSE fF | change |
| --- | --- | --- | ---: |
| TransmonCross ordered ports, under v2 | old/v2 → new/v2 | 0.04091 → 0.04083 | **−0.2%** |
| TransmonCross etch, under v0 | new/v0 → new/v0-etch | 0.3920 → 0.2766 | −29.5% |
| TransmonCross ordered ports, under v0 | new/v0-etch → new/v0-ports | 0.2766 → 0.2656 | −4.0% |
| CapN geometry + ports, under v2 | old/v2 → new/v2 | 0.7423 → 0.7081 | −4.6% |
| CapN ordered ports, under v0 | new/v0-etch → new/v0-ports | 1.9307 → 1.6270 | −15.7% |

**The TransmonCross port result is a null, and it was predictable.** Its conductor
geometry is byte-identical between releases and its port ordering never disagrees
with the old area fallback in any of 1,934 rows, so under v2 the only coordinate
that can move is `log1p_port_area_um2`. The measured change, −0.2% RMSE, is inside
the repeat-to-repeat spread. Ordered TransmonCross ports are correct and worth
having for semantic clarity, but they do not improve v2 prediction.

**The larger TransmonCross effect is the etch layer, not the ports.** Published v0
maps `(1,11)` to etch only for `CapNInterdigitalTee` and `CavityClawRouteMeander`,
so TransmonCross etch was being discarded. Restoring it is worth −29.5% RMSE, seven
times the port effect.

**For CapN the ports do matter**, at −15.7% under v0, consistent with the 9.7% of
designs whose ordering the ports correct.

## 6. Cross-family transfer, GeneralizedCapNInterdigital → CapNInterdigitalTee

Fitted on `log1p(C)`; errors reported in fF after `expm1`.

| method | representation | RMSE fF | R² |
| --- | --- | ---: | ---: |
| source-only, zero-shot | old/v0 | 18.97 | −1.99 |
| source-only, zero-shot | old/v2 (portless target) | 14.13 | −0.66 |
| **source-only, zero-shot** | **v2, port-complete target** | **3.54** | **+0.90** |
| target-only, 100% | old/v2 | 0.248 | 0.9994 |
| target-only, 100% | v2, port-complete target | 0.277 | 0.9993 |
| transfer, 2% labels | old/v2 | 1.352 | 0.9842 |
| transfer, 2% labels | v2, port-complete target | 2.522 | 0.9449 |

**This is the strongest result in the study.** Correcting the CapN conductor
geometry moves cross-family zero-shot from R² −0.66 to **+0.90**, a 4× reduction in
RMSE, with no labels from the target family at all. The previously published CapN
GDS was missing enough positive CPW path that the family did not sit in the right
place relative to the Generalized couplers; with the corrected geometry a model
trained purely on Generalized NCaps transfers to Tee couplers.

Note the honest counterweight: once target labels are plentiful the advantage
disappears and slightly reverses (0.277 vs 0.248 fF at 100% labels). The correction
buys **cross-family alignment**, not within-family fit.

Only v2 supports this comparison without a rebuild, because it consults no
catalogue statistics and the published Generalized rows and freshly built CapN rows
already share a space. No fit-on-write representation can be mixed this way.

## 7. A fit-on-write confound, quantified

TransmonCross geometry is provably identical between the two releases, yet
`new/v0` (0.392 fF) is **20% worse** than `old/v0` (0.326 fF). Since the bitmaps
are byte-identical, the entire difference is renormalization: v0's parameter and
moment statistics were refit over a 2,828-row two-family cohort instead of the
published 24,106-row four-family cohort.

A second symptom: `new/v0` and `new/v0-etch` differ on **CapN** (2.008 vs 1.931 fF)
even though CapN's role map is identical in both builds. Changing how *TransmonCross*
is encoded changed CapN's vectors, through the shared normalization.

This is a clean, quantified demonstration of why v2 was made catalogue-free, and it
means any old-versus-new v0 or v1 comparison mixes geometry with renormalization.
Only the v2 comparisons in section 5 isolate geometry cleanly.

## 8. Limitations

- The Generalized source family still uses the published portless-era GDS; it was
  not regenerated, so the cross-family study is "new target, old source".
- `new/v1-local` is fit on a two-family cohort and is not comparable to published
  v1. It is reported for completeness only.
- `new/v0-ports` versus `new/v0` conflates ports with etch for TransmonCross; the
  `new/v0-etch` variant exists precisely to separate them and should be preferred
  for that decomposition.
- Twelve grouped repeats of one simulated catalogue are not independent physical
  experiments.
- CapN has 894 designs; its error bars are correspondingly wide (RMSE sd up to
  0.4 fF for old/v0).
- Only the two primary targets were modelled. The other capacitance-matrix entries
  (`cross_to_ground`, `claw_to_ground`, `top_to_ground`, `bottom_to_ground`) are
  scientifically meaningful and were left untouched by design.

## 9. Recommended next experiment

Regenerate the **GeneralizedCapNInterdigital** sweep with the same direct-qgeometry
exporter and re-run section 6. That would make the cross-family study
port-complete on both sides and would test whether the zero-shot R² +0.90 result
improves further or was specific to fixing the target. It would also close the
upstream gap where the published layout repository holds 10,000 of 16,379
`q3d_cap` artifacts.

Second: extend the published v0/v1 role maps to recognize etch and ports for every
family. The −29.5% TransmonCross etch effect shows the current map is discarding
real geometry, and that is a defect in the published standards rather than a
property of the data.

## Files

| file | contents |
| --- | --- |
| `metrics.json` | machine-readable config, all tables, effect isolation |
| `within_family.csv`, `cross_family.csv` | flat metric tables |
| `raw_metrics.parquet` | every repeat and fraction, unaggregated |
| `terminal_ordering.json`, `.parquet` | full-sweep terminal and port verification |
| `learning_curves.png`, `cross_family.png` | figures |
| `config.json` | seeds, split policy, revisions, targets, units |
