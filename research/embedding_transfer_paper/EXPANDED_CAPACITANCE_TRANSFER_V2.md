# Expanded capacitance-transfer study v2

Date: 2026-10-01

This report supersedes the earlier 2,682-row Tutorial 24 closeout. The updated
study uses the new Tutorial 25 campaigns, the complete v4 release, and all
currently joinable capacitance labels.

## Corpus

| Domain | Unique labeled geometries |
| --- | ---: |
| Generalized IDC `exports` | 2,243 |
| Generalized IDC `exp7_exports` | 1,440 |
| Generalized IDC `exp8_exports` | 16,379 |
| CapNInterdigitalTee | 894 |
| TransmonCross | 1,933 |
| **Total** | **22,889** |

The 45,021-layout v4 atlas supplies additional unlabeled support sets: 15,000
asymmetric IDC, 2,000 concentric, 2,000 star, and 3,082 Firas layouts. All
13,683 GIDC layouts shared by the earlier release and new campaign data have
bit-identical 544-D embeddings. The new data add 6,379 labeled GIDC designs.

## Findings

- Cross-campaign GIDC transfer is strong. Held-campaign median mutual error is
  4.7–9.7%; median ground-sum error is 3.0–6.6%.
- Expanded, family-balanced training improves CapN zero-shot mutual error from
  36.3% to 21.4%.
- Transmon zero-shot remains unreliable: roughly 40–45% mutual error and
  82–87% ground-sum error.
- Cross-topology transfer helps mainly at low K. Target-only training overtakes
  by about K=30 for CapN mutual and K=10 for Transmon mutual.
- Star and concentric are genuine topology-OOD lockboxes but currently lack
  high-fidelity labels, so a confirmatory unseen-topology claim is not yet
  possible.
- A learned target latent/JEPA objective is not the next priority. Topology
  coverage, campaign consistency, family balancing, and local graph structure
  are the limiting factors.

## Matrix audit and model correction

All 20,062 GIDC records contain a full 3x3 database matrix. The 16,379 exp8
designs also retain per-pass solver CSVs. The raw final matrices match the
database matrices exactly, have zero reciprocal asymmetry at stored precision,
contain no positive off-diagonal entries, and are positive definite. Final-pass
branch drift has median 0.0161%, P99 0.0438%, and maximum 0.1639%.

The matrices do not have zero row sums: median maximum row-sum fraction is
0.52–0.57 depending on campaign. They are SPD matrices referenced to an omitted
outer/infinite boundary, not three-node Laplacians containing every reference.
The proposed graph model must therefore predict an SPD matrix, add the missing
reference branches, or use a correctly derived branch circuit. It must not
force the observed 3x3 labels to zero row sum.

## Reproducibility

The deterministic runner is
`scripts/run_expanded_capacitance_transfer_study.py`. It uses 20 nested shot
draws, five model seeds, fixed held-target splits, physical log-capacitance
targets, and input/implementation hashes. Two complete executions reproduced
all 24 generated artifacts bit-for-bit.

The complete generated tables, manifest, and long-form executive summary are
stored under the versioned Hugging Face study path
`studies/expanded-capacitance-transfer-v2/` in the experimental v3/v4 embedding
pull request.
