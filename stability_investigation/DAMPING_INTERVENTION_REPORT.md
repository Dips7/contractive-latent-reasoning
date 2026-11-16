# Damping Parameter Sweep and Empirical Sensitivity Report (Phase 3 Reconciled)

**Date**: 2026-11-16 (Revised)  
**Document Status**: PHASE 3 DELIVERABLE & GATE REPORT (REVISED WITH TRACEABLE TEST LOSS RECORDS & QUALIFIED SCOPE)  
**Governing Branch**: `stability_investigation/`  
**Execution Script**: `stability_investigation/scripts/run_damping_intervention.py`  
**Audit Evaluation Script**: `stability_investigation/scripts/evaluate_and_persist_test_loss.py`  
**Data Source**: `stability_investigation/data/synthetic/` (36 audited instances, 7,230 verified queries)  
**Results Data**: `stability_investigation/results/damping_intervention/damping_intervention_results.json`  
**Audit Certificate**: `stability_investigation/results/damping_intervention/damping_test_loss_certificate.json`  

---

## 1. Executive Summary & Epistemic Boundaries

This report delivers the results of the Phase 3 Damping Parameter Sweep across 72 trained model instances (3 graph families $\times$ 8 model configurations $\times$ 3 PRNG seeds), evaluating whether varying nominal damping settings ($d_{\min}$) empirically associates with differences in validation loss, held-out test loss, and multi-hop reachability on synthetic digraphs under fixed training budgets.

### Key Empirical Findings:
1. **Monotonic Loss Scaling with Nominal Damping**:
   Across all three topological families (`chain_ladder`, `bottleneck`, `random_dag`), seed-averaged validation loss and held-out test loss scale **monotonically** with the nominal damping lower bound $d_{\min} \in [0.50, 2.00]$:
   - In `chain_ladder`: mean test loss rises from $0.000761$ ($d_{\min} = 0.50$) to $0.001223$ ($d_{\min} = 2.00$).
   - In `bottleneck`: mean test loss rises from $0.000932$ to $0.001642$.
   - In `random_dag`: mean test loss rises from $0.005160$ to $0.009249$.
   - The undamped continuous ODE ($D = 0$) achieves the lowest continuous loss across all families ($0.000213$ to $0.002014$).
2. **Scope of the Damping Trend & Per-Checkpoint Margin Audit**:
   We audited the relationship between loss, nominal damping, and the per-checkpoint computed conservative contraction margin lower bound:
   $$\alpha_{\text{realized}} = \min(d) - \sigma_1(B_1)\sigma_1(B_2)$$
   Across all three families, the seed-averaged realized margin tracks nominal damping monotonically (rising from $\bar{\alpha}_{\text{realized}} \approx -0.05$ at $d_{\min} = 0.50$ to $+1.41$ at $d_{\min} = 2.00$). However, due to run-to-run optimization fluctuations across individual checkpoints (e.g., minor validation/test loss inversions in seed 42 for `bottleneck` and `chain_ladder`), individual checkpoints do not follow a strict monotonic ordering against either nominal damping or computed per-checkpoint margins. Therefore, the monotonic claim is tied strictly to nominal damping settings at the seed-averaged level under this specific protocol, rather than asserted as a per-checkpoint law across realized margins. Furthermore, because varying nominal damping under fixed hyperparameters evaluates associations across configurations without isolating confounders, this relationship is framed descriptively rather than as an established causal mechanism.
3. **Ceiling Effect & Hop Distance Scope**:
   - Under multi-instance training (5 train graphs, batch size 64, validation checkpoint selection), all evaluated models—whether contractive ($d_{\min} = 1.50, \bar{\alpha} = +0.93$), marginally stable ($d_{\min} = 1.05$), non-contractive ($d_{\min} = 0.50$), undamped ($D = 0$), or discrete (`GNN6`)—achieved **100.0% test accuracy, 100.0% specificity, and 100.0% recall across all evaluated queries**.
   - **Scope of Multi-Hop Evidence**: Full reachability up to 6 hops is evaluated strictly on `chain_ladder` and `bottleneck`. Because `random_dag` contains only 14 queries at hop 4, 1 at hop 5, and 0 at hop 6 across all instances, it evaluates reachability across hops 1 through 3 and provides no empirical evidence for 6-hop performance.
4. **What This Result Does and Does Not Establish**:
   - **Does Establish**: This demonstrates that continuous-time contraction does *not* mathematically preclude learning multi-hop reachability up to 6 hops on small directed chains ($N=40$) and bottlenecks ($N=48$) under multi-instance training.
   - **Does NOT Establish**: This does **not** prove why recall previously dropped on large transductive citation networks (Cora, CiteSeer, PubMed). The hypothesis that degree over-squashing ($(1/d)^k$), contraction damping ($\rho^k$), and readout thresholding caused the citation network collapse remains an **unproven hypothesis** that requires controlled testing on scale-free graphs.

---

## 2. Quantitative Summary Tables (Mean $\pm$ Sample SD over 3 Seeds)

### 2.1 Family: `chain_ladder` (Balanced hops 1–6)

| Configuration | Test Acc (%) | Spec (%) | Rec (%) | Hop 1..6 | Mean Val Loss | Held-Out Test Loss | $\bar{\alpha}_{\text{realized}}$ (Lower Bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Damped (d_min=0.50)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000675 $\pm$ 0.000288 | 0.000761 $\pm$ 0.000307 | -0.03 |
| `Damped (d_min=0.80)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000762 $\pm$ 0.000257 | 0.000825 $\pm$ 0.000260 | +0.25 |
| `Damped (d_min=1.05)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000858 $\pm$ 0.000264 | 0.000929 $\pm$ 0.000300 | +0.49 |
| `Damped (d_min=1.20)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000888 $\pm$ 0.000268 | 0.000970 $\pm$ 0.000306 | +0.64 |
| `Damped (d_min=1.50)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000943 $\pm$ 0.000282 | 0.001051 $\pm$ 0.000324 | +0.93 |
| `Damped (d_min=2.00)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.001057 $\pm$ 0.000321 | 0.001223 $\pm$ 0.000376 | +1.40 |
| `ContinuousUndamped` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000056 $\pm$ 0.000037 | 0.000213 $\pm$ 0.000116 | -1.00 |
| `DiscreteGNN_L6` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000032 $\pm$ 0.000014 | 0.000234 $\pm$ 0.000072 | 0.00 |

### 2.2 Family: `bottleneck` (Balanced hops 1–6)

| Configuration | Test Acc (%) | Spec (%) | Rec (%) | Hop 1..6 | Mean Val Loss | Held-Out Test Loss | $\bar{\alpha}_{\text{realized}}$ (Lower Bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Damped (d_min=0.50)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000780 $\pm$ 0.000222 | 0.000932 $\pm$ 0.000261 | -0.08 |
| `Damped (d_min=0.80)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000874 $\pm$ 0.000267 | 0.001074 $\pm$ 0.000298 | +0.19 |
| `Damped (d_min=1.05)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000936 $\pm$ 0.000282 | 0.001169 $\pm$ 0.000321 | +0.43 |
| `Damped (d_min=1.20)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.001009 $\pm$ 0.000284 | 0.001295 $\pm$ 0.000339 | +0.57 |
| `Damped (d_min=1.50)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.001060 $\pm$ 0.000249 | 0.001366 $\pm$ 0.000338 | +0.84 |
| `Damped (d_min=2.00)` | 100.0 | 100.0 | 100.0 | 100.0 | 0.001236 $\pm$ 0.000234 | 0.001642 $\pm$ 0.000335 | +1.30 |
| `ContinuousUndamped` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000326 $\pm$ 0.000155 | 0.000598 $\pm$ 0.000322 | -1.00 |
| `DiscreteGNN_L6` | 100.0 | 100.0 | 100.0 | 100.0 | 0.000039 $\pm$ 0.000018 | 0.000076 $\pm$ 0.000022 | 0.00 |

### 2.3 Family: `random_dag` (Hops 1–3 dense; hop 4: 14 queries, hop 5: 1 query, hop 6: 0 queries)

| Configuration | Test Acc (%) | Spec (%) | Rec (%) | Hop 1..3 | Hop 4* | Hop 5* | Mean Val Loss | Held-Out Test Loss | $\bar{\alpha}_{\text{realized}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Damped (d_min=0.50)` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.004727 $\pm$ 0.002839 | 0.005160 $\pm$ 0.003119 | -0.05 |
| `Damped (d_min=0.80)` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.005620 $\pm$ 0.004128 | 0.006142 $\pm$ 0.004518 | +0.25 |
| `Damped (d_min=1.05)` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.006293 $\pm$ 0.005167 | 0.006908 $\pm$ 0.005558 | +0.50 |
| `Damped (d_min=1.20)` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.006671 $\pm$ 0.005707 | 0.007346 $\pm$ 0.006096 | +0.65 |
| `Damped (d_min=1.50)` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.007334 $\pm$ 0.006524 | 0.008138 $\pm$ 0.006977 | +0.94 |
| `Damped (d_min=2.00)` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.008216 $\pm$ 0.007386 | 0.009249 $\pm$ 0.007984 | +1.43 |
| `ContinuousUndamped` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.001288 $\pm$ 0.000622 | 0.002014 $\pm$ 0.000952 | -1.00 |
| `DiscreteGNN_L6` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.000286 $\pm$ 0.000096 | 0.000458 $\pm$ 0.000102 | 0.00 |

*\*Note: Evaluated test queries in `random_dag` total 2 queries at hop 4, 1 query at hop 5, and 0 queries at hop 6.*

---

## 3. Provenance and Traceability

- **Audited Dataset Manifest**: `stability_investigation/data/synthetic/benchmark_manifest.json`  
  - SHA-256: `8d5b02c591989e49a168a662727dd51ede02f6778dde643aca7c3d584e316342`
- **Evaluation Script**: `stability_investigation/scripts/evaluate_and_persist_test_loss.py`
- **Audit Certificate**: `stability_investigation/results/damping_intervention/damping_test_loss_certificate.json`  
  Records individual checkpoint SHA-256 hashes, seed-specific test losses, and validation losses for all 72 models.
