# Architectural Comparison and Stability-Preserving Evaluation Report (Phases 4 & 5 Reconciled)

**Date**: 2026-11-16 (Revised)  
**Document Status**: PHASES 4 & 5 DELIVERABLE & GATE REPORT (REVISED WITH TRACEABLE LOSS RECORDS & QUALIFIED CLAIMS)  
**Governing Branch**: `stability_investigation/`  
**Execution Script**: `stability_investigation/scripts/run_architectural_comparison.py`  
**Audit Evaluation Script**: `stability_investigation/scripts/evaluate_and_persist_test_loss.py`  
**Results Data**: `stability_investigation/results/architectural_comparison/architectural_comparison_results.json`  
**Audit Certificate**: `stability_investigation/results/architectural_comparison/test_loss_evaluation_certificate.json`  

---

## 1. Executive Summary & Epistemic Boundaries

This report evaluates four architectural strategies designed to analyze long-range signal propagation while preserving continuous-time Demidovich contraction guarantees. Across 45 trained models evaluated on held-out test graph instances, we benchmarked:
1. `Baseline_Damped_1.5`: Certified single-scale contractive ODE ($d_{\min}=1.5, \alpha \ge 0.5$).
2. `Candidate_A_Tuned_1.05`: Contraction margin tuning ($d_{\min}=1.05, \alpha \ge 0.05$).
3. `Candidate_B_Multiscale_1.5`: Multiscale Skip-Hop Contractive ODE ($P_1 = P, P_2 = P^2, d_{\min}=1.5, \alpha \ge 0.5$).
4. `Candidate_C_Structural_1.5`: Structural Readout ODE ($d_{\min}=1.5, \alpha \ge 0.5$ + local degree features).
5. `Baseline_Discrete_GNN6`: 6-layer discrete message-passing baseline.

### Primary Observations & Critical Qualifications:
1. **Benchmark Ceiling Effect**:
   All 45 evaluated models—including all continuous models and the discrete GNN6 baseline—achieved **100.0% test accuracy, 100.0% specificity, and 100.0% recall across all evaluated queries**. 
   - *Epistemic Consequence*: Because classification performance is at ceiling on this synthetic benchmark, the results **cannot and do not demonstrate an improvement in reachability classification or generalization performance** over the baseline.
2. **Scope of Hop Distances**:
   Full reachability up to 6 hops is evaluated strictly on `chain_ladder` and `bottleneck`. Because `random_dag` contains only 14 queries at hop 4, 1 at hop 5, and 0 at hop 6 across all instances, it evaluates reachability across hops 1 through 3 and provides no empirical evidence for 6-hop performance.
3. **Reproducible Test-Loss Record**:
   Differences between architectures are observed in **cross-entropy loss** (both validation loss and held-out test loss), independently audited and persisted across all 45 checkpoints with SHA-256 hashes:
   - **Candidate B (Multiscale Skip-Hop)** achieved lower held-out test loss across all three families:
     - `chain_ladder`: Test loss reduced by **53.0%** ($0.001051 \to 0.000494$).
     - `bottleneck`: Test loss reduced by **47.7%** ($0.001366 \to 0.000715$).
     - `random_dag`: Test loss reduced by **69.3%** ($0.008138 \to 0.002501$).
   - **Candidate C (Structural Readout)** achieved an **88.8% test loss reduction** on `random_dag` ($0.008138 \to 0.000913$), but showed slightly higher test loss on `bottleneck` ($+0.4\%$).
4. **Architectural Qualifications**:
   - **Candidate B (Multiscale Skip-Hop)**: The proof that $\|P^2\|_2 \le \|P\|_2^2 \le 1.0$ guarantees uniform Demidovich contraction ($\mu_2(J) \le -\alpha < 0$) is mathematically exact. However, the theoretical spatial influence speedup ($\mathcal{O}(\rho^{\lceil k/2 \rceil})$) is an **upper-bound argument on graph support**: the earliest *potentially* non-zero term can appear at $\lceil k/2 \rceil$, but weight alignments or activation derivatives could render it zero. It does not guarantee larger learned signals or superior predictions on real tasks.
   - **Candidate C (Structural Readout)**: In synthetic families such as chains and layered DAGs, node degrees can correlate with topological depth and reachability. Because no feature-only baseline was conducted, the degree inputs cannot be definitively asserted to be "zero leakage" on these synthetic structures. The claim is narrowed to: degree features provide local topological context without explicit path matrices.

---

## 2. Comparative Results Across Graph Families

### Table 1: Comprehensive Optimization & Generalization Metrics (Mean $\pm$ Sample SD over 3 Seeds per Architecture)

| Graph Family | Architecture | Test Acc (%) | Test Spec (%) | Test Rec (%) | Realized Margin Lower Bound $\bar{\alpha}$ | Mean Val Loss | Held-Out Test Loss | $\Delta$ Test Loss vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **chain_ladder** | `Baseline_Damped_1.5` | 100.0 | 100.0 | 100.0 | +0.93 | 0.00094 $\pm$ 0.00028 | 0.001051 $\pm$ 0.000324 | Baseline |
| | `Candidate_A_Tuned_1.05` | 100.0 | 100.0 | 100.0 | +0.49 | 0.00086 $\pm$ 0.00026 | 0.000929 $\pm$ 0.000300 | -11.6% |
| | `Candidate_B_Multiscale_1.5` | 100.0 | 100.0 | 100.0 | +0.89 | **0.00045 $\pm$ 0.00004** | **0.000494 $\pm$ 0.000031** | **-53.0%** |
| | `Candidate_C_Structural_1.5` | 100.0 | 100.0 | 100.0 | +0.98 | **0.00046 $\pm$ 0.00018** | **0.000496 $\pm$ 0.000191** | **-52.8%** |
| | `Baseline_Discrete_GNN6` | 100.0 | 100.0 | 100.0 | 0.00 | 0.00003 $\pm$ 0.00001 | 0.000234 $\pm$ 0.000072 | -77.7% |
| **bottleneck** | `Baseline_Damped_1.5` | 100.0 | 100.0 | 100.0 | +0.84 | 0.00106 $\pm$ 0.00025 | 0.001366 $\pm$ 0.000338 | Baseline |
| | `Candidate_A_Tuned_1.05` | 100.0 | 100.0 | 100.0 | +0.43 | 0.00094 $\pm$ 0.00028 | 0.001169 $\pm$ 0.000321 | -14.4% |
| | `Candidate_B_Multiscale_1.5` | 100.0 | 100.0 | 100.0 | +0.88 | **0.00055 $\pm$ 0.00019** | **0.000715 $\pm$ 0.000191** | **-47.7%** |
| | `Candidate_C_Structural_1.5` | 100.0 | 100.0 | 100.0 | +0.93 | 0.00089 $\pm$ 0.00048 | 0.001371 $\pm$ 0.000530 | +0.4% |
| | `Baseline_Discrete_GNN6` | 100.0 | 100.0 | 100.0 | 0.00 | 0.00004 $\pm$ 0.00002 | 0.000076 $\pm$ 0.000022 | -94.4% |
| **random_dag** | `Baseline_Damped_1.5` | 100.0 | 100.0 | 100.0 | +0.94 | 0.00733 $\pm$ 0.00652 | 0.008138 $\pm$ 0.006977 | Baseline |
| | `Candidate_A_Tuned_1.05` | 100.0 | 100.0 | 100.0 | +0.50 | 0.00629 $\pm$ 0.00517 | 0.006908 $\pm$ 0.005558 | -15.1% |
| | `Candidate_B_Multiscale_1.5` | 100.0 | 100.0 | 100.0 | +0.95 | **0.00212 $\pm$ 0.00032** | **0.002501 $\pm$ 0.000318** | **-69.3%** |
| | `Candidate_C_Structural_1.5` | 100.0 | 100.0 | 100.0 | +1.05 | **0.00104 $\pm$ 0.00026** | **0.000913 $\pm$ 0.000267** | **-88.8%** |
| | `Baseline_Discrete_GNN6` | 100.0 | 100.0 | 100.0 | 0.00 | 0.00029 $\pm$ 0.00010 | 0.000458 $\pm$ 0.000102 | -94.4% |

---

## 3. Provenance and Independent Traceability

- **Audited Dataset Manifest**: `stability_investigation/data/synthetic/benchmark_manifest.json`  
  - SHA-256 Digest: `8d5b02c591989e49a168a662727dd51ede02f6778dde643aca7c3d584e316342`
- **Audit Evaluation Script**: `stability_investigation/scripts/evaluate_and_persist_test_loss.py`
- **Machine-Readable Certificate**: `stability_investigation/results/architectural_comparison/test_loss_evaluation_certificate.json`
- Every individual checkpoint file in `stability_investigation/results/architectural_comparison/checkpoints/` has its SHA-256 digest, individual per-seed test loss, and validation loss recorded in the certificate.

---

## 4. Status

- [x] Mathematical stability proof for Candidate B verified under Demidovich norm.
- [x] Multiscale spatial influence explicitly qualified as an upper bound on structural graph support.
- [x] Per-run test cross-entropy losses evaluated, persisted, and hashed across all 45 checkpoints.
- [x] Benchmark ceiling effect explicitly disclosed.
- [x] Scope of 6-hop reachability limited strictly to chains and bottlenecks.

**Phases 4 & 5 Status**: **AUDITED WITH QUALIFIED CLAIMS & REPRODUCIBLE TEST LOSS RECORDS**
