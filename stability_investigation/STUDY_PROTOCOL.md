# Study Protocol: Stability and Long-Range Information Propagation

**Date**: 2026-11-16  
**Document Status**: PHASE 0 STUDY PROTOCOL  
**Governing Branch**: `stability_investigation/`  
**Parent Investigation**: Contractive Graph Dynamics Rework  

---

## 1. Scientific Aim and Hypotheses

### Primary Question
Does mathematical contraction in continuous-depth graph vector fields inherently trade off against long-range relational information propagation, and can an architecture retain a mathematically certified Demidovich contraction margin while improving multi-hop signal transmission?

### Hypotheses
1. **H1 (Contraction–Attenuation Trade-off)**: In the autonomous system $\dot{Z} = \tanh(P Z B_1) B_2 - Z D + X$, the mathematical sensitivity $\|\partial Z^*_v / \partial X_s\|$ decays exponentially with shortest-path distance $d(s, v)$ at a rate governed by the contraction margin $\alpha = d_{\min} - \beta$.
2. **H2 (Empirical Damping Sweep Hypothesis)**: Under fixed initialization PRNGs, optimization budgets, readout heads, and graph splits, varying nominal damping settings is hypothesized to correlate with systematic differences in validation/test cross-entropy loss. This is preregistered as an empirical association to be evaluated descriptively under this protocol, not as an established causal mechanism.
3. **H3 (Stability-Preserving Architectural Recovery)**: Incorporating a multiscale bounded pathway or structural features preserves strict continuous-time contraction while restoring long-range reachability.

---

## 2. Experimental Boundaries & Safeguards

1. **Separation from Authoritative Cora Manuscript**:
   - The authoritative Cora manuscript (`manuscript/manuscript.md`) and submission package (`submission_package/reworked/`) remain strictly frozen.
   - All code, data generators, manifests, and checkpoints for this new study are isolated in `stability_investigation/`.
2. **Phase 2 Exploratory Preservation**:
   - Existing Phase 2 materials in `external_validation/results/confirmatory_v2/` remain cataloged as exploratory with disclosed deviations.
3. **Controlled Benchmark Rules (Phase 2 & 3)**:
   - **No Replacement Sampling**: If a shortest-path hop bucket has fewer queries than the target count, the target count is lowered. No replacement sampling is permitted.
   - **Zero Train/Val/Test Leakage**: Evaluated across directed node pairs $(s, t)$. Train, validation, and test partitions must be strictly disjoint.
   - **Zero Internal Duplicates**: Each partition must contain strictly unique query pairs.
   - **Graph-Level Splits**: Graph instances are the unit of independent replication. Query pairs sampled within the same graph topology are not independent graph replicates.
   - **Controlled Initialization**: PyTorch PRNG seeds must be set *before* model construction, isolating initial weight states.

---

## 3. Phase-Wise Execution Gates

| Phase | Description | Key Deliverables | Gate Condition to Proceed |
| :---: | :--- | :--- | :--- |
| **0** | Freeze Starting Point & Define Claims | `BASELINE_MANIFEST.json`, `CLAIM_REGISTER.md`, `STUDY_PROTOCOL.md` | Existing manuscript claims and new hypotheses strictly firewalled. |
| **1** | Mathematical Sensitivity Analysis | `THEORY_NOTE.md`, `SENSITIVITY_ANALYSIS.md`, Proof obligations | Sensitivity bounds derived; candidate modifications mathematically specified. |
| **2** | Controlled Synthetic Benchmark | Generator script, clean query manifests, data audit report | Audit verifies 0 overlaps, 0 duplicates, graph-level split integrity. |
| **3** | Empirical Damping Parameter Sweep | Factorial execution records, checkpoint inventory, replay log | Descriptive parameter sweep evaluates nominal damping associations under fixed budgets. |
| **4** | Stability-Preserving Improvements | Modified architecture, contraction proof, comparative tables | Contraction proved under implemented assumptions; no unproven stability claims. |
| **5** | Independent Audit & Reproduction | Audit certificate, replay script output, discrepancy ledger | 100% of checkpoints reloaded and predictions replayed bit-for-bit. |
| **6** | Synthesis & Manuscript Update | Revised draft, claim-evidence matrix update | All quantitative claims trace to verified artifacts. |

---

## 4. Phase 0 Gate Sign-off

- Baseline manifest verified: Yes (`BASELINE_MANIFEST.json`, 25 core files).
- Claim register established: Yes (`CLAIM_REGISTER.md`).
- Study protocol established: Yes (`STUDY_PROTOCOL.md`).
- Gate sign-off: **PHASE 0 GATE PASSED**. Proceeding to Phase 1.
