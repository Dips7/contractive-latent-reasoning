# Claim Register: Stability and Long-Range Information Propagation Study

**Date**: 2026-11-16  
**Document Status**: PHASE 0 CLAIM REGISTER & HYPOTHESIS BOUNDARY  
**Governing Branch**: `stability_investigation/`  
**Parent Investigation**: Contractive Graph Dynamics Rework  

---

## 1. Context and Objective

This investigation examines whether stronger continuous-time contraction is associated with weaker long-distance reachability performance, identifies what mathematical and architectural mechanisms explain the observed pattern, and tests whether a model can retain a mathematically defensible stability guarantee while improving long-range predictions.

To maintain strict scientific hygiene, this claim register establishes an unambiguous firewall between:
1. What has been mathematically proved in the primary manuscript;
2. What has been empirically observed in completed historical runs;
3. What is hypothesized for this new investigation; and
4. What remains open and unresolved.

---

## 2. Established Claims (Mathematical Guarantees in Current Manuscript)

The following claims are proved by ordinary mathematics under the explicit assumptions of the continuous matrix vector field $\dot{Z}(t) = \tanh(P Z(t) B_1) B_2 - Z(t) D + X$:

| Claim ID | Proposition / Mathematical Guarantee | Formal Scope & Assumptions | Verification Status |
| :--- | :--- | :--- | :---: |
| **EST-01** | **Global Solution Existence & Uniqueness**: For any initial state $Z(0) \in \mathbb{R}^{n \times d}$, a unique global trajectory $Z(t)$ exists on $[0, \infty)$. | Picard–Lindelöf theorem on complete Euclidean space $\mathbb{R}^{n \times d}$ under global Lipschitz constant $L = \beta + d_{\max}$. | **PROVED (MATH)** |
| **EST-02** | **Continuous Trajectory Contraction**: For two trajectories $Z_1(t), Z_2(t)$ driven by the same source $X$, $\|Z_1(t) - Z_2(t)\|_F \le e^{-\alpha t} \|Z_1(0) - Z_2(0)\|_F$ where $\alpha \triangleq d_{\min} - \beta > 0$. | Demidovich contraction via Frobenius energy method and Grönwall inequality under coordinate damping $d_{\min} > \beta \triangleq \|P\|_2 \|B_1\|_2 \|B_2\|_2$. | **PROVED (MATH)** |
| **EST-03** | **Equilibrium Existence & Uniqueness**: There exists a unique stationary state $Z^* \in \mathbb{R}^{n \times d}$ such that $f(Z^*; X) = 0$, attracting all trajectories exponentially. | Banach Fixed-Point Theorem on $\mathbb{R}^{n \times d}$ applied to strongly monotone Picard resolvent $H(Z) = Z - \frac{\alpha}{L^2} F(Z)$ with modulus $\gamma = \sqrt{1 - \alpha^2/L^2} < 1$. | **PROVED (MATH)** |
| **EST-04** | **Additive Disturbance Dissipation**: For Carathéodory solutions under locally Bochner-integrable disturbance $w(t)$, $\|Z(t) - Z^*\|_F \le e^{-\alpha t}\|Z(0) - Z^*\|_F + \int_0^t e^{-\alpha(t-s)}\|w(s)\|_F ds$. | Dini derivative along perturbed trajectory with integrating factor $e^{\alpha t}$. | **PROVED (MATH)** |
| **EST-05** | **Spectral Boundedness of Directed Graph Operator**: For directed bipartite normalization $N_{ij} = C_{ij}/\sqrt{\text{out}_i \text{in}_j}$ with $P = N^\top$, $\|P\|_2 \le 1.000000$ strictly. | Weighted Cauchy–Schwarz / Jensen inequality on directed edges, handling isolated and sink/source vertices. | **PROVED (MATH)** |
| **EST-06** | **Mechanized Arithmetic Boundary**: Integer margin deduction `damping_margin` is verified in Lean 4 via `omega` with 0 custom axioms. | Bounded strictly to integer margin arithmetic; real-analysis and ODE proofs are ordinary mathematics. | **PROVED (LEAN 4)** |

*Critical Theoretical Boundary*: None of the established mathematical guarantees bound discrete numerical integrator stability (e.g. RK4 or Euler with finite step size $h$), classification margin robustness after applying an MLP readout, or performance on downstream reachability queries.

---

## 3. Observed Empirical Results (Existing Historical Benchmarks)

The following empirical observations are recorded in verified run records across the 40 Cora runs (`results/reworked/`) and Phase 2 exploratory evaluations (`external_validation/results/confirmatory_v2/`):

| Observation ID | Empirical Finding | Tested Condition | Source Artifact |
| :--- | :--- | :--- | :--- |
| **OBS-01** | **Multi-Hop Positive Recall Decay**: Damped continuous models (`RK4_damped`, `Euler96_damped`) exhibit steep recall suppression as shortest-path distance increases: 50.00% at hop 1, 8.46% at hop 2, and 0.00% at hops 3–6. | Cora citation network, 10 runs (5 query splits $\times$ 2 seeds). | `results/processed/reworked_summary.json` |
| **OBS-02** | **Higher Discrete Baseline Recall**: Discrete `GNN6` maintains substantial mean positive recall across all hops: 59.45% (hop 1), 72.45% (hop 2), 67.51% (hop 3), 64.09% (hop 4), 60.25% (hop 5), and 65.53% (hop 6). | Cora citation network, 10 runs (5 query splits $\times$ 2 seeds). | `results/processed/reworked_summary.json` |
| **OBS-03** | **Undamped Superiority over Damped**: `RK4_undamped` achieves significantly higher mean accuracy (65.33% ± 0.91%) and retains non-zero recall up to hop 4 (34.24% at hop 3, 6.30% at hop 4), compared to `RK4_damped` (54.70% ± 0.45%, 0% at hops 3–6). | Cora citation network, 10 runs. | `results/processed/reworked_summary.json` |
| **OBS-04** | **Discrete Discretization Invariance**: `RK4_damped` (24 steps, $h \approx 0.1667$) and `Euler96_damped` (96 steps, $h \approx 0.0417$) produce identical predicted labels on all 3,000 held-out test queries across all 10 split/seed conditions. | Cora citation network, 10 matched conditions. | `results/processed/reworked_summary.json` |
| **OBS-05** | **Exploratory Damping Sweep Pattern**: In the Phase 2 exploratory sweep on CiteSeer split 42, nominal damping offsets $\alpha \in \{-0.5, 0.0, 0.5, 1.0, 2.0\}$ showed progressive suppression of hop-1 recall under bilinear readout (87.1% $\to$ 35.5% $\to$ 3.2% $\to$ 0.0% $\to$ 0.0%). | CiteSeer split 42, exploratory sweep. | `external_validation/results/gate_d_audit_report.json` |

*Critical Empirical Boundary*: The observed empirical associations do NOT prove that contraction *caused* the recall decay. Damping, optimization horizon, learning dynamics, readout thresholding, initialization PRNG, and graph topology may confound the observed outcome.

---

## 4. Hypothesized Claims (To Be Tested in this Study)

The following hypotheses define the scientific scope of the new study:

| Hypothesis ID | Statement | Rationale | Falsification / Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **HYP-01** | **Contraction Margin Attenuation**: Higher realized contraction rate $\alpha_{\text{realized}} \triangleq d_{\min} - \|P\|_2 \|B_1\|_2 \|B_2\|_2$ monotonically suppresses the mathematical sensitivity of equilibrium states $\partial Z^*_v / \partial X_s$ as graph distance $d(s, v)$ increases. | Sensitivity analysis of the implicit equilibrium map $Z^*(X)$ under contraction bounds. | **Accepted** if analytical derivation establishes an exponential decay bound in $d(s, v)$ scaled by $\alpha$, and empirical sensitivity matches; **Falsified** if sensitivity is independent of $\alpha$ or non-monotonic. |
| **HYP-02** | **Damping Sweep Parameter Association**: In a parameter sweep over nominal damping settings $d_{\min} \in [0.5, 2.0]$ under fixed optimization budgets and initialization PRNGs, increasing nominal damping is hypothesized to correlate with reduced multi-hop recall and higher cross-entropy loss. | Evaluates whether varying coordinate damping parameters systematically associates with performance differences under this protocol. | **Uninformative for Recall on Saturated Benchmark; Retained Descriptively for Loss**: Due to a 100.0% test accuracy/recall ceiling across all damping settings on the tested small digraphs, the benchmark was non-discriminative for classification recall under these conditions (it does not establish that damping has no effect on non-saturated tasks). However, both validation and held-out test cross-entropy losses scaled monotonically with nominal damping at the seed-averaged level under the evaluated protocol (reported strictly as a descriptive empirical association, not an established causal mechanism). |
| **HYP-03** | **Stability-Preserving Long-Range Recovery**: An augmented architecture combining a provably contractive latent vector field with a bounded multiscale pathway or state-independent global features can significantly improve long-range recall while preserving a mathematically proved contraction margin $\alpha > 0$. | Preserves contractive equilibrium guarantees while overcoming local message-passing bottlenecks. | **Partially Supported for Cross-Entropy Loss on Synthetic Digraphs; Open for Classification Accuracy**: Candidate B preserves certified Demidovich stability while reducing held-out test loss by 47%–69%. However, because test accuracy and recall saturated at 100% ceiling across all models, improvement in classification reachability remains unverified. |

---

## 5. Open Questions (Unresolved Confounds)

1. **Optimization Horizon & Depth**: Models were trained for only 3 epochs (Cora) or 5 epochs (synthetic). Does extended training enable damped continuous dynamics to learn non-zero long-hop reachability, or is the suppression structurally insurmountable under coordinate damping?
2. **Readout Bottlenecks**: The two-layer MLP readout operates on the concatenation of $Z_{\text{target}}$ and $\ln(\|Z_{\text{target}}\|_2 + 10^{-12})$. Does the readout map's decision threshold artificially collapse predictions to negative when latent norms attenuate below a numerical threshold?
3. **Graph Topology & Cut Size**: Does information attenuation in contractive models depend primarily on shortest-path hop count $d(s, t)$, or is it governed by topological bottlenecks (cut size, conductance, directed curvature)?
4. **Architectural Alternatives**: Can graph rewiring (e.g., FoSR) or global-mixing layers (e.g., $S^3$GNN) mitigate long-range decay in continuous G-NODEs without violating the contractive Lipschitz bound?

---

## 6. Phase 0 Gate Sign-off

- [x] Baseline manifest recorded for all current manuscript, source code, verified results, and audit records (`stability_investigation/BASELINE_MANIFEST.json`).
- [x] Phase 2 exploratory results preserved with disclosed protocol deviations.
- [x] Explicit claim register distinguishing Established, Observed, Hypothesized, and Open questions created.
- [x] Primary hypotheses and decision criteria frozen before running new experiments.
- **GATE RESULT**: **PHASE 0 PASSED**. Ready to proceed to Phase 1 (Theoretical Sensitivity Analysis).
