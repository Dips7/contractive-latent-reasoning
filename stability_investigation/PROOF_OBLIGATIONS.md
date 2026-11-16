# Formal Proof Obligations for Stability-Preserving Architectural Candidates

**Date**: 2026-11-16  
**Document Status**: PHASE 1 THEORY DELIVERABLE  
**Governing Branch**: `stability_investigation/`  
**Parent Framework**: Contractive Graph Neural Dynamics  

---

## 1. Executive Summary & Gating Objectives

In `SENSITIVITY_ANALYSIS.md`, we proved that the spatial signal decay across $k$ shortest-path hops in the continuous contractive ODE $\dot{Z} = \tanh(P Z B_1) B_2 - Z D + X$ obeys the upper bound:
$$\|\mathcal{J}_{v, s}\|_2 \le \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k, \quad \text{where } \alpha = d_{\min} - \beta > 0$$
For the default configuration ($\beta = 1.0, d_{\min} = 1.5, \alpha = 0.5$), the theoretical decay factor upper bound evaluates to $\rho = \frac{1}{1.5} \approx 0.667$ (yielding $\rho^6 \approx 0.088$). We emphasize that this geometric decay is an analytical upper bound on linear equilibrium sensitivity, not a proof of actual learned representation decay.

To enhance long-range reachability without abandoning the mathematical foundation of continuous-time stability, any proposed architecture must satisfy explicit **Proof Obligations** prior to empirical implementation. 

This document defines the mathematical conditions, proof structures, and verification criteria for four candidate architectures:
1. **Candidate A: Margin Tuning ($\alpha \to 0^+$)**
2. **Candidate B: Multiscale Skip-Hop Contractive Dynamics**
3. **Candidate C: State-Independent Structural Readout Augmentation**
4. **Candidate D: Curvature-Preserving Graph Rewiring (FoSR / Directed Expansion)**

---

## 2. Universal Contraction Criteria (The Demidovich Baseline)

For any autonomous continuous-depth vector field $\dot{z} = f(z) + x$ on a Hilbert space $(\mathcal{H}, \langle \cdot, \cdot \rangle_2)$ with induced Euclidean norm $\|\cdot\|_2$:
1. **Demidovich Contraction**: The Jacobian $J_f(z) = \frac{\partial f}{\partial z}(z)$ must satisfy uniform negative one-sided Lipschitz (logarithmic norm) bound:
   $$\mu_2(J_f(z)) \triangleq \sup_{v \neq 0} \frac{\langle v, J_f(z) v \rangle}{\|v\|_2^2} = \lambda_{\max}\left(\frac{J_f(z) + J_f(z)^\top}{2}\right) \le -\alpha < 0 \quad \forall z \in \mathcal{H}$$
2. **Exponential Trajectory Convergence**:
   $$\|z_1(t) - z_2(t)\|_2 \le e^{-\alpha t} \|z_1(0) - z_2(0)\|_2 \quad \forall t \ge 0$$
3. **Picard-Lindelöf Unique Equilibrium**:
   $$\exists ! z^* \in \mathcal{H} \text{ such that } f(z^*) + x = 0, \quad \text{and} \quad \lim_{t \to \infty} z(t) = z^*$$
4. **Disturbance Dissipation**:
   $$\|z^*(x_1) - z^*(x_2)\|_2 \le \frac{1}{\alpha} \|x_1 - x_2\|_2$$

---

## 3. Candidate A: Margin Tuning ($\alpha \to 0^+$)

### 3.1 Architectural Specification
Maintain the existing vector field unmodified:
$$\dot{Z} = \tanh(P Z B_1) B_2 - Z D + X$$
where $P = N^\top$, $\|B_1\|_2 \le 1, \|B_2\|_2 \le 1$, and diagonal damping $d_j = \text{softplus}(\theta_j) + d_{\min}$.  
Vary the nominal lower bound parameter $d_{\min} \in (1.0, 1.5]$ so that the contraction margin $\alpha = d_{\min} - 1.0$ is strictly positive but small (e.g., $\alpha \in [0.05, 0.50]$).

### 3.2 Proof Obligations
- **PO-A1 (Strict Positive Contraction Margin)**:
  - *Requirement*: Prove that for any choice $d_{\min} = 1 + \epsilon$ with $\epsilon > 0$, $\mu_2(J) \le -\epsilon < 0$.
  - *Proof Outline*: Since $\|P\|_2 \le 1$ and $\|B_1\|_2 \|B_2\|_2 \le 1$, $\mu_2(J) \le \|P\|_2 \|B_1\|_2 \|B_2\|_2 - d_{\min} \le 1.0 - (1 + \epsilon) = -\epsilon$. Contraction holds uniformly.
- **PO-A2 (Spatial Influence Expansion)**:
  - *Requirement*: Compute the theoretical upper-bound decay factor $\rho(\epsilon) = \frac{1}{1 + \epsilon}$. Show that as $\epsilon \to 0^+$, $\lim_{\epsilon \to 0^+} \rho(\epsilon) = 1$.
  - *Proof Outline*: Direct evaluation: for $\epsilon = 0.05$, $\rho = 1/1.05 \approx 0.952$. At hop 6, $\rho^6 \approx 0.746$, representing a 5.6-fold increase in signal amplitude over $\epsilon = 0.50$ ($\rho^6 \approx 0.088$).
- **PO-A3 (Condition Number & Integration Horizon Trade-off)**:
  - *Requirement*: Characterize the trade-off between spatial sensitivity and temporal convergence.
  - *Proof Outline*: The relaxation time constant of the continuous system is $\tau = 1/\alpha = 1/\epsilon$. To achieve within $\eta$ of the equilibrium ($\|Z(T) - Z^*\| \le \eta$), the required integration time is:
    $$T \ge \frac{1}{\epsilon} \ln\left(\frac{\|Z(0) - Z^*\|}{\eta}\right)$$
    For $\epsilon = 0.05$, $T \ge 20 \ln(1/\eta)$. The fixed integration horizon $T = 4.0$ in current runs corresponds to only $0.2 \tau$, meaning the trajectory remains far from equilibrium.
  - *Empirical Consequence*: Margin tuning requires scaling the integration horizon $T$ proportionally to $1/\alpha$ or using implicit/adaptive ODE solvers to prevent stiff truncation.

---

## 4. Candidate B: Multiscale Skip-Hop Contractive Dynamics

### 4.1 Architectural Specification
To bridge long graph distances in fewer continuous propagation steps without reducing the damping margin, introduce higher-order graph diffusion operators into the vector field:
$$\dot{Z} = \sum_{m=1}^M \gamma_m \tanh(P_m Z B_{m, 1}) B_{m, 2} - Z D + X$$
where:
- $P_1 = P = N^\top$ (1-hop normalized adjacency).
- $P_m = (N^m)^\top = (P_1)^m$ (m-hop normalized walk operator).
- $\sum_{m=1}^M \gamma_m = 1$ with $\gamma_m \ge 0$ (convex combination).
- Weights satisfy SVD spectral bounds: $\|B_{m, 1}\|_2 \le 1$ and $\|B_{m, 2}\|_2 \le 1$ for all $m \in \{1, \dots, M\}$.
- Damping: $D = \operatorname{diag}(d_1, \dots, d_d)$ with $d_j \ge d_{\min} > 1.0$.

### 4.2 Proof Obligations
- **PO-B1 (Operator Norm Preservation of Higher-Order Walks)**:
  - *Requirement*: Prove that for all $m \ge 1$, $\|P_m\|_2 \le 1$.
  - *Proof*: By submultiplicativity of matrix operator norms:
    $$\|P_m\|_2 = \|(P_1)^m\|_2 \le \|P_1\|_2^m$$
    By Proposition 3.1 of the authoritative manuscript, the bipartite normalized directed matrix $N$ satisfies $\|N\|_2 \le 1$. Hence $\|P_1\|_2 = \|N^\top\|_2 = \|N\|_2 \le 1$. Therefore, $\|P_m\|_2 \le 1^m = 1$ for all $m \in \mathbb{N}$. $\blacksquare$
- **PO-B2 (Preservation of Contraction with Identical Damping Bound)**:
  - *Requirement*: Prove that the multiscale vector field satisfies $\mu_2(J) \le \sum_{m=1}^M \gamma_m - d_{\min} = 1 - d_{\min} = -\alpha < 0$.
  - *Proof*: The total Jacobian with respect to $z = \operatorname{vec}(Z)$ is:
    $$J(z) = \sum_{m=1}^M \gamma_m \left( B_{m, 2}^\top \otimes P_m \right) \mathcal{D}_m(z) \left( B_{m, 1}^\top \otimes I_n \right) - (I_n \otimes D)$$
    where $\mathcal{D}_m(z) = \operatorname{diag}(\operatorname{vec}(1 - \tanh^2(P_m Z B_{m, 1})))$ satisfies $\|\mathcal{D}_m(z)\|_2 \le 1$.
    Using the submultiplicative and subadditive properties of the logarithmic norm:
    $$\mu_2(J(z)) \le \sum_{m=1}^M \gamma_m \|B_{m, 2}^\top \otimes P_m\|_2 \|\mathcal{D}_m(z)\|_2 \|B_{m, 1}^\top \otimes I_n\|_2 - d_{\min}$$
    $$\le \sum_{m=1}^M \gamma_m \|P_m\|_2 \|B_{m, 1}\|_2 \|B_{m, 2}\|_2 - d_{\min}$$
    Since $\|P_m\|_2 \le 1$ (PO-B1) and $\|B_{m, 1}\|_2, \|B_{m, 2}\|_2 \le 1$:
    $$\mu_2(J(z)) \le \sum_{m=1}^M \gamma_m (1)(1)(1) - d_{\min} = \left(\sum_{m=1}^M \gamma_m\right) - d_{\min} = 1 - d_{\min} = -\alpha < 0$$
    Therefore, Candidate B guarantees identical Demidovich contraction ($\alpha = d_{\min} - 1.0$) with zero relaxation of stability guarantees! $\blacksquare$
- **PO-B3 (Exponential Spatial Acceleration Upper Bound)**:
  - *Requirement*: Prove that Candidate B reduces the effective shortest-path nilpotency order from $k$ to $\lceil k/M \rceil$.
  - *Proof*: If shortest-path distance $d(s, v) = k$, then $P_M = P^M$ connects $s$ to $v$ in $\lceil k/M \rceil$ macro-steps. Specifically, the $(v, s)$ entry of $(P_M)^p = P^{M \cdot p}$ can be non-zero for $M \cdot p \ge k$, i.e., $p \ge \lceil k/M \rceil$.
    Thus, the earliest *potentially* non-zero term in the Neumann resolvent can appear at $p = \lceil k/M \rceil$ (governed by the graph support of $P^{M \cdot p}$, though feature weights or activation derivatives could render it zero); for all $p < \lceil k/M \rceil$, the term is identically zero by shortest-path nilpotency.
    The spatial influence upper bound is therefore:
    $$\|\mathcal{J}_{v, s}\|_2 \sim \mathcal{O}\left( \rho^{\lceil k/M \rceil} \right)$$
    For a 2-scale system ($M = 2$), hop 6 can potentially be reached in $p = 3$ macro-steps. The attenuation upper bound is $\rho^3 \approx 0.296$ instead of $\rho^6 \approx 0.088$, representing a $>3.3$-fold increase in the theoretical spatial transmission bound without changing $\alpha$ or $d_{\min}$. $\blacksquare$  
    *(Qualification: This spatial acceleration is an upper-bound property on structural graph support; it does not guarantee larger learned signal magnitudes or superior empirical predictions.)*

---

## 5. Candidate C: State-Independent Structural Readout Augmentation

### 5.1 Architectural Specification
Keep the continuous dynamical core completely unmodified:
$$\dot{Z} = \tanh(P Z B_1) B_2 - Z D + X, \quad d_{\min} \ge 1.5, \quad \alpha \ge 0.5$$
Augment the readout function with node-invariant structural representations:
$$\hat{y}(s, t) = \operatorname{MLP}\left( [Z^*(t); \; \ln(\|Z^*(t)\|_2 + 10^{-12}); \; \phi(s); \; \phi(t)] \right)$$
where $\phi(v) \in \mathbb{R}^k$ is a precomputed structural topological embedding of node $v$.

### 5.2 Proof Obligations
- **PO-C1 (Orthogonal Decoupling of Dynamics and Readout)**:
  - *Requirement*: Prove that augmenting the readout function $\operatorname{MLP}$ with $\phi(s), \phi(t)$ has zero effect on the logarithmic norm $\mu_2(J_f)$ of the dynamical core.
  - *Proof*: The vector field $\dot{Z} = f(Z, X)$ does not depend on $\phi$. The Jacobian $J_f(Z) = \frac{\partial f}{\partial Z}$ is identical to the baseline model. Thus $\mu_2(J_f) \le -\alpha$ is preserved unconditionally. $\blacksquare$
- **PO-C2 (Information Fairness and Zero Reachability Leakage)**:
  - *Requirement*: Prove that $\phi(v)$ does not encode directed reachability $s \rightsquigarrow t$ directly (which would render the dynamical ODE redundant and compromise experimental validity).
  - *Admissible Features*:
    1. Local degree statistics: $\operatorname{deg}_{\text{in}}(v), \operatorname{deg}_{\text{out}}(v)$.
    2. Directed PageRank score $\pi(v)$.
    3. Low-dimensional non-backtracking or Laplacian positional embeddings.
  - *Forbidden Features*:
    1. Global transitive closure matrix $\mathcal{T}_{uv} = \mathbb{I}(u \rightsquigarrow v)$.
    2. Exact shortest path matrices $D_{uv} = d(u, v)$.
    3. Pairwise reachability labels or heuristic reachability proxies.
  - *Verification Obligation*: Provide a unit test asserting that for any disjoint pair $(s, t)$, predicting reachability solely from $[\phi(s), \phi(t)]$ without $Z^*$ achieves AUC $\le 0.55$ on random DAGs/digraphs.
  - *Status*: **OPEN / UNMET IN CURRENT EXPERIMENTS**. While the implemented Candidate C model provides only scalar node degree counts ($\ln(1 + d_{\text{out}}(s)), \ln(1 + d_{\text{in}}(t))$) rather than explicit path matrices, node degrees in directed chains and layered DAGs naturally correlate with topological depth and reachability. The proposed feature-only AUC unit test was not executed. Consequently, Candidate C's performance is qualified as an exploratory observation with acknowledged structural correlation, rather than certified zero-leakage.

---

## 6. Candidate D: Curvature-Preserving Graph Rewiring (FoSR / Directed Expansion)

### 6.1 Architectural Specification
Modify the input graph topology $G = (V, E) \to \tilde{G} = (V, E \cup E_{\text{add}})$ using directed curvature rewiring (e.g., directed First-Order Spectral Rewiring) to reduce graph diameter and over-squashing bottlenecks.

### 6.2 Proof Obligations
- **PO-D1 (Bipartite Operator Norm Bound on Rewired Adjacency)**:
  - *Requirement*: Prove that the rewired normalized adjacency matrix $\tilde{P} = N(\tilde{G})^\top$ strictly satisfies $\|\tilde{P}\|_2 \le 1$.
  - *Proof*: By Proposition 3.1 of the authoritative paper, for *any* directed graph $\tilde{G}$ (regardless of edge additions), the bipartite normalization $N_{ij} = \tilde{C}_{ij} / \sqrt{\operatorname{deg}_{\text{out}}(i) \operatorname{deg}_{\text{in}}(j)}$ corresponds to the bi-normalized incidence matrix of a bipartite graph. The spectral norm of any row-stochastic or bi-normalized bipartite matrix is bounded by $1.0$. Hence $\|\tilde{P}\|_2 \le 1$ holds for any valid directed graph $\tilde{G}$. $\blacksquare$
- **PO-D2 (Ground-Truth Reachability Invariance)**:
  - *Requirement*: Edge additions $E_{\text{add}}$ must NOT alter the true reachability labels between evaluated test pairs $(s, t)$.
  - *Condition*: If $(s, t)$ is a negative query ($s \not\rightsquigarrow_G t$), then $s \not\rightsquigarrow_{\tilde{G}} t$.
  - *Consequence*: Arbitrary undirected rewiring or arbitrary directed edge addition is strictly prohibited, as adding an edge $(u, v)$ may create a new path from $s$ to $t$, corrupting the test ground truth.
  - *Admissible Rewiring*: Only rewiring that respects ground truth (or rewiring evaluated on transductive graphs where ground-truth labels are recomputed after rewiring) is valid.

---

## 7. Comparative Assessment of Candidates

| Candidate | Contraction Preserved? | Spatial Signal Improvement | Proof Complexity | Practical Feasibility |
| :--- | :---: | :---: | :---: | :---: |
| **A: Margin Tuning** | Yes ($\alpha > 0$ strictly) | Exponential ($\rho \to 1^-$) | Low (PO-A1 to A3) | High (Hyperparameter sweep) |
| **B: Multiscale Skip-Hop** | Yes ($\alpha \ge 0.5$ identical) | High ($\rho^k \to \rho^{\lceil k/2 \rceil}$) | Moderate (PO-B1 to B3) | High (Linear combination of walks) |
| **C: Structural Readout** | Yes (Core ODE unchanged) | Readout compensation | Low (PO-C1 to C2) | High (Feature concatenation) |
| **D: Graph Rewiring** | Yes ($\|\tilde{P}\|_2 \le 1$) | High (Shortens paths) | High (PO-D1 to D2) | Moderate (Label preservation constraint) |

---

## 8. Phase 1 Theory Scope and Gating Status

### Scope of this Gate
This gate certifies the **completion and auditing of the theory formulation, analytical sensitivity derivations, and architectural proof register only**. It does **not** signify that all candidate empirical obligations are met or closed. Specifically:
- **Closed Mathematical Proofs**: The proofs for Candidates A and B (PO-A1, PO-A2, PO-B1, PO-B2, PO-B3, PO-C1) are mathematically established under their stated assumptions.
- **Open / Unmet Obligations**: Obligation PO-C2 (feature-only AUC unit test to rule out structural correlation of degree features) was **not conducted** and remains **OPEN / UNMET**. Candidate C is consequently restricted to an exploratory architectural observation.

### Theory Document Audit Checklist
- [x] **Criterion 1 (Contraction vs. Influence Distinction)**: `THEORY_NOTE.md` formally establishes that trajectory contraction $\|Z_1(t) - Z_2(t)\| \le e^{-\alpha t} \|Z_1(0) - Z_2(0)\|$ governs temporal convergence, whereas spatial reachability is governed by the equilibrium Neumann operator $(I - \mathcal{K})^{-1}$.
- [x] **Criterion 2 (Mathematical Sensitivity Bounds)**: `SENSITIVITY_ANALYSIS.md` proves Proposition 1.1, deriving an analytical upper bound on equilibrium spatial sensitivity $\rho^k = \left(\frac{1}{1 + \alpha/\beta}\right)^k$ induced by shortest-path nilpotency, while framing readout thresholding and degree over-squashing as plausible structural hypotheses.
- [x] **Criterion 3 (Proof Obligations Formulated)**: `PROOF_OBLIGATIONS.md` establishes mathematical obligations for Candidates A, B, C, and D, with complete proofs for PO-A1, PO-B1, PO-B2, and PO-C1, and an explicit OPEN / UNMET designation for PO-C2.

**Theory Gate Status**: **THEORY FORMULATION & PROOF REGISTER AUDITED (Exploratory Working Document; Candidate C Retained as Unverified for Zero-Leakage)**
