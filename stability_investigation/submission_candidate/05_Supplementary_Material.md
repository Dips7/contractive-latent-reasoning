# Supplementary Material: Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics

**Manuscript Title**: Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures  
**Target Journal**: *Neurocomputing* (Elsevier)  
**Article Type**: Full Research Paper  
**Status**: Submission Candidate Supplementary Document  
**Date**: November 2026

---

## Table of Contents
- [Section S1: Mathematical Foundations and Extended Proofs](#section-s1-mathematical-foundations-and-extended-proofs)
  - [S1.1 Operator Spectral Norm Bound (Proof of Proposition 2.1)](#s11-operator-spectral-norm-bound-proof-of-proposition-21)
  - [S1.2 Demidovich Contraction of Continuous Latent Vector Field (Proof of Theorem 2.1)](#s12-demidovich-contraction-of-continuous-latent-vector-field-proof-of-theorem-21)
  - [S1.3 Derivation of Equilibrium Resolvent and Spatial Attenuation Bound (Proof of Proposition 3.1)](#s13-derivation-of-equilibrium-resolvent-and-spatial-attenuation-bound-proof-of-proposition-31)
  - [S1.4 Certified Stability of Multiscale Dynamics (Proof of Theorem 4.1)](#s14-certified-stability-of-multiscale-dynamics-proof-of-theorem-41)
  - [S1.5 Spatial Acceleration Upper Bound for Multiscale Dynamics (Proof of Theorem 4.2)](#s15-spatial-acceleration-upper-bound-for-multiscale-dynamics-proof-of-theorem-42)
- [Section S2: Complete Audited Experimental Loss Records](#section-s2-complete-audited-experimental-loss-records)
  - [S2.1 Experiment 1: Descriptive Damping Sweep Complete Loss Records](#s21-experiment-1-descriptive-damping-sweep-complete-loss-records)
  - [S2.2 Experiment 2: Architectural Comparison Complete Loss Records](#s22-experiment-2-architectural-comparison-complete-loss-records)
- [Section S3: Benchmark Dataset Specification, Split Protocol, and Automated Audit](#section-s3-benchmark-dataset-specification-split-protocol-and-automated-audit)
  - [S3.1 Digraph Family Specifications](#s31-digraph-family-specifications)
  - [S3.2 Multi-Instance Split and Sampling Protocol](#s32-multi-instance-split-and-sampling-protocol)
  - [S3.3 Distribution of Reachability Hops Across Graph Families](#s33-distribution-of-reachability-hops-across-graph-families)
  - [S3.4 Dataset Integrity Manifest and Cryptographic Hashes](#s34-dataset-integrity-manifest-and-cryptographic-hashes)
- [Section S4: Open Verification Obligations and Research Boundaries](#section-s4-open-verification-obligations-and-research-boundaries)
  - [S4.1 Status of Proof Obligation PO-C2 (Candidate C Feature-Only Baseline)](#s41-status-of-proof-obligation-po-c2-candidate-c-feature-only-baseline)
  - [S4.2 Protocol Limitations and Status of Citation-Network Experiments](#s42-protocol-limitations-and-status-of-citation-network-experiments)
  - [S4.3 Benchmark Ceiling Effects and Future Research Directions](#s43-benchmark-ceiling-effects-and-future-research-directions)

---

## Section S1: Mathematical Foundations and Extended Proofs

### S1.1 Operator Spectral Norm Bound (Proof of Proposition 2.1)

**Proposition 2.1**:
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be an arbitrary directed graph with $n = |\mathcal{V}|$ nodes and $m = |\mathcal{E}|$ directed edges. Let $C \in \{0, 1\}^{n \times n}$ denote its directed adjacency matrix ($C_{ij} = 1 \iff (i, j) \in \mathcal{E}$). With degree vectors $d_{\text{out}, i} = \max(1, \sum_j C_{ij})$ and $d_{\text{in}, j} = \max(1, \sum_i C_{ij})$, define the normalized message-passing operator:
$$N_{ij} \triangleq \frac{C_{ij}}{\sqrt{d_{\text{out}, i} d_{\text{in}, j}}}, \quad P \triangleq N^\top$$
Then the induced matrix 2-norm (spectral norm) satisfies:
$$\|P\|_2 = \|N\|_2 \le 1.0$$

*Proof*:
Recall the variational characterization of the matrix 2-norm:
$$\|N\|_2 = \sup_{u, v \in \mathbb{R}^n, \|u\|_2 = 1, \|v\|_2 = 1} |u^\top N v|$$
Expanding the bilinear form $|u^\top N v|$:
$$|u^\top N v| = \left| \sum_{i=1}^n \sum_{j=1}^n u_i N_{ij} v_j \right| = \left| \sum_{(i, j) \in \mathcal{E}} \frac{u_i v_j}{\sqrt{d_{\text{out}, i} d_{\text{in}, j}}} \right| \le \sum_{(i, j) \in \mathcal{E}} \frac{|u_i| |v_j|}{\sqrt{d_{\text{out}, i} d_{\text{in}, j}}}$$
We apply the Cauchy-Schwarz inequality over the index set of edges $(i, j) \in \mathcal{E}$:
$$\sum_{(i, j) \in \mathcal{E}} \left( \frac{|u_i|}{\sqrt{d_{\text{out}, i}}} \right) \left( \frac{|v_j|}{\sqrt{d_{\text{in}, j}}} \right) \le \sqrt{\sum_{(i, j) \in \mathcal{E}} \frac{|u_i|^2}{d_{\text{out}, i}}} \sqrt{\sum_{(i, j) \in \mathcal{E}} \frac{|v_j|^2}{d_{\text{in}, j}}}$$
Evaluating the first summation under the square root by grouping edges by their source node $i$:
$$\sum_{(i, j) \in \mathcal{E}} \frac{|u_i|^2}{d_{\text{out}, i}} = \sum_{i=1}^n \frac{|u_i|^2}{d_{\text{out}, i}} \sum_{j:(i, j) \in \mathcal{E}} 1 = \sum_{i=1}^n \frac{|u_i|^2}{d_{\text{out}, i}} d_{\text{out}, i} = \sum_{i=1}^n |u_i|^2 = \|u\|_2^2 = 1.0$$
Similarly, grouping the second summation by receiver node $j$:
$$\sum_{(i, j) \in \mathcal{E}} \frac{|v_j|^2}{d_{\text{in}, j}} = \sum_{j=1}^n \frac{|v_j|^2}{d_{\text{in}, j}} \sum_{i:(i, j) \in \mathcal{E}} 1 = \sum_{j=1}^n \frac{|v_j|^2}{d_{\text{in}, j}} d_{\text{in}, j} = \sum_{j=1}^n |v_j|^2 = \|v\|_2^2 = 1.0$$
Combining these two factors:
$$|u^\top N v| \le \sqrt{1.0} \cdot \sqrt{1.0} = 1.0$$
Taking the supremum over all unit vectors $u, v$ yields $\|N\|_2 \le 1.0$. Because the spectral norm of a matrix equals the spectral norm of its transpose, $\|P\|_2 = \|N^\top\|_2 = \|N\|_2 \le 1.0$. $\blacksquare$

---

### S1.2 Demidovich Contraction of Continuous Latent Vector Field (Proof of Theorem 2.1)

**Theorem 2.1**:
Consider the continuous latent dynamical system:
$$\dot{Z}(t) = \tanh(P Z(t) B_1) B_2 - Z(t) D + X$$
where $Z(t) \in \mathbb{R}^{n \times d}$, $\|B_1\|_2 \le 1.0$, $\|B_2\|_2 \le 1.0$, $D = \operatorname{diag}(d_1, \dots, d_d)$ with $d_j \ge d_{\min} > 1.0$, and $\|P\|_2 \le 1.0$.
In vectorized coordinates $z(t) = \operatorname{vec}(Z(t)) \in \mathbb{R}^{nd}$, the vector field is $\dot{z} = f(z) + x$ with:
$$f(z) = (B_2^\top \otimes I_n) \Phi\left( (B_1^\top \otimes P) z \right) - (D \otimes I_n) z$$
where $\Phi$ applies $\tanh(\cdot)$ component-wise. Then $f(z)$ is uniformly Demidovich contractive with margin:
$$\alpha \triangleq d_{\min} - 1.0 > 0$$

*Proof*:
The Jacobian $J_f(z) = \frac{\partial f}{\partial z}(z) \in \mathbb{R}^{nd \times nd}$ is computed using the Kronecker chain rule:
$$J_f(z) = (B_2^\top \otimes I_n) \mathcal{D}(z) (B_1^\top \otimes P) - (D \otimes I_n)$$
where $\mathcal{D}(z) = \operatorname{diag}(1 - \Phi((B_1^\top \otimes P) z)^2) \in \mathbb{R}^{nd \times nd}$.
Because the derivative of hyperbolic tangent satisfies $\frac{d}{du}\tanh(u) = 1 - \tanh^2(u) \in (0, 1]$ for all $u \in \mathbb{R}$, every diagonal entry of $\mathcal{D}(z)$ satisfies $0 < \mathcal{D}_{ii}(z) \le 1.0$, which implies $\|\mathcal{D}(z)\|_2 \le 1.0$.

The logarithmic norm (matrix measure) of $J_f(z)$ induced by the Euclidean 2-norm is defined as:
$$\mu_2(J_f(z)) \triangleq \lim_{h \to 0^+} \frac{\|I + h J_f(z)\|_2 - 1}{h} = \lambda_{\max}\left( \frac{J_f(z) + J_f(z)^\top}{2} \right)$$
Using the properties of the matrix measure:
1. **Subadditivity**: $\mu_2(A + B) \le \mu_2(A) + \mu_2(B)$.
2. **Norm Bounding**: $\mu_2(A) \le \|A\|_2$.
3. **Diagonal Decomposition**: For a symmetric negative-definite matrix $-(D \otimes I_n)$, its matrix measure equals its maximum eigenvalue: $\mu_2(-(D \otimes I_n)) = -\lambda_{\min}(D \otimes I_n) = -\min_j d_j = -d_{\min}$.

Applying these properties:
$$\mu_2(J_f(z)) \le \|(B_2^\top \otimes I_n) \mathcal{D}(z) (B_1^\top \otimes P)\|_2 + \mu_2(-(D \otimes I_n))$$
By submultiplicativity of the spectral norm and properties of the Kronecker product ($\|A \otimes B\|_2 = \|A\|_2 \|B\|_2$):
$$\|(B_2^\top \otimes I_n) \mathcal{D}(z) (B_1^\top \otimes P)\|_2 \le \|B_2^\top \otimes I_n\|_2 \|\mathcal{D}(z)\|_2 \|B_1^\top \otimes P\|_2$$
$$\le \|B_2\|_2 \cdot 1.0 \cdot \|B_1\|_2 \|P\|_2 \le (1.0)(1.0)(1.0)(1.0) = 1.0$$
Therefore:
$$\mu_2(J_f(z)) \le 1.0 - d_{\min} = -(d_{\min} - 1.0) = -\alpha < 0$$
Since $\mu_2(J_f(z)) \le -\alpha < 0$ uniformly for all $z \in \mathbb{R}^{nd}$, the system is strictly Demidovich contractive. By the Demidovich Contraction Theorem (Desoer & Vidyasagar, 1975; Sontag, 1998):
1. **Global Convergence**: Any two trajectories $z_1(t), z_2(t)$ driven by the same input $x(t)$ satisfy:
   $$\frac{d}{dt} \|z_1(t) - z_2(t)\|_2 \le \mu_2(J_f) \|z_1(t) - z_2(t)\|_2 \le -\alpha \|z_1(t) - z_2(t)\|_2$$
   Integrating gives $\|z_1(t) - z_2(t)\|_2 \le e^{-\alpha t} \|z_1(0) - z_2(0)\|_2$.
2. **Unique Fixed Point**: Under constant input $x$, setting $\dot{z} = 0$ yields the implicit equilibrium equation $f(z^*) + x = 0$. Since the Jacobian $J_f(z)$ has eigenvalues with real part $\le -\alpha < 0$, it is globally invertible, guaranteeing a unique fixed point $z^* = \lim_{t \to \infty} z(t)$.
3. **Input-to-State Dissipation**: For two static inputs $x_1, x_2$, the difference in equilibria satisfies $\|z^*(x_1) - z^*(x_2)\|_2 \le \frac{1}{\alpha} \|x_1 - x_2\|_2$. $\blacksquare$

---

### S1.3 Derivation of Equilibrium Resolvent and Spatial Attenuation Bound (Proof of Proposition 3.1)

**Proposition 3.1**:
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a directed graph. Let $d_{\mathcal{G}}(s, v) = k \ge 1$ denote the directed shortest-path distance from source node $s$ to target node $v$. Under elementwise activations $\tanh(\cdot)$ and the edge-support condition $P_{iu} \neq 0 \implies (u, i) \in \mathcal{E}$:
1. **Shortest-Path Nilpotency**: The $m$-th order sensitivity block vanishes identically for all $m < k$:
   $$J_{v, s}^{(m)} = 0 \quad \forall m < k$$
2. **Spatial Sensitivity Upper Bound**: The operator norm of the equilibrium sensitivity block $[J_{\text{eq}}]_{v, s} = \frac{\partial z_v^*}{\partial x_s} \in \mathbb{R}^{d \times d}$ is bounded above by:
   $$\left\| \frac{\partial z^*_v}{\partial x_s} \right\|_2 \le \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$
   where $\beta = \|B_1\|_2 \|B_2\|_2 \le 1.0$ and $\alpha = d_{\min} - \beta > 0$.

*Proof*:

#### Part 1: Node-Level Equilibrium Differentiation and Recurrence
The equilibrium state $Z^* \in \mathbb{R}^{n \times d}$ satisfies:
$$Z^*(X) D - \tanh(P Z^*(X) B_1) B_2 = X$$
Expressing this row-wise, let $z_i^* \in \mathbb{R}^{1 \times d}$ denote row $i$ of $Z^*$ (the equilibrium state of node $i$) and $x_i \in \mathbb{R}^{1 \times d}$ denote row $i$ of $X$ (the external input at node $i$). The equilibrium equation for node $i$ is:
$$z_i^* D - \tanh(a_i) B_2 = x_i$$
where $a_i \in \mathbb{R}^{1 \times d}$ is the incoming pre-activation aggregate:
$$a_i \triangleq \sum_{u=1}^n P_{i u} z_u^* B_1$$
Because $\tanh(\cdot)$ is applied elementwise (node-separable across nodes), the partial derivative $\frac{\partial \tanh(a_i)}{\partial a_u}$ is identically zero for all $u \neq i$. For node $i$, the internal Jacobian with respect to $a_i$ is the diagonal matrix:
$$\Lambda_i(z^*) \triangleq \operatorname{diag}(1 - \tanh^2(a_i)) \in \mathbb{R}^{d \times d}$$
with diagonal entries in $(0, 1]$, so $\|\Lambda_i(z^*)\|_2 \le 1.0$.

Differentiating node $i$'s equilibrium equation with respect to an input perturbation vector $x_s \in \mathbb{R}^{1 \times d}$ at source node $s$:
Define the block sensitivity Jacobian:
$$J_{i, s} \triangleq \frac{\partial z_i^*}{\partial x_s} \in \mathbb{R}^{d \times d}$$
Applying the multivariable chain rule:
$$\frac{\partial}{\partial x_s} \left[ z_i^* D \right] - \frac{\partial}{\partial x_s} \left[ \tanh(a_i) B_2 \right] = \frac{\partial x_i}{\partial x_s}$$
$$J_{i, s} D - \left( \frac{\partial a_i}{\partial x_s} \right) \Lambda_i(z^*) B_2 = \delta_{i, s} I_d$$
Substituting $\frac{\partial a_i}{\partial x_s} = \sum_{u=1}^n P_{i u} \frac{\partial z_u^*}{\partial x_s} B_1 = \sum_{u=1}^n P_{i u} J_{u, s} B_1$:
$$J_{i, s} D - \sum_{u=1}^n P_{i u} J_{u, s} B_1 \Lambda_i(z^*) B_2 = \delta_{i, s} I_d$$
Since $D \ge d_{\min} I_d > 0$, $D$ is strictly invertible. Multiplying on the right by $D^{-1}$:
$$J_{i, s} - \sum_{u=1}^n P_{i u} J_{u, s} \left( B_1 \Lambda_i(z^*) B_2 D^{-1} \right) = \delta_{i, s} D^{-1}$$
Define the internal node-localized channel-mixing operator:
$$C_i(z^*) \triangleq B_1 \Lambda_i(z^*) B_2 D^{-1} \in \mathbb{R}^{d \times d}$$
Its spectral norm is bounded by:
$$\|C_i(z^*)\|_2 \le \|B_1\|_2 \|\Lambda_i(z^*)\|_2 \|B_2\|_2 \|D^{-1}\|_2 \le \|B_1\|_2 (1.0) \|B_2\|_2 \left(\frac{1}{d_{\min}}\right) = \frac{\beta}{d_{\min}} = \rho < 1.0$$
where $\rho \triangleq \frac{\beta}{d_{\min}} = \frac{\beta}{\beta + \alpha} < 1.0$ because $\alpha = d_{\min} - \beta > 0$.

The node sensitivity equation becomes:
$$J_{i, s} = \delta_{i, s} D^{-1} + \sum_{u=1}^n P_{i u} J_{u, s} C_i(z^*)$$

#### Part 2: Neumann Series Unrolling and Directed Walk Induction
We unroll this linear recurrence as a formal Neumann power series:
$$J_{i, s} = \sum_{m=0}^\infty J_{i, s}^{(m)}$$
where:
- Base term ($m = 0$):
  $$J_{i, s}^{(0)} = \delta_{i, s} D^{-1}$$
- Recursive terms ($m \ge 1$):
  $$J_{i, s}^{(m)} = \sum_{u=1}^n P_{i u} J_{u, s}^{(m-1)} C_i(z^*)$$

We prove by mathematical induction on $m$ that for any target node $v$ and source node $s$, $J_{v, s}^{(m)}$ expands as a sum over directed walks of length exactly $m$ from $s$ to $v$.

*Inductive Hypothesis*: For all $m \ge 1$:
$$J_{v, s}^{(m)} = \sum_{u_1, u_2, \dots, u_{m-1} \in \mathcal{V}} \left( \prod_{l=1}^m P_{u_l, u_{l-1}} \right) D^{-1} \prod_{l=1}^m C_{u_l}(z^*)$$
with $u_0 = s$ and $u_m = v$.

*Base Case ($m = 1$)*:
$$J_{v, s}^{(1)} = \sum_{u=1}^n P_{v u} J_{u, s}^{(0)} C_v(z^*) = \sum_{u=1}^n P_{v u} (\delta_{u, s} D^{-1}) C_v(z^*) = P_{v s} D^{-1} C_v(z^*)$$
This matches the formula for $m=1$ with $u_0 = s$ and $u_1 = v$.

*Inductive Step*: Assume the formula holds for $m-1$. Then:
$$J_{v, s}^{(m)} = \sum_{u=1}^n P_{v u} J_{u, s}^{(m-1)} C_v(z^*)$$
Substituting the inductive hypothesis for $J_{u, s}^{(m-1)}$:
$$J_{v, s}^{(m)} = \sum_{u=1}^n P_{v u} \left[ \sum_{u_1, \dots, u_{m-2}} \left( \prod_{l=1}^{m-1} P_{u_l, u_{l-1}} \right) D^{-1} \prod_{l=1}^{m-1} C_{u_l}(z^*) \right] C_v(z^*)$$
Setting $u_{m-1} = u$ and $u_m = v$:
$$J_{v, s}^{(m)} = \sum_{u_1, \dots, u_{m-1}} \left( P_{v, u_{m-1}} \prod_{l=1}^{m-1} P_{u_l, u_{l-1}} \right) D^{-1} \left( \prod_{l=1}^{m-1} C_{u_l}(z^*) \right) C_v(z^*)$$
$$= \sum_{u_1, \dots, u_{m-1}} \left( \prod_{l=1}^m P_{u_l, u_{l-1}} \right) D^{-1} \prod_{l=1}^m C_{u_l}(z^*)$$
which completes the induction.

Now, consider the scalar graph weight product:
$$\prod_{l=1}^m P_{u_l, u_{l-1}} = P_{v, u_{m-1}} P_{u_{m-1}, u_{m-2}} \cdots P_{u_2, u_1} P_{u_1, s}$$
Under the stated edge support condition, $P_{u_l, u_{l-1}} = N_{u_{l-1}, u_l} = \frac{C_{u_{l-1}, u_l}}{\sqrt{d_{\text{out}} d_{\text{in}}}} \neq 0$ only if $(u_{l-1}, u_l) \in \mathcal{E}$.
Thus, the product $\prod_{l=1}^m P_{u_l, u_{l-1}}$ is non-zero if and only if the sequence:
$$(s = u_0 \to u_1 \to u_2 \to \cdots \to u_{m-1} \to u_m = v)$$
forms a valid directed path/walk of length exactly $m$ from $s$ to $v$ in $\mathcal{G}$.

By assumption, the directed shortest-path distance is $d_{\mathcal{G}}(s, v) = k \ge 1$.
By definition of shortest path, there exists **no directed walk of length $m$ from $s$ to $v$ in $\mathcal{G}$ for any $m < k$**.
Consequently, for all $m < k$ and all intermediate node sequences $(u_1, \dots, u_{m-1})$, the product $\prod_{l=1}^m P_{u_l, u_{l-1}}$ is identically zero.
Therefore:
$$J_{v, s}^{(m)} = 0 \quad \forall m < k$$
This establishes **shortest-path nilpotency**. $\blacksquare$

#### Part 3: Tail Series Summation and Exact Prefactor Cancellation
Because $J_{v, s}^{(m)} = 0$ for all $m < k$, the full sensitivity block $[J_{\text{eq}}]_{v, s}$ is given by the tail of the Neumann series starting at $m = k$:
$$[J_{\text{eq}}]_{v, s} = \sum_{m=k}^\infty J_{v, s}^{(m)}$$
In vectorized form, $[J_{\text{eq}}]_{v, s} = \sum_{m=k}^\infty [\mathcal{T}^m]_{v, s} (D^{-1} \otimes I_n)$, where $\|\mathcal{T}\|_2 \le \rho < 1.0$.
Taking the spectral norm of this tail sum:
$$\left\| [J_{\text{eq}}]_{v, s} \right\|_2 \le \sum_{m=k}^\infty \left\| [\mathcal{T}^m]_{v, s} (D^{-1} \otimes I_n) \right\|_2 \le \|D^{-1}\|_2 \sum_{m=k}^\infty \|\mathcal{T}\|_2^m \le \frac{1}{d_{\min}} \sum_{m=k}^\infty \rho^m$$
Evaluating the geometric tail series:
$$\sum_{m=k}^\infty \rho^m = \rho^k \sum_{j=0}^\infty \rho^j = \frac{\rho^k}{1 - \rho}$$
Multiplying by $\frac{1}{d_{\min}}$:
$$\left\| [J_{\text{eq}}]_{v, s} \right\|_2 \le \frac{1}{d_{\min}} \left( \frac{\rho^k}{1 - \rho} \right) = \left( \frac{1}{d_{\min}(1 - \rho)} \right) \rho^k$$
We now evaluate the tail prefactor $\frac{1}{d_{\min}(1 - \rho)}$. Substituting $\rho = \frac{\beta}{d_{\min}}$:
$$1 - \rho = 1 - \frac{\beta}{d_{\min}} = \frac{d_{\min} - \beta}{d_{\min}}$$
$$d_{\min}(1 - \rho) = d_{\min} \left( \frac{d_{\min} - \beta}{d_{\min}} \right) = d_{\min} - \beta = \alpha$$
Thus, $d_{\min}$ in the numerator cancels identically with $d_{\min}$ in the denominator, yielding the exact identity:
$$\frac{1}{d_{\min}(1 - \rho)} = \frac{1}{\alpha}$$
Substituting this into the tail inequality gives:
$$\left\| \frac{\partial z^*_v}{\partial x_s} \right\|_2 \le \frac{1}{\alpha} \rho^k = \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$
which completes the proof of Proposition 3.1. $\blacksquare$

---

### S1.4 Certified Stability of Multiscale Dynamics (Proof of Theorem 4.1)

**Theorem 4.1**:
Consider the multiscale skip-hop continuous dynamical system:
$$\dot{Z}(t) = \gamma_1 \tanh(P Z(t) B_{1, 1}) B_{1, 2} + \gamma_2 \tanh(P^2 Z(t) B_{2, 1}) B_{2, 2} - Z(t) D + X$$
where $\gamma_1, \gamma_2 \ge 0$ with $\gamma_1 + \gamma_2 = 1.0$, $\|B_{l, 1}\|_2 \le 1.0$, $\|B_{l, 2}\|_2 \le 1.0$ for $l \in \{1, 2\}$, and $D = \operatorname{diag}(d_1, \dots, d_d)$ with $d_j \ge d_{\min} > 1.0$.
Then the system satisfies uniform Demidovich contraction with margin:
$$\alpha = d_{\min} - 1.0 > 0$$

*Proof*:
First, by submultiplicativity of induced matrix norms:
$$\|P^2\|_2 \le \|P\|_2^2 \le (1.0)^2 = 1.0$$
In vectorized coordinates $z = \operatorname{vec}(Z)$, the Jacobian $J(z)$ is:
$$J(z) = \gamma_1 (B_{1, 2}^\top \otimes P) \mathcal{D}_1(z) (B_{1, 1}^\top \otimes I_n) + \gamma_2 (B_{2, 2}^\top \otimes P^2) \mathcal{D}_2(z) (B_{2, 1}^\top \otimes I_n) - (D \otimes I_n)$$
where $\mathcal{D}_1(z), \mathcal{D}_2(z) \in \mathbb{R}^{nd \times nd}$ are diagonal matrices with diagonal entries in $(0, 1]$, so $\|\mathcal{D}_1(z)\|_2 \le 1.0$ and $\|\mathcal{D}_2(z)\|_2 \le 1.0$.

Applying the subadditive property of the logarithmic norm $\mu_2(\cdot)$:
$$\mu_2(J(z)) \le \gamma_1 \mu_2\left( (B_{1, 2}^\top \otimes P) \mathcal{D}_1(z) (B_{1, 1}^\top \otimes I_n) \right) + \gamma_2 \mu_2\left( (B_{2, 2}^\top \otimes P^2) \mathcal{D}_2(z) (B_{2, 1}^\top \otimes I_n) \right) + \mu_2(-(D \otimes I_n))$$
Using $\mu_2(A) \le \|A\|_2$ and $\|A \otimes B\|_2 = \|A\|_2 \|B\|_2$:
$$\mu_2\left( (B_{1, 2}^\top \otimes P) \mathcal{D}_1(z) (B_{1, 1}^\top \otimes I_n) \right) \le \|B_{1, 2}\|_2 \|P\|_2 \|\mathcal{D}_1(z)\|_2 \|B_{1, 1}\|_2 \le (1.0)(1.0)(1.0)(1.0) = 1.0$$
$$\mu_2\left( (B_{2, 2}^\top \otimes P^2) \mathcal{D}_2(z) (B_{2, 1}^\top \otimes I_n) \right) \le \|B_{2, 2}\|_2 \|P^2\|_2 \|\mathcal{D}_2(z)\|_2 \|B_{2, 1}\|_2 \le (1.0)(1.0)(1.0)(1.0) = 1.0$$
Furthermore, $\mu_2(-(D \otimes I_n)) = -d_{\min}$. Combining these terms:
$$\mu_2(J(z)) \le \gamma_1(1.0) + \gamma_2(1.0) - d_{\min} = (\gamma_1 + \gamma_2) - d_{\min}$$
Since $\gamma_1 + \gamma_2 = 1.0$:
$$\mu_2(J(z)) \le 1.0 - d_{\min} = -(d_{\min} - 1.0) = -\alpha < 0$$
This guarantees that Candidate B satisfies uniform Demidovich contraction under identical damping bounds as the single-scale baseline. $\blacksquare$

---

### S1.5 Spatial Acceleration Upper Bound for Multiscale Dynamics (Proof of Theorem 4.2)

**Theorem 4.2**:
In the multiscale architecture Candidate B, the spatial sensitivity upper bound is accelerated to $\mathcal{O}(\rho^{\lceil k/2 \rceil})$, reducing the shortest-path nilpotency threshold from $k$ to $\lceil k/2 \rceil$.

*Proof*:
In Candidate B, the propagation operator includes $P_2 = P^2$.
Let $s$ and $v$ be two nodes separated by shortest-path distance $d_{\mathcal{G}}(s, v) = k$.
In a single step of the 2-hop operator $P^2$, the $(v, s)$ entry is:
$$[P^2]_{v, s} = \sum_{u=1}^n P_{v u} P_{u s}$$
This entry is non-zero if there exists an intermediate node $u$ such that $s \to u \to v$ is a 2-hop walk in $\mathcal{G}$.
More generally, in a walk consisting of $m$ applications of the multiscale operator, the maximum path distance traversable in $m$ steps is $2m$ (by taking $P^2$ at every step).
Thus, the $(v, s)$ entry of $(P^2)^m = P^{2m}$ can be non-zero whenever $2m \ge k$, which is equivalent to:
$$m \ge \left\lceil \frac{k}{2} \right\rceil$$
Conversely, for all $m < \lceil k/2 \rceil$, the maximum reachable walk length using any combination of $P$ and $P^2$ in $m$ steps is $2m < k$. By definition of shortest path, no path of length $< k$ connects $s$ to $v$. Therefore, all walk products of order $m < \lceil k/2 \rceil$ vanish identically:
$$J_{v, s}^{(m)} = 0 \quad \forall m < \left\lceil \frac{k}{2} \right\rceil$$
Summing the geometric tail series from $m = \lceil k/2 \rceil$ to $\infty$:
$$\left\| [J_{\text{eq}}]_{v, s} \right\|_2 \le \frac{1}{\alpha} \rho^{\lceil k/2 \rceil}$$
*(Qualification: This spatial acceleration is an upper-bound property on structural graph support; it proves that the earliest potentially non-zero term in the series can appear at $\lceil k/2 \rceil$ steps rather than $k$ steps. It does not guarantee larger learned signals or superior task performance.)* $\blacksquare$

---

## Section S2: Complete Audited Experimental Loss Records

### S2.1 Experiment 1: Descriptive Damping Sweep Complete Loss Records
All 72 models were trained for 25 epochs using Adam ($lr = 0.005$) across fixed PRNG seeds (42, 101, 777). Checkpoints and test losses were evaluated on held-out test graph instances and cryptographically audited in `damping_test_loss_certificate.json`.

#### Table S1: Descriptive Damping Parameter Sweep Results Across 72 Models

| Family | Nominal Configuration | Seed 42 Test Loss | Seed 101 Test Loss | Seed 777 Test Loss | Mean Test Loss $\pm$ Sample SD | Mean Val Loss $\pm$ Sample SD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **chain_ladder** | `ContinuousDamped_dmin_0.50` | 0.001036 | 0.000429 | 0.000817 | 0.000761 $\pm$ 0.000307 | 0.000675 $\pm$ 0.000329 |
| | `ContinuousDamped_dmin_0.80` | 0.001014 | 0.000528 | 0.000932 | 0.000825 $\pm$ 0.000260 | 0.000762 $\pm$ 0.000270 |
| | `ContinuousDamped_dmin_1.05` | 0.001178 | 0.000595 | 0.001015 | 0.000929 $\pm$ 0.000300 | 0.000858 $\pm$ 0.000290 |
| | `ContinuousDamped_dmin_1.20` | 0.001219 | 0.000628 | 0.001063 | 0.000970 $\pm$ 0.000306 | 0.000888 $\pm$ 0.000283 |
| | `ContinuousDamped_dmin_1.50` | 0.001306 | 0.000687 | 0.001161 | 0.001051 $\pm$ 0.000324 | 0.000943 $\pm$ 0.000278 |
| | `ContinuousDamped_dmin_2.00` | 0.001532 | 0.000805 | 0.001333 | 0.001223 $\pm$ 0.000376 | 0.001057 $\pm$ 0.000290 |
| | `ContinuousUndamped` ($D=0$) | 0.000136 | 0.000156 | 0.000346 | 0.000213 $\pm$ 0.000116 | 0.000056 $\pm$ 0.000025 |
| | `DiscreteGNN_L6` | 0.000172 | 0.000216 | 0.000313 | 0.000234 $\pm$ 0.000072 | 0.000032 $\pm$ 0.000010 |
| **bottleneck** | `ContinuousDamped_dmin_0.50` | 0.000913 | 0.000680 | 0.001202 | 0.000932 $\pm$ 0.000261 | 0.000780 $\pm$ 0.000235 |
| | `ContinuousDamped_dmin_0.80` | 0.001084 | 0.000771 | 0.001368 | 0.001074 $\pm$ 0.000298 | 0.000874 $\pm$ 0.000239 |
| | `ContinuousDamped_dmin_1.05` | 0.001147 | 0.000859 | 0.001500 | 0.001169 $\pm$ 0.000321 | 0.000936 $\pm$ 0.000240 |
| | `ContinuousDamped_dmin_1.20` | 0.001387 | 0.000920 | 0.001579 | 0.001295 $\pm$ 0.000339 | 0.001009 $\pm$ 0.000252 |
| | `ContinuousDamped_dmin_1.50` | 0.001311 | 0.001059 | 0.001728 | 0.001366 $\pm$ 0.000338 | 0.001060 $\pm$ 0.000233 |
| | `ContinuousDamped_dmin_2.00` | 0.001584 | 0.001339 | 0.002002 | 0.001642 $\pm$ 0.000335 | 0.001236 $\pm$ 0.000217 |
| | `ContinuousUndamped` ($D=0$) | 0.000297 | 0.000558 | 0.000938 | 0.000598 $\pm$ 0.000322 | 0.000326 $\pm$ 0.000179 |
| | `DiscreteGNN_L6` | 0.000055 | 0.000098 | 0.000076 | 0.000076 $\pm$ 0.000022 | 0.000039 $\pm$ 0.000013 |
| **random_dag** | `ContinuousDamped_dmin_0.50` | 0.004456 | 0.002453 | 0.008571 | 0.005160 $\pm$ 0.003119 | 0.004727 $\pm$ 0.002835 |
| | `ContinuousDamped_dmin_0.80` | 0.004469 | 0.002700 | 0.011257 | 0.006142 $\pm$ 0.004518 | 0.005620 $\pm$ 0.004162 |
| | `ContinuousDamped_dmin_1.05` | 0.004538 | 0.002928 | 0.013258 | 0.006908 $\pm$ 0.005558 | 0.006293 $\pm$ 0.005146 |
| | `ContinuousDamped_dmin_1.20` | 0.004632 | 0.003078 | 0.014328 | 0.007346 $\pm$ 0.006096 | 0.006671 $\pm$ 0.005653 |
| | `ContinuousDamped_dmin_1.50` | 0.004862 | 0.003401 | 0.016150 | 0.008138 $\pm$ 0.006977 | 0.007334 $\pm$ 0.006468 |
| | `ContinuousDamped_dmin_2.00` | 0.005340 | 0.003973 | 0.018434 | 0.009249 $\pm$ 0.007984 | 0.008216 $\pm$ 0.007382 |
| | `ContinuousUndamped` ($D=0$) | 0.001531 | 0.001401 | 0.003110 | 0.002014 $\pm$ 0.000952 | 0.001288 $\pm$ 0.000629 |
| | `DiscreteGNN_L6` | 0.000346 | 0.000545 | 0.000483 | 0.000458 $\pm$ 0.000102 | 0.000286 $\pm$ 0.000067 |

*(Note: In all 72 configurations, test accuracy, specificity, and recall evaluated to 100.0% due to the synthetic benchmark ceiling.)*

---

### S2.2 Experiment 2: Architectural Comparison Complete Loss Records
Evaluated across 45 trained models on held-out test graph instances and cryptographically audited in `test_loss_evaluation_certificate.json`.

#### Table S2: Architectural Comparison Per-Seed Held-Out Test Cross-Entropy Losses

| Family | Architecture | Seed 42 Test Loss | Seed 101 Test Loss | Seed 777 Test Loss | Mean Test Loss $\pm$ Sample SD | $\Delta$ vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **chain_ladder** | `Baseline_Damped_1.5` | 0.001306 | 0.000687 | 0.001161 | 0.001051 $\pm$ 0.000324 | Baseline |
| | `Candidate_A_Tuned_1.05` | 0.001178 | 0.000595 | 0.001015 | 0.000929 $\pm$ 0.000300 | -11.6% |
| | `Candidate_B_Multiscale_1.5` | 0.000516 | 0.000507 | 0.000459 | **0.000494 $\pm$ 0.000031** | **-53.0%** |
| | `Candidate_C_Structural_1.5`* | 0.000572 | 0.000279 | 0.000638 | **0.000496 $\pm$ 0.000191** | **-52.8%** |
| | `Baseline_Discrete_GNN6` | 0.000172 | 0.000216 | 0.000313 | 0.000234 $\pm$ 0.000072 | -77.7% |
| **bottleneck** | `Baseline_Damped_1.5` | 0.001311 | 0.001059 | 0.001728 | 0.001366 $\pm$ 0.000338 | Baseline |
| | `Candidate_A_Tuned_1.05` | 0.001147 | 0.000859 | 0.001500 | 0.001169 $\pm$ 0.000321 | -14.4% |
| | `Candidate_B_Multiscale_1.5` | 0.000880 | 0.000505 | 0.000760 | **0.000715 $\pm$ 0.000191** | **-47.7%** |
| | `Candidate_C_Structural_1.5`* | 0.001309 | 0.000874 | 0.001929 | 0.001371 $\pm$ 0.000530 | +0.4% |
| | `Baseline_Discrete_GNN6` | 0.000055 | 0.000098 | 0.000076 | 0.000076 $\pm$ 0.000022 | -94.4% |
| **random_dag** | `Baseline_Damped_1.5` | 0.004862 | 0.003401 | 0.016150 | 0.008138 $\pm$ 0.006977 | Baseline |
| | `Candidate_A_Tuned_1.05` | 0.004538 | 0.002928 | 0.013258 | 0.006908 $\pm$ 0.005558 | -15.1% |
| | `Candidate_B_Multiscale_1.5` | 0.002455 | 0.002209 | 0.002840 | **0.002501 $\pm$ 0.000318** | **-69.3%** |
| | `Candidate_C_Structural_1.5`* | 0.001068 | 0.000604 | 0.001066 | **0.000913 $\pm$ 0.000267** | **-88.8%** |
| | `Baseline_Discrete_GNN6` | 0.000346 | 0.000545 | 0.000483 | 0.000458 $\pm$ 0.000102 | -94.4% |

*(Note: Candidate C results are subject to open proof obligation PO-C2, as detailed in Section S4.1.)*

---

## Section S3: Benchmark Dataset Specification, Split Protocol, and Automated Audit

### S3.1 Digraph Family Specifications
1. **Chain/Ladder (`chain_ladder`)**: $N = 40$ nodes. Formed by a directed backbone chain $i \to i+1$ ($i=0, \dots, 38$) with forward shortcut edges $i \to i+2$ sampled with probability $p=0.3$. Average degree $\bar{d} \approx 2.3$. Evaluates linear information propagation with minimal degree heterogeneity.
2. **Bottleneck (`bottleneck`)**: $N = 48$ nodes. Composed of two dense clusters $A$ ($N_A = 24$) and $B$ ($N_B = 24$), each with internal forward edges ($p=0.25$). All inter-cluster paths must pass strictly through a single directed bridge edge $u_A \to v_B$. Evaluates information transfer through a severe structural bottleneck.
3. **Layered Random DAG (`random_dag`)**: $N = 42$ nodes arranged in 7 layers of 6 nodes each. Directed edges are allowed only from layer $l$ to layer $l+1$ (probability $p=0.25$). Evaluates feedforward multi-path DAG dynamics.

### S3.2 Multi-Instance Split and Sampling Protocol
- **Partitioning**: Each family contains 12 independent graph instances: 5 Train instances, 2 Validation instances, and 5 Test instances. Total across all three families: 36 graph instances.
- **Query Allocation**: Within each graph instance, query pairs $(s, t)$ are generated across hop buckets $k \in \{1, 2, 3, 4, 5, 6\}$ (for positive pairs) alongside balanced negative pairs (disconnected pairs verified via BFS).
- **Zero Replacement Sampling**: All queries within each hop bucket are sampled strictly without replacement.
- **Split Isolation**: Because train, validation, and test sets are defined on distinct graph instances, **query pair overlap across splits is exactly 0.0%**.

### S3.3 Distribution of Reachability Hops Across Graph Families

#### Table S3: Reconciled Query Pair Distribution by Hop Distance Across All 36 Instances

| Graph Family | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Hop 5 | Hop 6 | Negative Pairs | Total Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chain_ladder` (12 inst.) | 240 | 240 | 240 | 240 | 240 | 240 | 1,440 | 2,880 |
| `bottleneck` (12 inst.) | 240 | 240 | 240 | 240 | 240 | 240 | 1,440 | 2,880 |
| `random_dag` (12 inst.) | 240 | 240 | 240 | 14 | 1 | 0 | 735 | 1,470 |
| **Total (36 instances)** | **720** | **720** | **720** | **494** | **481** | **480** | **3,615** | **7,230** |

*Critical Structural Note on `random_dag`*: Because dense random connections between adjacent layers create numerous short paths, the structural diameter of `random_dag` is short. Across all 12 instances, only 14 queries exist at hop 4, 1 query at hop 5, and 0 queries at hop 6. Consequently, empirical conclusions regarding reachability at hops 4, 5, and 6 are supported strictly by `chain_ladder` and `bottleneck`.

### S3.4 Dataset Integrity Manifest and Cryptographic Hashes
The synthetic benchmark suite was audited using `stability_investigation/scripts/audit_benchmark_data.py`. All 7,230 queries were certified for:
- 100% agreement with BFS-computed ground truth.
- Zero duplicate queries across instances.
- Operator spectral norm condition $\|P\|_2 \le 1.000000119$ (passing within float32 numerical tolerance of $10^{-6}$).

The dataset integrity manifest is persisted at:
`stability_investigation/data/synthetic/benchmark_manifest.json`  
**SHA-256 Digest**: `8d5b02c591989e49a168a662727dd51ede02f6778dde643aca7c3d584e316342`

---

## Section S4: Open Verification Obligations and Research Boundaries

### S4.1 Status of Proof Obligation PO-C2 (Candidate C Feature-Only Baseline)
- **Proof Obligation PO-C2**: Requires training and evaluating an isolated baseline classifier given *only* the node degree features $(\ln(1+d_{\text{out}}(s)), \ln(1+d_{\text{in}}(t)))$ without any ODE latent representations, to determine whether node degrees alone can predict reachability on these synthetic digraphs.
- **Current Status**: **OPEN / UNMET**. This test was not performed.
- **Epistemic Implication**: On regular graph families like chains and layered DAGs, node degrees are non-trivially correlated with topological position (e.g., initial and terminal nodes have characteristic degree profiles). Therefore, Candidate C's test loss improvements cannot be attributed to superior continuous graph dynamics, and degree features cannot be asserted to be free from structural leakage.

### S4.2 Protocol Limitations and Status of Citation-Network Experiments
In our earlier preliminary experiments evaluating contractive graph dynamics on transductive citation networks (CiteSeer and PubMed across path lengths 1–6; archived in `external_validation/EXTERNAL_VALIDATION_REPORT.md`):
- Contractive models were observed to collapse to 0.00% positive recall on reachability queries at hops 3–6.
- A 6-layer discrete message-passing baseline on CiteSeer maintained ~33.3% recall at hops 5–6 (with 66.7% negative specificity).
- **Documented Protocol Limitations**:
  1. *Query Overlaps and Duplicates*: Query generation in the extended Phase 2 evaluation allowed overlap between training, validation, and test query pairs, and internal duplicate queries were present.
  2. *Weight Initialization*: Labeled seeds did not correspond to independently randomized initialization weights.
- **Classification**: These citation-network findings are classified strictly as **exploratory observations**, not confirmatory evidence. They motivate the research questions addressed in this paper but do not serve as rigorous benchmark evidence.

### S4.3 Benchmark Ceiling Effects and Future Research Directions
Under the audited multi-instance training protocol on small bounded-degree digraphs, all evaluated models (including continuous damped ODEs, undamped ODEs, and discrete GNNs) achieved **100.0% test accuracy, 100.0% specificity, and 100.0% recall across all evaluated queries**.
- This demonstrates that continuous contraction does not mathematically preclude learning multi-hop reachability up to 6 hops on small directed graphs.
- However, because test classification metrics are saturated at ceiling, this benchmark does not discriminate models by classification recall. Differences between architectures are reflected strictly in continuous cross-entropy loss.
- Future work must evaluate non-saturated relational reasoning tasks (e.g. continuous regression, graded path reasoning) and scale-free graphs with heavy-tailed degree distributions ($d \ge 50$) to evaluate degree over-squashing under contractive dynamics.
