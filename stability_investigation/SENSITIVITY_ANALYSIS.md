# Mathematical Sensitivity Analysis: Upper Bounds on Equilibrium Graph Influence

**Date**: 2026-11-16 (Revised)  
**Document Status**: PHASE 1 THEORY DELIVERABLE (REVISED WITH BOUNDARY QUALIFICATIONS)  
**Governing Branch**: `stability_investigation/`  
**Parent Framework**: Contractive Continuous Graph Neural Dynamics  

---

## 1. Executive Summary & Epistemic Boundaries

In contractive continuous-time graph neural dynamics, empirical evaluations on transductive citation networks have observed near-zero classification recall at multi-hop distances ($\ge 3$ hops). 

The goal of this document is to derive an **analytical upper bound on linear equilibrium sensitivity** in continuous graph vector fields, explicitly distinguish trajectory contraction from spatial sensitivity, and define the exact scope and limitations of the mathematical theory:

1. **Analytical Sensitivity Bound (Proposition 1.1)**: From the exact Neumann series identity for the linearized equilibrium resolvent $(I_{nd} - \mathcal{T})^{-1} = \sum_{m=0}^\infty \mathcal{T}^m$ in $\mathbb{R}^{nd}$, we derive an analytical upper bound on the $(v, s)$ spatial block of the equilibrium input-sensitivity Jacobian:
   $$\left\| \frac{\partial z^*_v}{\partial x_s} \right\|_2 \le \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$
   where $k = d_{\mathcal{G}}(s, v)$ is the directed shortest-path distance, $\beta = \|B_1\|_2 \|B_2\|_2 \le 1.0$, and $\alpha = d_{\min} - \beta > 0$ is the Demidovich contraction margin.
2. **Crucial Epistemic Qualifications**:
   - **Upper Bound, Not Equality**: The geometric rate $\rho^k = (\frac{\beta}{\beta + \alpha})^k$ is an *upper bound* on the maximum possible linear sensitivity at equilibrium. It does not establish that the actual learned representations or downstream classification recall must decay at this rate.
   - **Equilibrium vs. Finite Horizon**: The sensitivity bound is derived for the infinite-time equilibrium state $z^* = \lim_{t \to \infty} z(t)$. In practice, models use finite-time numerical integration ($T = 4.0$ with explicit Runge-Kutta steps), meaning transient dynamics and truncation effects also operate.
   - **Hypothesis Status of the "Root Cause"**: While this sensitivity bound provides a plausible mathematical mechanism that can upper-bound signal transfer, the empirical failure on citation graphs cannot be attributed to contraction alone without controlled empirical isolation of degree over-squashing, training optimization, and readout thresholding.

---

## 2. Derivation of the Four Sensitivity Regimes

### 2.1 Continuous Latent Dynamics
Recall the continuous vector field for node representations $Z(t) \in \mathbb{R}^{n \times d}$:
$$\dot{Z}(t) = \tanh(P Z(t) B_1) B_2 - Z(t) D + X$$
where $P = N^\top \in \mathbb{R}^{n \times n}$ with $\|P\|_2 \le 1.0$, $\|B_1\|_2, \|B_2\|_2 \le 1.0$, $D = \operatorname{diag}(d_1, \dots, d_d)$ with $d_j \ge d_{\min} > 1.0$, and $X \in \mathbb{R}^{n \times d}$ is an external input matrix.

Vectorizing $z(t) = \operatorname{vec}(Z(t)) \in \mathbb{R}^{nd}$ and $x = \operatorname{vec}(X) \in \mathbb{R}^{nd}$ using the identity $\operatorname{vec}(A Y B) = (B^\top \otimes A) \operatorname{vec}(Y)$:
$$\dot{z}(t) = (B_2^\top \otimes I_n) \Phi\left( (B_1^\top \otimes P) z(t) \right) - (D \otimes I_n) z(t) + x \triangleq f(z(t)) + x$$
where $\Phi: \mathbb{R}^{nd} \to \mathbb{R}^{nd}$ applies $\tanh(\cdot)$ component-wise.

---

### 2.2 Regime 1: Trajectory Contraction (Temporal Stability)
The Jacobian of $f(z)$ is:
$$J_f(z) = (B_2^\top \otimes I_n) \mathcal{D}(z) (B_1^\top \otimes P) - (D \otimes I_n)$$
where $\mathcal{D}(z) = \operatorname{diag}(1 - \Phi((B_1^\top \otimes P) z)^2) \in \mathbb{R}^{nd \times nd}$ with diagonal entries in $[0, 1]$.
Under the Euclidean norm, the logarithmic norm satisfies:
$$\mu_2(J_f(z)) \le \|B_2\|_2 \|\mathcal{D}(z)\|_2 \|B_1\|_2 \|P\|_2 - d_{\min} \le \beta - d_{\min} = -\alpha < 0$$
where $\alpha = d_{\min} - \beta > 0$.

*Conclusion*: Trajectory contraction ensures that for any two initial states $z_1(0), z_2(0)$ under the same input $x$:
$$\|z_1(t) - z_2(t)\|_2 \le e^{-\alpha t} \|z_1(0) - z_2(0)\|_2$$
This is a **purely temporal property** governing how rapidly states forget their initial conditions and converge to the unique equilibrium $z^*$. It does not govern spatial influence across the graph.

---

### 2.3 Regime 2: Global Equilibrium Input Sensitivity
At equilibrium, $\dot{z} = 0 \implies (D \otimes I_n) z^* - (B_2^\top \otimes I_n) \Phi((B_1^\top \otimes P) z^*) = x$.
Let $x_1, x_2 \in \mathbb{R}^{nd}$ be two inputs, and $z^*(x_1), z^*(x_2)$ their corresponding equilibria:
$$\langle z^*(x_1) - z^*(x_2), f(z^*(x_1)) - f(z^*(x_2)) \rangle \le -\alpha \|z^*(x_1) - z^*(x_2)\|_2^2$$
Since $f(z^*(x_1)) - f(z^*(x_2)) = -(x_1 - x_2)$, by Cauchy-Schwarz:
$$\alpha \|z^*(x_1) - z^*(x_2)\|_2^2 \le \langle z^*(x_1) - z^*(x_2), x_1 - x_2 \rangle \le \|z^*(x_1) - z^*(x_2)\|_2 \|x_1 - x_2\|_2$$
$$\implies \|z^*(x_1) - z^*(x_2)\|_2 \le \frac{1}{\alpha} \|x_1 - x_2\|_2$$

*Conclusion*: The equilibrium mapping $x \mapsto z^*(x)$ is globally Lipschitz continuous with constant $L_{\text{input}} \le 1/\alpha$. Note that a larger contraction margin $\alpha$ reduces the overall input sensitivity bound.

---

### 2.4 Regime 3: Spatial Graph Influence & Shortest-Path Nilpotency
We now derive the spatial Jacobian resolving how a localized input at source node $s$ affects target node $v$ located at shortest-path distance $d_{\mathcal{G}}(s, v) = k$.

Differentiating the equilibrium equation $(D \otimes I_n) z^* - (B_2^\top \otimes I_n) \Phi((B_1^\top \otimes P) z^*) = x$ with respect to $x$:
Let $J_{\text{eq}} \triangleq \frac{\partial z^*}{\partial x} \in \mathbb{R}^{nd \times nd}$. Then:
$$\left[ (D \otimes I_n) - (B_2^\top \otimes I_n) \mathcal{D}(z^*) (B_1^\top \otimes P) \right] J_{\text{eq}} = I_{nd}$$
Since $D \ge d_{\min} I_d > 0$, the block $(D \otimes I_n)$ is strictly invertible with $(D \otimes I_n)^{-1} = (D^{-1} \otimes I_n)$. Left-multiplying by $(D^{-1} \otimes I_n)$:
$$\left[ I_{nd} - (D^{-1} B_2^\top \otimes I_n) \mathcal{D}(z^*) (B_1^\top \otimes P) \right] J_{\text{eq}} = (D^{-1} \otimes I_n)$$
Define the linear operator $\mathcal{T} \in \mathbb{R}^{nd \times nd}$:
$$\mathcal{T} \triangleq (D^{-1} B_2^\top \otimes I_n) \mathcal{D}(z^*) (B_1^\top \otimes P)$$
The operator norm of $\mathcal{T}$ is bounded by:
$$\|\mathcal{T}\|_2 \le \|D^{-1} B_2^\top \otimes I_n\|_2 \|\mathcal{D}(z^*)\|_2 \|B_1^\top \otimes P\|_2 \le \frac{1}{d_{\min}} \|B_2\|_2 (1) \|B_1\|_2 \|P\|_2 \le \frac{\beta}{d_{\min}}$$
Since $d_{\min} = \beta + \alpha$ with $\alpha > 0$:
$$\|\mathcal{T}\|_2 \le \frac{\beta}{\beta + \alpha} \triangleq \rho < 1.0$$
Because $\|\mathcal{T}\|_2 \le \rho < 1.0$, the matrix $(I_{nd} - \mathcal{T})$ is invertible via its **Neumann series**:
$$J_{\text{eq}} = \sum_{m=0}^\infty \mathcal{T}^m (D^{-1} \otimes I_n)$$

Now consider the spatial block $[J_{\text{eq}}]_{v, s} \in \mathbb{R}^{d \times d}$ corresponding to source node $s$ and target node $v$:

**Proposition 1.1 (Shortest-Path Nilpotency and Spatial Sensitivity Upper Bound)**:
*Let $\mathcal{G}$ be a directed graph, and let $d_{\mathcal{G}}(s, v) = k \ge 1$ denote the directed shortest-path distance from source node $s$ to target node $v$. Assuming that the activation function is elementwise (node-separable) and that $P$'s non-zero entries follow the directed edges of $\mathcal{G}$ ($P_{i u} \neq 0 \implies (u, i) \in \mathcal{E}$), then:*
1. *Shortest-Path Nilpotency: For all $m < k$, the $(v, s)$ node-block of the $m$-th order sensitivity term is identically zero:*
   $$J_{v, s}^{(m)} = 0 \quad \forall m < k$$
2. *Spatial Influence Upper Bound: The operator norm of the spatial influence $[J_{\text{eq}}]_{v, s} \triangleq \frac{\partial z^*_v}{\partial x_s} \in \mathbb{R}^{d \times d}$ is bounded above by:*
   $$\|[J_{\text{eq}}]_{v, s}\|_2 \le \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$

*Proof*:
1. *Shortest-Path Nilpotency via Node-Level Block Recurrence*:  
   To resolve the node-level sparsity of the sensitivity operator without ambiguity from vectorization, we differentiate the equilibrium condition directly at the node level. In matrix form:
   $$Z^*(X) D - \tanh(P Z^*(X) B_1) B_2 = X$$
   For each node $i \in \{1, \dots, n\}$, let $z_i^* \in \mathbb{R}^{1 \times d}$ denote row $i$ of $Z^*$, and $x_i \in \mathbb{R}^{1 \times d}$ denote row $i$ of $X$. Then:
   $$z_i^* D - \tanh(a_i) B_2 = x_i, \quad \text{where } a_i \triangleq \sum_{u=1}^n P_{i u} z_u^* B_1 \in \mathbb{R}^{1 \times d}$$
   Because the activation $\tanh$ is applied elementwise (node-separable), the Jacobian of $\tanh(a_i)$ with respect to its pre-activation is the diagonal matrix $\Lambda_i(z^*) \triangleq \operatorname{diag}(1 - \tanh^2(a_i)) \in \mathbb{R}^{d \times d}$, acting purely within the $d$-dimensional feature space of node $i$ (with $\frac{\partial \tanh(a_i)}{\partial a_u} = 0$ for all $u \neq i$).

   Differentiating node $i$'s equilibrium equation with respect to an input perturbation at source node $x_s$:
   Let $J_{i, s} \triangleq \frac{\partial z_i^*}{\partial x_s} \in \mathbb{R}^{d \times d}$. By the chain rule:
   $$J_{i, s} D - \left( \sum_{u=1}^n P_{i u} J_{u, s} B_1 \right) \Lambda_i(z^*) B_2 = \delta_{i, s} I_d$$
   Multiplying on the right by $D^{-1}$:
   $$J_{i, s} - \sum_{u=1}^n P_{i u} J_{u, s} C_i(z^*) = \delta_{i, s} D^{-1}$$
   where $C_i(z^*) \triangleq B_1 \Lambda_i(z^*) B_2 D^{-1} \in \mathbb{R}^{d \times d}$ is an internal channel-mixing operator localized strictly to node $i$, with operator norm $\|C_i(z^*)\|_2 \le \|B_1\|_2 \|\Lambda_i\|_2 \|B_2\|_2 \|D^{-1}\|_2 \le \frac{\beta}{d_{\min}} = \rho < 1.0$.

   Unrolling this linear recurrence as a formal Neumann power series $J_{i, s} = \sum_{m=0}^\infty J_{i, s}^{(m)}$:
   - At base step $m = 0$: $J_{i, s}^{(0)} = \delta_{i, s} D^{-1}$.
   - At step $m \ge 1$: $J_{i, s}^{(m)} = \sum_{u=1}^n P_{i u} J_{u, s}^{(m-1)} C_i(z^*)$.

   By induction, for target node $v$ and source node $s$, the $m$-th order term expands explicitly over node sequences $(u_0, u_1, \dots, u_m)$ with $u_0 = s$ and $u_m = v$:
   $$J_{v, s}^{(m)} = \sum_{u_1, \dots, u_{m-1}} \left( \prod_{l=1}^m P_{u_l, u_{l-1}} \right) D^{-1} \prod_{l=1}^m C_{u_l}(z^*)$$
   Crucially, graph propagation is governed entirely by the scalar product of adjacency entries:
   $$\prod_{l=1}^m P_{u_l, u_{l-1}} = P_{v, u_{m-1}} \cdots P_{u_2, u_1} P_{u_1, s}$$
   This product is non-zero if and only if the sequence $(s = u_0, u_1, \dots, u_m = v)$ constitutes a valid directed walk of length $m$ from $s$ to $v$ in $\mathcal{G}$.
   If the directed shortest-path distance $d_{\mathcal{G}}(s, v) = k$, then by definition no directed walk of length $m < k$ exists in $\mathcal{G}$. Therefore, every walk product $\prod_{l=1}^m P_{u_l, u_{l-1}}$ is identically zero, which rigorously establishes:
   $$J_{v, s}^{(m)} = 0 \quad \forall m < k$$
2. For $m \ge k$, the operator norm of each power is bounded by $\|\mathcal{T}^m\|_2 \le \|\mathcal{T}\|_2^m \le \rho^m$, where the convergence condition $\rho = \beta / d_{\min} = \beta / (\beta + \alpha) < 1.0$ is guaranteed by $\alpha > 0$. Taking the operator norm of the tail sum of the Neumann series from $m = k$ to $\infty$:
   $$\|[J_{\text{eq}}]_{v, s}\|_2 = \left\| \sum_{m=k}^\infty [\mathcal{T}^m]_{v, s} (D^{-1} \otimes I_n) \right\|_2 \le \|D^{-1}\|_2 \sum_{m=k}^\infty \|\mathcal{T}^m\|_2 \le \frac{1}{d_{\min}} \sum_{m=k}^\infty \rho^m$$
   Evaluating the geometric tail sum:
   $$\sum_{m=k}^\infty \rho^m = \rho^k \sum_{j=0}^\infty \rho^j = \frac{\rho^k}{1 - \rho}$$
   Multiplying by $\|D^{-1}\|_2 \le 1/d_{\min}$:
   $$\|[J_{\text{eq}}]_{v, s}\|_2 \le \frac{1}{d_{\min}} \left( \frac{\rho^k}{1 - \rho} \right) = \left( \frac{1}{d_{\min}(1 - \rho)} \right) \rho^k$$
   We explicitly evaluate the tail prefactor $\frac{1}{d_{\min}(1 - \rho)}$ by substituting $\rho = \beta / d_{\min}$:
   $$1 - \rho = 1 - \frac{\beta}{d_{\min}} = \frac{d_{\min} - \beta}{d_{\min}}$$
   $$d_{\min}(1 - \rho) = d_{\min} \left( \frac{d_{\min} - \beta}{d_{\min}} \right) = d_{\min} - \beta = \alpha$$
   Therefore, the factor $d_{\min}$ in $\|D^{-1}\|_2$ cancels identically with the denominator of the geometric series sum, producing the exact tail prefactor:
   $$\frac{1}{d_{\min}(1 - \rho)} = \frac{1}{d_{\min} - \beta} = \frac{1}{\alpha}$$
   Substituting this identity into the tail inequality yields:
   $$\|[J_{\text{eq}}]_{v, s}\|_2 \le \frac{1}{\alpha} \rho^k = \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$
   *(Note on the single-term leading bound vs. full-tail bound: For the isolated leading term at $m = k$, the bound is $\|[\mathcal{T}^k]_{v, s} (D^{-1} \otimes I_n)\|_2 \le \frac{1}{d_{\min}} \rho^k$. Because $d_{\min} = \beta + \alpha \ge \alpha > 0$, the leading-term prefactor satisfies $\frac{1}{d_{\min}} \le \frac{1}{\alpha}$, meaning $\frac{1}{\alpha} \rho^k$ is a valid, slightly looser upper bound for the single $m=k$ step alone. Summing all infinite higher-order path contributions $m \ge k$ scales this leading bound by the geometric accumulation factor $\frac{1}{1 - \rho} = \frac{d_{\min}}{\alpha}$, which cancels $d_{\min}$ and renders $\frac{1}{\alpha} \rho^k$ the rigorous upper bound for the entire equilibrium series tail.)* $\blacksquare$

---

### 2.5 Interpretation and Limits of the Upper Bound

#### What the Bound Establishes:
- Proposition 1.1 proves that the *maximum possible* linear sensitivity of the equilibrium state at distance $k$ decays at least as fast as $\rho^k = (\frac{\beta}{\beta + \alpha})^k$.
- For nominal parameters ($\beta = 1.0, d_{\min} = 1.50 \implies \alpha = 0.50$), $\rho \le 2/3 \approx 0.667$.
- This sets an explicit analytical upper bound on equilibrium signal transmission that tightens with larger contraction margins $\alpha$.

#### What the Bound Does NOT Establish:
1. **Not a Proof of Vanishing Accuracy**: An upper bound on sensitivity does not prove that classification accuracy must decay or collapse. If the classification decision requires only an $\epsilon$-small perturbation, a network could theoretically classify correctly even under low sensitivity.
2. **Equilibrium vs. Finite-Time Discretization**: Proposition 1.1 is an asymptotic equilibrium result ($t \to \infty$). Models integrated over finite horizon $T=4.0$ with 16 RK4 steps may be influenced by transient dynamics, solver discretization errors, or insufficient integration time.
3. **Hypothesis Status of Empirical Confounds**:
   - On citation graphs (Cora, CiteSeer, PubMed), the empirical drop to 0.00% recall at hops 3–6 is correlated with high maximum degrees ($d > 50-100$). The product of degree normalizations $\prod \frac{1}{\sqrt{d_{\text{out}} d_{\text{in}}}}$ could compound with $\rho^k$.
   - Furthermore, the readout head $\ln(\|Z_t\|_2 + 10^{-12})$ maps states below a threshold to large negative values.
   - However, **these interactions remain hypotheses**. The sensitivity bound does not prove that this interaction is the sole or primary cause of empirical failure on citation graphs.
