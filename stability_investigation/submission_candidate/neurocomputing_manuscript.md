# Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures

**Target Journal**: *Neurocomputing* (Elsevier)  
**Article Type**: Full Research Paper  
**Status**: Anonymized Manuscript for Peer Review  
**Date**: November 2026  

---

## Abstract

Continuous-depth graph neural ODEs offer guaranteed dynamical stability under Demidovich contraction. However, prior exploratory evaluations on transductive citation networks (CiteSeer and PubMed across path lengths 1–6)—subject to protocol limitations including query overlaps and unisolated initializations—showed contractive models collapsing to 0.00% positive reachability recall beyond path length 2, while a discrete baseline on CiteSeer maintained non-zero recall. We investigate the mechanisms and empirical properties governing long-range signal propagation in contractive graph vector fields. First, separating temporal contraction from spatial sensitivity, we derive an equilibrium sensitivity upper bound via an exact linearized Neumann resolvent identity $(I_{nd} - \mathcal{T})^{-1} = \sum_{m=0}^\infty \mathcal{T}^m$. We prove that the $(v, s)$ spatial block is bounded above by $\|\frac{\partial z^*_v}{\partial x_s}\|_2 \le \frac{1}{\alpha} \left(\frac{\beta}{\beta + \alpha}\right)^k$ via shortest-path nilpotency under elementwise activations. We emphasize that this formula establishes an asymptotic upper bound rather than a tight rate, a proof of vanishing classification accuracy, or an established failure cause on complex graphs. Second, on an audited synthetic benchmark suite of 36 graph instances and 7,230 query pairs across chains, bottlenecks, and acyclic digraphs with strict split controls, a descriptive parameter sweep across 72 models shows continuous contractive models achieving 100.0% recall across evaluated queries (with 6-hop queries supported on chains and bottlenecks), while seed-averaged losses scale monotonically with nominal damping without establishing causal mechanisms. Finally, we introduce a certified Multiscale Skip-Hop Contractive Architecture ($P, P^2$) that preserves Demidovich stability while reducing held-out test cross-entropy loss by 47.7% to 69.3% on these ceiling-saturated benchmarks.

**Keywords**: Continuous-time Graph Neural Networks; Neural Ordinary Differential Equations; Demidovich Contraction; Dynamical Stability; Long-Range Graph Reachability; Over-Squashing.

---

## 1. Introduction

Continuous-time graph neural networks formulate representation learning as the integration of a parameterized dynamical system over a continuous depth horizon $t \in [0, T]$ [1, 5]:
$$\frac{d Z(t)}{dt} = F(Z(t), \mathcal{G}, X), \quad Z(0) = Z_0$$
where $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denotes a directed graph, $X$ is an external input matrix, and $Z(t) \in \mathbb{R}^{n \times d}$ represents the continuous latent node states. By replacing discrete stacked layers with continuous vector fields, these models bridge discrete message passing with the rich mathematical theory of dynamical systems, numerical ordinary differential equation (ODE) solvers, and equilibrium computation [1, 4, 8].

A primary theoretical appeal of continuous graph architectures is the ability to enforce rigorous **dynamical stability guarantees**. By constraining the vector field $F$ to satisfy a negative one-sided Lipschitz condition—known in control theory as **Demidovich contraction** [2, 3, 10]—the dynamical flow contracts uniformly in the state space. Demidovich contraction guarantees three crucial mathematical properties:
1. **Exponential Convergence**: All trajectories converge exponentially to a common trajectory regardless of initialization.
2. **Existence and Uniqueness of a Globally Attractive Equilibrium**: Under static inputs $X$, the dynamical system possesses a unique, globally exponentially attractive equilibrium state $Z^*(X)$ (guaranteed by Demidovich's theorem [2, 10] or the Banach fixed-point theorem applied to the implicit equilibrium map).
3. **Input-to-State Dissipation**: Bounded input perturbations dissipate at an exponential rate governed by the contraction margin $\alpha > 0$.

### The Long-Range Reachability Problem
Despite these compelling theoretical properties, in our earlier exploratory experiments evaluating contractive graph dynamics on transductive directed citation networks (specifically CiteSeer and PubMed; archived in the project repository under `external_validation/EXTERNAL_VALIDATION_REPORT.md`), we observed a striking empirical limitation: while continuous contractive models achieved 100.0% negative specificity (correctly predicting disconnected pairs as unreachable), their positive classification recall collapsed to 0.00% on multi-hop reachability queries at path lengths of 3 to 6 hops (with recall concentrated at hop 1: 74.6%–92.1% and trace hop 2: 0.7%–6.1%). In contrast, a 6-layer discrete message-passing baseline [8] evaluated on CiteSeer under that same setup maintained non-zero positive recall across all 6 hops (~33.3% at hops 5 and 6), though accompanied by significant false-positive over-prediction (66.7% mean negative specificity). Crucially, as documented in the archived validation report, those earlier citation experiments were subject to known protocol limitations—including cross-split query overlaps, internal duplicate queries, and unisolated initialization weights—classifying them as exploratory observations rather than confirmatory evidence. It remained unestablished whether the observed multi-hop recall drop was caused by continuous contraction, or was an artifact of network topology (such as degree over-squashing in hub nodes), readout conditioning, or benchmark protocol.

This empirical discrepancy has led to an active research question: **Does enforcing continuous-time contraction inherently limit long-range relational information propagation?** If stability requires dissipative damping, must continuous graph models sacrifice the ability to reason over long graph distances?

### Core Contributions & Epistemic Scope
In this paper, we address this question through formal sensitivity analysis, audited multi-instance synthetic benchmarks, and stability-preserving architectural designs:

1. **Analytical Sensitivity Upper Bound (Proposition 3.1)**: We derive the spatial sensitivity operator of contractive graph equilibria via a Neumann series expansion in $\mathbb{R}^{nd}$. We prove that while temporal state contraction $\|Z_1(t) - Z_2(t)\|_F \le e^{-\alpha t} \|Z_1(0) - Z_2(0)\|_F$ is governed by $\alpha = d_{\min} - \beta$ [2, 3], the spatial graph sensitivity across $k$ shortest-path hops is bounded above by $\|\frac{\partial z^*_v}{\partial x_s}\|_2 \le \frac{1}{\alpha} \left(\frac{\beta}{\beta + \alpha}\right)^k$. This bound is an analytical consequence of **shortest-path nilpotency** in directed graphs under elementwise activations. We clearly distinguish this upper bound on linearized equilibrium sensitivity from actual learned representation magnitudes or classification performance.
2. **Analysis of Topological and Readout Hypotheses**: We examine how the theoretical spatial attenuation bound could interact with graph structure. We discuss the hypothesis that on power-law graphs, degree over-squashing ($\prod \frac{1}{\sqrt{\text{deg}_{\text{out}}(u) \text{deg}_{\text{in}}(v)}}$) [6, 7] combined with contraction damping and non-linear readout thresholding ($\ln(\|Z_t\|_2 + \epsilon)$) could explain the empirical hop cutoff. We treat this mechanism as a plausible explanatory hypothesis requiring dedicated future testing on scale-free graphs.
3. **Audited Multi-Instance Benchmark Suite**: To evaluate continuous models under simplified, known graph structures with strict sampling and split controls, we generate an audited synthetic benchmark suite comprising 36 graph instances and 7,230 query pairs across directed chains, bottlenecks, and layered DAGs. We formulate an automated audit script certifying zero replacement sampling, zero internal duplicates, zero split leakage, and 100% BFS-verified ground truth. We explicitly disclose the structural sparsity of long-hop queries in layered DAGs, noting that 6-hop conclusions are supported strictly by chains and bottlenecks.
4. **Descriptive Damping Sweep**: In a descriptive parameter sweep across 72 trained models sweeping nominal damping settings $d_{\min} \in [0.50, 2.00]$ under fixed training budgets and PRNG seeds, seed-averaged validation and held-out test cross-entropy losses scale monotonically with nominal damping settings under the evaluated protocol (while individual run checkpoints exhibit minor run-to-run optimization variance). This sweep establishes an empirical association under this protocol rather than a causal law. Crucially, on the evaluated small bounded-degree graphs, contractive models achieve **100.0% recall across all evaluated queries**, showing that under this protocol, continuous contraction does not preclude learning reachability up to 6 hops on chains and bottlenecks.
5. **Certified Multiscale Skip-Hop Architecture**: We propose a multiscale continuous vector field integrating $P$ and $P^2$. We prove that it satisfies the exact Demidovich contraction condition under identical damping bounds ($\mu_2(J) \le 1 - d_{\min} = -\alpha < 0$) while accelerating the spatial influence upper bound from $\mathcal{O}(\rho^k)$ to $\mathcal{O}(\rho^{\lceil k/2 \rceil})$. Across all graph families, the multiscale architecture achieves 47.7% to 69.3% lower held-out test cross-entropy loss relative to the single-scale contractive baseline, with all per-run losses, checkpoint hashes, and dataset hashes fully audited. We explicitly note that classification accuracy is at ceiling across all evaluated models on this benchmark.

---

## 2. Mathematical Foundations of Contractive Graph Latent Dynamics

### 2.1 Graph Operator and Bipartite Normalization
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a directed graph with $n = |\mathcal{V}|$ nodes and $m = |\mathcal{E}|$ directed edges. The directed adjacency matrix $C \in \{0, 1\}^{n \times n}$ has entries $C_{ij} = 1$ if and only if directed edge $(i, j) \in \mathcal{E}$ exists (information flowing from sender $i$ to receiver $j$).

Define the out-degree vector $d_{\text{out}} \in \mathbb{R}^n$ and in-degree vector $d_{\text{in}} \in \mathbb{R}^n$ by:
$$d_{\text{out}, i} = \max\left(1, \sum_{j=1}^n C_{ij}\right), \quad d_{\text{in}, j} = \max\left(1, \sum_{i=1}^n C_{ij}\right)$$
We define the bipartite normalized message-passing operator $N \in \mathbb{R}^{n \times n}$ and its transpose $P \in \mathbb{R}^{n \times n}$ by:
$$N_{ij} \triangleq \frac{C_{ij}}{\sqrt{d_{\text{out}, i} \cdot d_{\text{in}, j}}}, \quad P \triangleq N^\top$$
For any latent node matrix $Z \in \mathbb{R}^{n \times d}$, forward message passing along directed edge $i \to j$ is given by:
$$(P Z)_j = \sum_{i=1}^n N_{ij} Z_i = \sum_{i=1}^n \frac{C_{ij}}{\sqrt{d_{\text{out}, i} d_{\text{in}, j}}} Z_i$$

**Proposition 2.1 (Operator Spectral Norm Bound)**:
*For any directed graph $\mathcal{G}$, the normalized propagation operator $P$ satisfies:*
$$\|P\|_2 = \|N\|_2 \le 1.0$$
*Proof*: The matrix $N$ can be embedded as the off-diagonal block of the adjacency matrix of a bipartite graph $\mathcal{B} = (\mathcal{V}_{\text{out}} \cup \mathcal{V}_{\text{in}}, \mathcal{E}_{\mathcal{B}})$ with degree-normalized weights $D_{\text{out}}^{-1/2} C D_{\text{in}}^{-1/2}$. By the Cauchy-Schwarz inequality, for any vectors $u, v \in \mathbb{R}^n$:
$$|u^\top N v| \le \sum_{(i, j) \in \mathcal{E}} \frac{|u_i| |v_j|}{\sqrt{d_{\text{out}, i} d_{\text{in}, j}}} \le \sqrt{\sum_{i=1}^n |u_i|^2 \sum_{j:(i,j) \in \mathcal{E}} \frac{1}{d_{\text{out}, i}}} \sqrt{\sum_{j=1}^n |v_j|^2 \sum_{i:(i,j) \in \mathcal{E}} \frac{1}{d_{\text{in}, j}}} = \|u\|_2 \|v\|_2$$
Thus $\|N\|_2 = \sup_{\|u\|_2, \|v\|_2 \le 1} |u^\top N v| \le 1.0$. Since $\|P\|_2 = \|N^\top\|_2 = \|N\|_2$, the operator norm is bounded by $1.0$. $\blacksquare$

*(Note: In numerical computations using standard 32-bit floating-point precision, computed singular values of $P$ can evaluate to $1.000000119$, which satisfies the bound within standard numerical tolerance $\le 1.0 + 10^{-6}$.)*

### 2.2 Continuous Latent Vector Field
The continuous latent dynamics are governed by the non-linear matrix ODE [1, 5]:
$$\dot{Z}(t) = \tanh(P Z(t) B_1) B_2 - Z(t) D + X$$
where:
- $Z(t) \in \mathbb{R}^{n \times d}$ is the time-dependent latent node state matrix.
- $B_1, B_2 \in \mathbb{R}^{d \times d}$ are weight matrices constrained via SVD projection to satisfy $\|B_1\|_2 \le 1.0$ and $\|B_2\|_2 \le 1.0$.
- $D = \operatorname{diag}(d_1, \dots, d_d) \in \mathbb{R}^{d \times d}$ is a diagonal damping matrix with $d_j \ge d_{\min} > 0$.
- $X \in \mathbb{R}^{n \times d}$ is an external input excitation matrix.

Vectorizing $z(t) = \operatorname{vec}(Z(t)) \in \mathbb{R}^{nd}$ using $\operatorname{vec}(A Y B) = (B^\top \otimes A) \operatorname{vec}(Y)$, the system is written as $\dot{z} = f(z) + x$, where:
$$f(z) = (B_2^\top \otimes I_n) \Phi\left((B_1^\top \otimes P) z\right) - (D \otimes I_n) z$$
and $\Phi: \mathbb{R}^{nd} \to \mathbb{R}^{nd}$ applies $\tanh(\cdot)$ component-wise.

### 2.3 Demidovich Contraction Theorem
Let $\mu_2(J) = \lambda_{\max}\left(\frac{J + J^\top}{2}\right)$ denote the logarithmic norm (matrix measure) of Jacobian $J \in \mathbb{R}^{nd \times nd}$ induced by the standard Euclidean norm.

**Theorem 2.1 (Demidovich Contraction of Continuous Graph Dynamics)** [2, 3, 10]:
*If $\|B_1\|_2 \le 1.0$, $\|B_2\|_2 \le 1.0$, and $d_{\min} > 1.0$, then the vector field $f(z)$ is uniformly Demidovich contractive with contraction margin:*
$$\alpha \triangleq d_{\min} - 1.0 > 0$$
*Specifically, the Jacobian $J_f(z) = \frac{\partial f}{\partial z}(z)$ satisfies $\mu_2(J_f(z)) \le -\alpha < 0$ for all $z \in \mathbb{R}^{nd}$, guaranteeing:*
1. *Global Exponential Trajectory Convergence:*
   $$\|z_1(t) - z_2(t)\|_2 \le e^{-\alpha t} \|z_1(0) - z_2(0)\|_2 \quad \forall t \ge 0$$
2. *Unique Equilibrium Existence:*
   *Under constant input $x$, there exists a unique fixed point $z^* \in \mathbb{R}^{nd}$ satisfying $f(z^*) + x = 0$, and $\lim_{t \to \infty} z(t) = z^*$.*
3. *Input-to-State Dissipation:*
   $$\|z^*(x_1) - z^*(x_2)\|_2 \le \frac{1}{\alpha} \|x_1 - x_2\|_2$$

*Proof*: The Jacobian of $f(z)$ is:
$$J_f(z) = (B_2^\top \otimes I_n) \mathcal{D}(z) (B_1^\top \otimes P) - (D \otimes I_n)$$
where $\mathcal{D}(z) = \operatorname{diag}(1 - \Phi((B_1^\top \otimes P) z)^2)$ is diagonal with entries in $[0, 1]$.
Using the submultiplicative and subadditive properties of the logarithmic norm:
$$\mu_2(J_f(z)) \le \|(B_2^\top \otimes I_n) \mathcal{D}(z) (B_1^\top \otimes P)\|_2 - \lambda_{\min}(D \otimes I_n)$$
$$\le \|B_2\|_2 \|\mathcal{D}(z)\|_2 \|B_1\|_2 \|P\|_2 - d_{\min} \le 1.0 - d_{\min} = -\alpha < 0$$
Global contraction, existence and uniqueness of a globally attractive equilibrium, and exponential disturbance dissipation follow directly from Demidovich's theorem [2, 10]. $\blacksquare$

---

## 3. Analytical Sensitivity and Spatial Signal Attenuation

To understand long-range information propagation, we must distinguish between two fundamentally different concepts of sensitivity:
- **Temporal Trajectory Contraction**: Governs how fast two trajectories $z_1(t), z_2(t)$ converge to each other over time under identical inputs.
- **Spatial Equilibrium Sensitivity**: Governs how much an input excitation at source node $s$ ($x_s$) affects the equilibrium state at target node $v$ ($z^*_v$).

### 3.1 Derivation of the Equilibrium Influence Operator
At equilibrium $z^* \in \mathbb{R}^{nd}$, the system satisfies:
$$(D \otimes I_n) z^* - (B_2^\top \otimes I_n) \Phi\left( (B_1^\top \otimes P) z^* \right) = x$$
Differentiating with respect to $x \in \mathbb{R}^{nd}$, let $J_{\text{eq}} \triangleq \frac{\partial z^*}{\partial x} \in \mathbb{R}^{nd \times nd}$ denote the global equilibrium input-sensitivity Jacobian. Applying the chain rule:
$$\left[ (D \otimes I_n) - (B_2^\top \otimes I_n) \mathcal{D}(z^*) (B_1^\top \otimes P) \right] J_{\text{eq}} = I_{nd}$$
where $\mathcal{D}(z^*) = \operatorname{diag}(1 - \Phi((B_1^\top \otimes P) z^*)^2) \in \mathbb{R}^{nd \times nd}$ with diagonal entries in $[0, 1]$.
Because $D \ge d_{\min} I_d > 0$, the block $(D \otimes I_n)$ is strictly invertible with $(D \otimes I_n)^{-1} = (D^{-1} \otimes I_n)$. Left-multiplying by $(D^{-1} \otimes I_n)$:
$$\left[ I_{nd} - (D^{-1} B_2^\top \otimes I_n) \mathcal{D}(z^*) (B_1^\top \otimes P) \right] J_{\text{eq}} = (D^{-1} \otimes I_n)$$
Define the linear operator $\mathcal{T} \in \mathbb{R}^{nd \times nd}$:
$$\mathcal{T} \triangleq (D^{-1} B_2^\top \otimes I_n) \mathcal{D}(z^*) (B_1^\top \otimes P)$$
The operator norm of $\mathcal{T}$ is bounded by:
$$\|\mathcal{T}\|_2 \le \|D^{-1} B_2^\top \otimes I_n\|_2 \|\mathcal{D}(z^*)\|_2 \|B_1^\top \otimes P\|_2 \le \frac{1}{d_{\min}} \|B_2\|_2 (1.0) \|B_1\|_2 \|P\|_2 \le \frac{\beta}{d_{\min}}$$
where $\beta = \|B_1\|_2 \|B_2\|_2 \le 1.0$. Since $d_{\min} = \beta + \alpha$ with $\alpha > 0$:
$$\|\mathcal{T}\|_2 \le \frac{\beta}{\beta + \alpha} \triangleq \rho < 1.0$$
Because $\|\mathcal{T}\|_2 \le \rho < 1.0$, the matrix $(I_{nd} - \mathcal{T})$ is invertible via its **Neumann series**:
$$J_{\text{eq}} = \sum_{m=0}^\infty \mathcal{T}^m (D^{-1} \otimes I_n)$$

### 3.2 Shortest-Path Nilpotency and Spatial Attenuation Bound

**Proposition 3.1 (Shortest-Path Nilpotency and Spatial Sensitivity Upper Bound)**:
*Let $\mathcal{G}$ be a directed graph, and let $d_{\mathcal{G}}(s, v) = k \ge 1$ denote the directed shortest-path distance from source node $s$ to target node $v$. Assuming that the activation function is elementwise (node-separable) and that $P$'s non-zero entries follow the directed edges of $\mathcal{G}$ ($P_{i u} \neq 0 \implies (u, i) \in \mathcal{E}$), then:*
1. *Shortest-Path Nilpotency: For all $m < k$, the $(v, s)$ node-block of the $m$-th order sensitivity term is identically zero:*
   $$J_{v, s}^{(m)} = 0 \quad \forall m < k$$
2. *Spatial Influence Upper Bound: The operator norm of the spatial influence $[J_{\text{eq}}]_{v, s} \triangleq \frac{\partial z^*_v}{\partial x_s} \in \mathbb{R}^{d \times d}$ is bounded above by:*
   $$\left\| \frac{\partial z^*_v}{\partial x_s} \right\|_2 \le \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$
   *where $\alpha = d_{\min} - \beta > 0$ is the continuous contraction margin.*

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
2. For $m \ge k$, the operator norm of each power is bounded by $\|\mathcal{T}^m\|_2 \le \|\mathcal{T}\|_2^m \le \rho^m$, where the convergence condition $\rho = \beta / d_{\min} = \beta / (\beta + \alpha) < 1.0$ is strictly guaranteed by $\alpha > 0$. Taking the operator norm of the tail sum of the Neumann series from $m = k$ to $\infty$:
   $$\left\| [J_{\text{eq}}]_{v, s} \right\|_2 = \left\| \sum_{m=k}^\infty [\mathcal{T}^m]_{v, s} (D^{-1} \otimes I_n) \right\|_2 \le \|D^{-1}\|_2 \sum_{m=k}^\infty \|\mathcal{T}^m\|_2 \le \frac{1}{d_{\min}} \sum_{m=k}^\infty \rho^m$$
   Evaluating the geometric tail sum yields:
   $$\sum_{m=k}^\infty \rho^m = \rho^k \sum_{j=0}^\infty \rho^j = \frac{\rho^k}{1 - \rho}$$
   Multiplying by $\|D^{-1}\|_2 \le 1/d_{\min}$ gives:
   $$\left\| [J_{\text{eq}}]_{v, s} \right\|_2 \le \frac{1}{d_{\min}} \left( \frac{\rho^k}{1 - \rho} \right) = \left( \frac{1}{d_{\min}(1 - \rho)} \right) \rho^k$$
   We explicitly evaluate the tail prefactor $\frac{1}{d_{\min}(1 - \rho)}$ by substituting $\rho = \beta / d_{\min}$:
   $$1 - \rho = 1 - \frac{\beta}{d_{\min}} = \frac{d_{\min} - \beta}{d_{\min}}$$
   $$d_{\min}(1 - \rho) = d_{\min} \left( \frac{d_{\min} - \beta}{d_{\min}} \right) = d_{\min} - \beta = \alpha$$
   Therefore, the factor $d_{\min}$ in $\|D^{-1}\|_2$ cancels identically with the denominator of the geometric series sum, producing the exact tail prefactor:
   $$\frac{1}{d_{\min}(1 - \rho)} = \frac{1}{d_{\min} - \beta} = \frac{1}{\alpha}$$
   Substituting this identity into the tail inequality yields:
   $$\left\| \frac{\partial z^*_v}{\partial x_s} \right\|_2 \le \frac{1}{\alpha} \rho^k = \frac{1}{\alpha} \left( \frac{\beta}{\beta + \alpha} \right)^k$$
   *(Note on the single-term leading bound vs. full-tail bound: For the isolated leading term at $m = k$, the bound is $\|[\mathcal{T}^k]_{v, s} (D^{-1} \otimes I_n)\|_2 \le \frac{1}{d_{\min}} \rho^k$. Because $d_{\min} = \beta + \alpha \ge \alpha > 0$, the leading-term prefactor satisfies $\frac{1}{d_{\min}} \le \frac{1}{\alpha}$, meaning $\frac{1}{\alpha} \rho^k$ is a valid, slightly looser upper bound for the single $m=k$ step alone. Summing all infinite higher-order path contributions $m \ge k$ scales this leading bound by the geometric accumulation factor $\frac{1}{1 - \rho} = \frac{d_{\min}}{\alpha}$, which cancels $d_{\min}$ and renders $\frac{1}{\alpha} \rho^k$ the rigorous upper bound for the entire equilibrium series tail.)* $\blacksquare$

### 3.3 Boundary of the Sensitivity Upper Bound
We emphasize the formal mathematical scope of Proposition 3.1:
- **Upper Bound Status**: The bound states that the maximum possible linearized equilibrium sensitivity at distance $k$ decays at least as fast as $\rho^k$. It does not establish that actual representation magnitudes decay at this rate, nor that classification recall must drop.
- **Asymptotic Equilibrium vs. Finite Time**: Proposition 3.1 characterizes the stationary state $t \to \infty$. In empirical practice, continuous models are unrolled over a finite horizon $T=4.0$ with 16 explicit Runge-Kutta steps. Truncation before equilibrium and discretization error may also affect practical signal transfer.
- **Interaction with Graph Structure**: On graphs with bounded degrees ($\bar{d} \approx 2$), the normalization $P$ preserves signal energy. On power-law graphs with hub degrees $d \ge 50$, the degree factor $\prod \frac{1}{\sqrt{\text{deg}_{\text{out}} \text{deg}_{\text{in}}}}$ could compound with $\rho^k$. However, the hypothesis that this interaction causes multi-hop classification failure on citation networks remains an open hypothesis that requires direct empirical testing.

---

## 4. Stability-Preserving Architectural Candidates

To explore whether architectures can mitigate spatial attenuation while preserving strict Demidovich contraction ($\mu_2(J) \le -\alpha < 0$), we analyze four architectural designs.

### 4.1 Candidate A: Margin Tuning ($\alpha \to 0^+$)
Setting $d_{\min} \to 1.0^+$ increases the decay factor $\rho = \frac{1}{1 + \alpha} \to 1.0$. For example, setting $d_{\min} = 1.05$ yields $\rho \approx 0.952$.
- *Theoretical Limitation*: As $\alpha \to 0^+$, the system's relaxation time constant $\tau = 1/\alpha$ diverges ($\tau = 20.0$ for $\alpha = 0.05$). Under a fixed integration horizon $T = 4.0$, the trajectory captures only a fraction of convergence, introducing numerical stiffness.

### 4.2 Candidate B: Multiscale Skip-Hop Contractive Dynamics
Candidate B introduces higher-order graph diffusion operators directly into the continuous vector field:
$$\dot{Z} = \gamma_1 \tanh(P Z B_{1, 1}) B_{1, 2} + \gamma_2 \tanh(P^2 Z B_{2, 1}) B_{2, 2} - Z D + X$$
where $\gamma_1, \gamma_2 \ge 0$ with $\gamma_1 + \gamma_2 = 1.0$, $P^2 = (N^2)^\top$, and $\|B_{k, 1}\|_2, \|B_{k, 2}\|_2 \le 1.0$.

**Theorem 4.1 (Certified Stability of Multiscale Contractive Dynamics)**:
*For any convex combination $\gamma_1 + \gamma_2 = 1.0$ and any damping bound $d_{\min} > 1.0$, the multiscale vector field satisfies uniform Demidovich contraction with margin:*
$$\alpha = d_{\min} - 1.0 > 0$$
*Proof*: By submultiplicativity of operator norms, $\|P^2\|_2 \le \|P\|_2^2 \le 1.0^2 = 1.0$.
The total Jacobian is:
$$J(z) = \gamma_1 (B_{1, 2}^\top \otimes P) \mathcal{D}_1 (B_{1, 1}^\top \otimes I) + \gamma_2 (B_{2, 2}^\top \otimes P^2) \mathcal{D}_2 (B_{2, 1}^\top \otimes I) - (D \otimes I)$$
Applying the subadditive and submultiplicative properties of the logarithmic norm:
$$\mu_2(J(z)) \le \gamma_1 \|P\|_2 \|B_{1, 1}\|_2 \|B_{1, 2}\|_2 + \gamma_2 \|P^2\|_2 \|B_{2, 1}\|_2 \|B_{2, 2}\|_2 - d_{\min}$$
$$\le \gamma_1(1.0) + \gamma_2(1.0) - d_{\min} = (\gamma_1 + \gamma_2) - d_{\min} = 1.0 - d_{\min} = -\alpha < 0$$
Thus, Candidate B guarantees Demidovich stability under identical damping bounds. $\blacksquare$

**Theorem 4.2 (Spatial Acceleration Upper Bound)**:
*Candidate B reduces the effective shortest-path nilpotency order from $k$ to $\lceil k/2 \rceil$, transforming the spatial sensitivity upper bound to $\mathcal{O}(\rho^{\lceil k/2 \rceil})$.*
*Proof*: Because $P^2$ connects any node pair separated by a 2-step walk in a single operator evaluation, the $(v, s)$ entry of $(P^2)^m = P^{2m}$ can be non-zero for $2m \ge k$, i.e., $m \ge \lceil k/2 \rceil$. Thus, the earliest *potentially* non-zero term in the Neumann series can appear at $m = \lceil k/2 \rceil$ (governed by the graph support of $(P^2)^m$, though feature weights or activation derivatives could render it zero); for all $m < \lceil k/2 \rceil$, the term is identically zero by shortest-path nilpotency. $\blacksquare$  
*(Qualification: This spatial acceleration is an upper-bound property on structural support; it does not guarantee larger learned signal magnitudes or superior task performance.)*

### 4.3 Candidate C: Structural Readout Augmentation
Candidate C keeps the continuous dynamical core 100% identical to the certified baseline ($d_{\min} \ge 1.50, \alpha \ge 0.50$), but provides local node degree features to the readout MLP:
$$\hat{y}(s, t) = \operatorname{MLP}\left( [Z^*_t; \; \ln(\|Z^*_t\|_2 + 10^{-12}); \; \ln(1 + d_{\text{out}}(s)); \; \ln(1 + d_{\text{in}}(t))] \right)$$
- *Theoretical Guarantee*: The ODE core is unmodified, so $\mu_2(J) \le -\alpha$ holds unconditionally.
- *Crucial Caveat & Open Test (PO-C2)*: While degree features do not contain explicit path labels, in structured graphs (such as chains or layered DAGs) node degrees naturally correlate with topological depth and boundary positions. Because proof obligation PO-C2 (an isolated feature-only classification baseline without ODE latent states) was not conducted and remains OPEN / UNMET, degree features represent an unruled-out confounder. Candidate C's performance cannot be claimed to isolate dynamical reachability or rule out structural leakage, and must be strictly distinguished from Candidate B.

---

## 5. Audited Benchmark Suite and Experimental Safeguards

To evaluate continuous dynamics under simplified, known graph structures with strict sampling and partition controls, we generated an audited benchmark suite across three directed families:
1. **Chain/Ladder (`chain_ladder`)**: $N = 40$ nodes. Directed backbone $i \to i+1$ with forward cross-edges $i \to i+2$ ($p=0.3$). Evaluates distance propagation under low degree heterogeneity ($\bar{d} \approx 2$).
2. **Bottleneck (`bottleneck`)**: $N = 48$ nodes. Two 24-node clusters $A$ and $B$ connected strictly by a single directed bridge edge $u_A \to v_B$. Evaluates propagation across a structural cut.
3. **Layered Random DAG (`random_dag`)**: $N = 42$ nodes in 7 layers (6 nodes/layer) with inter-layer forward edges ($p=0.25$).

### Experimental Safeguards & Independent Audit
To avoid protocol limitations from prior exploratory studies, we enforced four controls:
- **Multi-Instance Graph Splits**: 5 Train instances, 2 Validation instances, and 5 Test instances per family (total 36 graph instances). Train, validation, and test sets are evaluated on distinct graph topologies, guaranteeing **zero query pair overlap**.
- **Zero Replacement Sampling**: Queries within each hop bucket were sampled strictly without replacement.
- **Independent Audit (7,230 Total Reconciled Queries)**: An automated audit script (`audit_benchmark_data.py`) verified all 7,230 queries (3,615 positive, 3,615 negative):
  - 100% agreement with BFS-verified shortest paths and non-reachability.
  - Zero duplicate queries across all instances.
  - Operator norms satisfy $\|P\|_2 \le 1.000000119$ (passing within float32 numerical tolerance of $10^{-6}$).
- **Explicit Scope of Multi-Hop Evaluation**: Because forward edges create shortcut paths, the diameter of the 7-layer DAG is short. Across all 12 instances, `random_dag` contains: hop 1 (240), hop 2 (240), hop 3 (240), hop 4 (14), hop 5 (1), and hop 6 (0). Multi-hop reachability at hops 4, 5, and 6 is supported strictly by `chain_ladder` and `bottleneck`; `random_dag` evaluates reachability across hops 1 through 3.

---

## 6. Empirical Results and Discussion

### 6.1 Experiment 1: Descriptive Damping Parameter Sweep
We evaluated 72 trained model instances (3 families $\times$ 8 configurations $\times$ 3 PRNG seeds) across nominal damping settings $d_{\min} \in \{0.50, 0.80, 1.05, 1.20, 1.50, 2.00\}$, undamped ODE ($D=0$), and discrete `GNN6`. All models were trained under identical optimization budgets (25 epochs with Adam, $lr = 0.005$) and fixed initialization PRNG seeds, with checkpoint selection based on validation graph loss. This sweep evaluates empirical associations under fixed training parameters, rather than establishing a causal effect of damping on performance.

#### Key Observation 1: Monotonic Loss Scaling with Nominal Damping
Seed-averaged cross-entropy loss (both validation and held-out test loss) scales monotonically with the nominal damping lower bound $d_{\min} \in [0.50, 2.00]$ across all three families:
- In `chain_ladder`: mean test loss rises from $0.000761$ ($d_{\min}=0.50$) to $0.001223$ ($d_{\min}=2.00$).
- In `bottleneck`: mean test loss rises from $0.000932$ to $0.001642$.
- In `random_dag`: mean test loss rises from $0.005160$ to $0.009249$.
- The undamped model ($D=0$) achieves the lowest continuous loss ($0.000213$ to $0.002014$).

We note that while seed-averaged losses scale monotonically with nominal damping settings across all three families, individual checkpoints exhibit small run-to-run optimization fluctuations (for example, seed 42 in `bottleneck` and `chain_ladder`). Furthermore, an audit of the per-checkpoint computed conservative contraction margin lower bound ($\alpha_{\text{realized}} = \min(d) - \sigma_1(B_1)\sigma_1(B_2)$) confirms that while seed-averaged realized margins track nominal damping monotonically ($\bar{\alpha}_{\text{realized}}$ rising from $-0.05$ at $d_{\min}=0.50$ to $+1.41$ at $d_{\min}=2.00$), individual checkpoints subject to optimization variance do not follow a strict per-checkpoint monotonic ordering against either metric. We therefore restrict the monotonic loss scaling claim strictly to nominal damping settings at the seed-averaged level under this specific protocol, rather than asserting a per-checkpoint law across realized margins. Finally, because varying nominal damping under fixed hyperparameters observes empirical trends across model configurations without isolating confounders, this relationship is characterized descriptively rather than as an established causal mechanism.

#### Key Observation 2: Benchmark Classification Ceiling
Under multi-instance training on these bounded-degree graphs, **all evaluated models—including all contractive variants, undamped models, and discrete GNN6—achieved 100.0% test accuracy, 100.0% specificity, and 100.0% recall across all evaluated queries**.
- *Interpretation*: Reachability up to 6 hops is supported strictly by the chain/ladder and bottleneck families. This demonstrates that continuous contraction does not mathematically preclude learning multi-hop reachability up to 6 hops on directed chains and bottlenecks under this protocol. However, because test classification metrics are at ceiling across all models, this benchmark does not discriminate models by classification recall.

---

### 6.2 Experiment 2: Architectural Comparison and Optimization Analysis
We evaluated the candidate architectures across 45 training runs (3 families $\times$ 5 architectures $\times$ 3 seeds). All checkpoints were audited, hashed, and evaluated on held-out test instances via `stability_investigation/scripts/evaluate_and_persist_test_loss.py`. Table 1 reports mean and sample standard deviation across the three seeds for both validation loss and **held-out test loss**.

#### Table 1: Comprehensive Optimization & Generalization Metrics Across Graph Families (Mean $\pm$ Sample SD)

| Family | Architecture | Test Acc (%) | Test Spec (%) | Test Rec (%) | Margin Lower Bound $\bar{\alpha}$ | Mean Val Loss | Held-Out Test Loss | $\Delta$ Test Loss vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **chain_ladder** | `Baseline_Damped_1.5` | 100.0 | 100.0 | 100.0 | +0.93 | 0.00094 $\pm$ 0.00028 | 0.001051 $\pm$ 0.000324 | Baseline |
| | `Candidate_A_Tuned_1.05` | 100.0 | 100.0 | 100.0 | +0.49 | 0.00086 $\pm$ 0.00029 | 0.000929 $\pm$ 0.000300 | -11.6% |
| | `Candidate_B_Multiscale_1.5` | 100.0 | 100.0 | 100.0 | +0.89 | **0.00045 $\pm$ 0.00004** | **0.000494 $\pm$ 0.000031** | **-53.0%** |
| | `Candidate_C_Structural_1.5`* | 100.0 | 100.0 | 100.0 | +0.98 | **0.00046 $\pm$ 0.00017** | **0.000496 $\pm$ 0.000191** | **-52.8%** |
| | `Baseline_Discrete_GNN6` | 100.0 | 100.0 | 100.0 | 0.00 | 0.00003 $\pm$ 0.00001 | 0.000234 $\pm$ 0.000072 | -77.7% |
| **bottleneck** | `Baseline_Damped_1.5` | 100.0 | 100.0 | 100.0 | +0.84 | 0.00106 $\pm$ 0.00023 | 0.001366 $\pm$ 0.000338 | Baseline |
| | `Candidate_A_Tuned_1.05` | 100.0 | 100.0 | 100.0 | +0.43 | 0.00094 $\pm$ 0.00024 | 0.001169 $\pm$ 0.000321 | -14.4% |
| | `Candidate_B_Multiscale_1.5` | 100.0 | 100.0 | 100.0 | +0.88 | **0.00055 $\pm$ 0.00016** | **0.000715 $\pm$ 0.000191** | **-47.7%** |
| | `Candidate_C_Structural_1.5`* | 100.0 | 100.0 | 100.0 | +0.93 | 0.00089 $\pm$ 0.00032 | 0.001371 $\pm$ 0.000530 | +0.4% |
| | `Baseline_Discrete_GNN6` | 100.0 | 100.0 | 100.0 | 0.00 | 0.00004 $\pm$ 0.00001 | 0.000076 $\pm$ 0.000022 | -94.4% |
| **random_dag** | `Baseline_Damped_1.5` | 100.0 | 100.0 | 100.0 | +0.94 | 0.00733 $\pm$ 0.00647 | 0.008138 $\pm$ 0.006977 | Baseline |
| | `Candidate_A_Tuned_1.05` | 100.0 | 100.0 | 100.0 | +0.50 | 0.00629 $\pm$ 0.00515 | 0.006908 $\pm$ 0.005558 | -15.1% |
| | `Candidate_B_Multiscale_1.5` | 100.0 | 100.0 | 100.0 | +0.95 | **0.00212 $\pm$ 0.00030** | **0.002501 $\pm$ 0.000318** | **-69.3%** |
| | `Candidate_C_Structural_1.5`* | 100.0 | 100.0 | 100.0 | +1.05 | **0.00104 $\pm$ 0.00031** | **0.000913 $\pm$ 0.000267** | **-88.8%** |
| | `Baseline_Discrete_GNN6` | 100.0 | 100.0 | 100.0 | 0.00 | 0.00029 $\pm$ 0.00007 | 0.000458 $\pm$ 0.000102 | -94.4% |

*(Note: Test Acc, Spec, and Rec are 100.0% across all models due to benchmark ceiling. Sample standard deviations are calculated across 3 seeds per architecture ($ddof=1$). Per-run losses, checkpoint SHA-256 hashes, and dataset hashes are recorded in `stability_investigation/results/architectural_comparison/test_loss_evaluation_certificate.json`. Asterisk (*) indicates Candidate C is subject to open proof obligation PO-C2, detailed below.)*

#### Key Observation 3: Test Loss Reductions for Multiscale Dynamics
`Candidate B` achieved substantial cross-entropy loss reductions on held-out test graphs relative to the single-scale baseline:
- Held-out test loss reduced by **53.0%** on `chain_ladder`, **47.7%** on `bottleneck`, and **69.3%** on `random_dag`.
- Realized contraction margin lower bound $\bar{\alpha}_{\text{realized}} \in [0.88, 0.95]$, matching the baseline's stability certification.
- *Critical Qualification*: Because test classification accuracy is at 100% ceiling across all models, these loss reductions reflect superior optimization and prediction confidence on these synthetic graphs, but **do not demonstrate improved classification recall or generalization** over the baseline.

#### Key Observation 4: Candidate C Structural Readout and Leakage Risk (PO-C2 Unmet)
`Candidate C` provides local node degree features to the readout MLP while retaining the baseline damped ODE core ($d_{\min}=1.5$). On `random_dag`, Candidate C achieved an 88.8% test loss reduction ($0.008138 \to 0.000913$) and on `chain_ladder` a 52.8% reduction ($0.001051 \to 0.000496$), while showing slightly higher loss on `bottleneck` (+0.4%, $0.001366 \to 0.001371$).
However, this result is subject to an essential epistemic caveat: **Proof obligation PO-C2 (evaluating an isolated feature-only classification baseline without ODE latent states) was not performed and remains OPEN / UNMET**. In structured graph families such as chains and layered DAGs, node degrees naturally correlate with topological depth (e.g. boundary layer nodes have distinct in/out degree signatures). Consequently, Candidate C's lower test loss cannot be attributed to superior continuous graph dynamics or leakage-free relational reasoning. Candidate C serves strictly as an exploratory structural readout probe and must not be conflated with the multiscale dynamical propagation demonstrated by Candidate B.

---

## 7. Discussion & Boundaries of Claims

### 7.1 Distinguishing Trajectory Contraction from Spatial Sensitivity
A frequent ambiguity in the literature is conflating temporal contraction with spatial attenuation:
- **Trajectory Contraction** ($\mu_2(J) \le -\alpha < 0$) is a temporal property of the vector field ensuring exponential convergence to a unique equilibrium $Z^*$.
- **Spatial Attenuation** ($\|\frac{\partial z^*_v}{\partial x_s}\| \le \frac{1}{\alpha} \rho^k$) is a spatial property of the equilibrium resolvent governing the linearized response to static input perturbations.

### 7.2 Open Questions and Next Steps
1. **Testing Degree Over-Squashing on Scale-Free Graphs**: Our synthetic graphs have low maximum degrees ($\bar{d} \approx 2$). To test whether degree over-squashing [6, 7] is the true cause of citation network failure [5, 9], future benchmarks must introduce synthetic hubs ($d \ge 50$) in a systematic manner.
2. **Non-Saturated Benchmark Tasks**: Benchmarks that evaluate regression or graded relational reasoning (where metrics do not saturate at 100%) are needed to determine whether Candidate B's loss advantage translates into superior task performance.

---

## 8. Conclusion

In this work, we investigated the relationship between continuous-time contraction and long-range information propagation in graph neural ODEs. Through analytical sensitivity analysis, we derived a formal analytical upper bound on equilibrium input sensitivity governed by shortest-path nilpotency. Using an audited multi-instance synthetic benchmark suite (7,230 queries across 36 graphs), we observed that continuous contractive models achieved 100.0% recall across evaluated queries on small bounded-degree graphs (with 6-hop evaluation supported on chains and bottlenecks), while validation and held-out test loss scaled monotonically with nominal damping under the evaluated protocol. Finally, we introduced a certified Multiscale Skip-Hop Contractive Architecture that preserves uniform Demidovich stability while reducing held-out test cross-entropy loss by up to 69.3%.

These findings show that under the tested training protocol, continuous Demidovich contraction is compatible with learning multi-hop reachability on the evaluated small, bounded-degree digraph families, and provide a formal upper bound on equilibrium spatial sensitivity. However, translating these insights to complex, scale-free graphs with heavy-tailed degree distributions and non-saturated relational reasoning tasks remains an open theoretical and empirical challenge.

---

## References

1. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. *NeurIPS*, 31.
2. Demidovich, B. P. (1967). *Lectures on the Mathematical Theory of Stability*. Nauka, Moscow.
3. Desoer, C. A., & Vidyasagar, M. (1975). *Feedback Systems: Input-Output Properties*. Academic Press.
4. Haber, E., & Ruthotto, L. (2017). Stable architectures for deep neural networks. *Inverse Problems*, 34(1), 014004.
5. Poli, M., Massaroli, S., Park, J., Yamashita, A., Asama, H., & Park, J. (2019). Graph neural ordinary differential equations. *arXiv preprint arXiv:1911.07532*.
6. Alon, U., & Yahav, E. (2021). On the bottleneck of graph neural networks and its practical implications. *ICLR*.
7. Topping, J., Giovanni, F. D., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2022). Understanding over-squashing and bottlenecks on graphs via curvature. *ICLR*.
8. Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2008). The graph neural network model. *IEEE TNN*, 20(1), 61-80.
9. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
10. Sontag, E. D. (1998). *Mathematical Control Theory: Deterministic Finite Dimensional Systems*. Springer.
