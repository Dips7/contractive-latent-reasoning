# Theory Note: Mathematical Foundations of Contractive G-NODEs

**Date**: 2026-11-16  
**Document Status**: PHASE 1 THEORY DELIVERABLE  
**Governing Branch**: `stability_investigation/`  
**Parent Investigation**: Contractive Graph Dynamics Rework  

---

## 1. Exact Continuous-Time Model Formulation

The system under investigation is an autonomous matrix ordinary differential equation modeling continuous-depth information flow over a directed graph $G = (V, E)$ with $n = |V|$ nodes and latent feature dimension $d = 8$:

$$\dot{Z}(t) = f(Z(t); X) = \tanh(P Z(t) B_1) B_2 - Z(t) D + X, \quad Z(0) = X$$

### 1.1 Mathematical Entities and Dimensions
- **State Matrix**: $Z(t) \in \mathbb{R}^{n \times d}$. Row $Z_{v, :}(t) \in \mathbb{R}^d$ represents the continuous latent embedding of node $v \in V$ at continuous time $t \in [0, T]$.
- **Inner Product & Norm**: $\mathbb{R}^{n \times d}$ is equipped with the standard Frobenius inner product:
  $$\langle U, V \rangle_F \triangleq \operatorname{tr}(U^\top V) = \sum_{i=1}^n \sum_{j=1}^d U_{ij} V_{ij}, \quad \|U\|_F = \sqrt{\langle U, U \rangle_F}$$
- **Directed Graph Adjacency & Bipartite Normalization**:
  Let $C \in \{0, 1\}^{n \times n}$ be the binary directed adjacency matrix ($C_{ij} = 1$ denotes a directed citation/link $i \to j$). Out-degree $\text{out}_i = \sum_j C_{ij}$ and in-degree $\text{in}_j = \sum_i C_{ij}$.
  The normalized adjacency operator $N \in \mathbb{R}^{n \times n}$ is defined as:
  $$N_{ij} = \begin{cases} \frac{C_{ij}}{\sqrt{\text{out}_i \text{in}_j}} & \text{if } \text{out}_i > 0 \text{ and } \text{in}_j > 0 \\ 0 & \text{otherwise} \end{cases}$$
  The message-passing propagation operator is $P = N^\top \in \mathbb{R}^{n \times n}$. For any state $Z$, the forward aggregated representation at node $v$ is:
  $$(P Z)_{v, :} = \sum_{u: u \to v} \frac{1}{\sqrt{\text{out}_u \text{in}_v}} Z_{u, :}$$
  By Proposition 3.1, $\|P\|_2 = \|N\|_2 \le 1.000000$ strictly.
- **Feature Transformation Operators**:
  Linear weight matrices $B_1 \in \mathbb{R}^{d \times d_{\text{hid}}}$ and $B_2 \in \mathbb{R}^{d_{\text{hid}} \times d}$ with $d = d_{\text{hid}} = 8$. In the implemented PyTorch model (`FixedFieldReasoner`), $B_1 = W_1^\top$ and $B_2 = W_2^\top$, where $W_1, W_2$ are constrained via singular value decomposition (SVD) projection once per forward pass:
  $$\tilde{W} = \frac{W}{\max(1, \sigma_{\max}(W)(1 + 10^{-5}))} \implies \|B_1\|_2 \le 1, \quad \|B_2\|_2 \le 1$$
- **Coupling Constant**:
  $$\beta \triangleq \|P\|_2 \|B_1\|_2 \|B_2\|_2 \le 1.000000$$
- **Coordinate Damping Operator**:
  $D = \operatorname{diag}(d_1, \dots, d_d) \in \mathbb{R}^{d \times d}$, parameterized as:
  $$d_j = \operatorname{softplus}(\theta_j) + 1.5, \quad \theta_j \in \mathbb{R} \implies d_{\min} \triangleq \min_{1 \le j \le d} d_j \ge 1.500000$$
- **Contraction Margin**:
  $$\alpha \triangleq d_{\min} - \beta \ge 1.500000 - 1.000000 = 0.500000 > 0$$
- **Query Input Embedding**:
  $X \in \mathbb{R}^{n \times d}$. For reachability query $(s, t)$, $X_{s, 1} = 1.0$ (index 0 in tensor coordinates) and $X_{v, j} = 0$ for all $(v, j) \ne (s, 1)$. $X$ remains fixed for all $t \ge 0$.
- **Readout Classifier**:
  Target node representation at final horizon $Z_{t, :}(T) \in \mathbb{R}^d$.
  Feature vector:
  $$u(s, t) = \left[ Z_{t, :}(T); \; \ln(\|Z_{t, :}(T)\|_2 + 10^{-12}) \right] \in \mathbb{R}^{d+1} = \mathbb{R}^9$$
  Two-layer MLP readout:
  $$\hat{y}(s, t) = W_{\text{out}, 2} \; \operatorname{GELU}(W_{\text{out}, 1} u(s, t) + b_1) + b_2 \in \mathbb{R}^2$$
  Predicted class $\hat{c} = \arg\max_{k \in \{0, 1\}} \hat{y}_k$.

---

## 2. Established Mathematical Theorems (What Contraction Directly Controls)

1. **Theorem 4.1 (Trajectory Contraction)**:
   For two solutions $Z_1(t), Z_2(t)$ driven by the **same** input $X$:
   $$\|Z_1(t) - Z_2(t)\|_F \le e^{-\alpha t} \|Z_1(0) - Z_2(0)\|_F$$
   *Meaning*: Trajectories originating from different initial states collapse together exponentially fast.
2. **Theorem 4.2 (Unique Equilibrium)**:
   There exists a unique $Z^*(X)$ satisfying $f(Z^*; X) = 0$, and every trajectory converges to $Z^*(X)$ at exponential rate $\alpha$:
   $$\|Z(t) - Z^*(X)\|_F \le e^{-\alpha t} \|Z(0) - Z^*(X)\|_F$$
3. **Proposition 4.1 (Disturbance Dissipation)**:
   Any additive time-varying disturbance $w(t)$ with $\|w(t)\|_F \le \epsilon$ induces a bounded steady-state deviation:
   $$\limsup_{t \to \infty} \|Z(t) - Z^*(X)\|_F \le \frac{\epsilon}{\alpha}$$

---

## 3. What the Contraction Theorem Does NOT Control

The contraction theorem states that trajectories with identical inputs converge to each other. It makes **zero statements** regarding:
1. How rapidly or effectively a source signal $X_s$ propagates to distant nodes $v$ across the graph topology;
2. Whether the target node embedding $Z_t(T)$ maintains a non-zero discriminative margin between reachable and unreachable queries;
3. Whether discrete solver trajectories (RK4 with $h = 4/24$) preserve contractive decay;
4. The classification accuracy of the downstream MLP readout.

In Phase 1, we formalize the missing link: **spatial influence sensitivity analysis**.
