# Engineering Implementation Plan: Contractive Latent Dynamical Reasoning

**Project Codename**: `contractive-latent-reasoning` (CLR)  
**Objective**: Replace discrete autoregressive chain-of-thought (CoT) token generation with an algebraically constrained, continuous-time contractive latent dynamical system $(\mathbf{z} \in \mathbb{R}^d, \dot{\mathbf{z}} = \mathbf{f}_\theta(\mathbf{z}; \mathbf{x}))$.

---

## 1. Executive Summary & Core Hypothesis

Current frontier models attempt to scale test-time reasoning by generating long sequences of autoregressive tokens. In open-ended domains without external verification oracles, this open-loop process exhibits **exponential error drift** ($e^{-\sum \epsilon_t}$) and high sample complexity.

This project implements a continuous-time reasoning architecture:
1. **Latent State Evolution**: Thinking is formulated as an ODE in continuous latent space $\mathbf{z}(t) \in \mathbb{R}^d$ driven by a problem-conditioned potential $E_\theta(\mathbf{z}; \mathbf{x})$:
   $$\frac{d\mathbf{z}}{dt} = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{x}) - \mathbf{D}\mathbf{z}, \quad \mathbf{D} \succ 0$$
2. **Demidovich Contraction**: The symmetric Jacobian $\text{Sym}(\mathbf{J}) = \frac{1}{2}(\mathbf{J} + \mathbf{J}^T)$ is bounded such that $\lambda_{\max}(\text{Sym}(\mathbf{J})) \le -\kappa < 0$. This mathematically guarantees exponential convergence to a unique fixed point $\mathbf{z}^*(\mathbf{x})$, rendering reasoning robust to numerical perturbations and eliminating runaway hallucinations.
3. **Test-Time Scaling**: Compute scales by increasing the integration horizon $T$ and solver precision in continuous time, not by filling discrete token memory buffers.

### Core Falsification Protocol & Kill Criterion
To prevent motivated reasoning and avoid sunk-cost traps on speculative architectures, this project adheres to an explicit falsification rule:
> **Kill Criterion**: If the contractive latent reasoner (Approach A on parity + reachability) fails to outperform a parameter-matched recurrent/LSTM baseline within **2 weeks of compute**, or if ICNN convexity renders the energy landscape fundamentally unable to separate parity states without numerical divergence, the core thesis will be declared **unvalidated at this scale** and the approach will be stopped or pivoted.

### Prioritized 5-Step Execution Sequencing
1. **Smallest End-to-End Loop First**: $N$-bit parity + Approach A (ICNN + positive damping) only. Bypass reachability, $S_n$, and Approach B until a clean data $\to$ latent ODE $\to$ probe $\to$ loss loop runs without NaNs.
2. **Online Contraction Diagnostics**: Track the Demidovich spectral bound $\lambda_{\max}(\text{Sym}(\mathbf{J})) \le -\kappa < 0$ and empirical multi-seed distance decay $\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\| \le e^{-\kappa t} \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\|$ dynamically every $N$ training steps on real trained weights, not merely at initialization.
3. **ICNN Expressivity Bottleneck Decision**: Use parity training dynamics to immediately decide whether ICNN convexity is too restrictive before moving to multi-step tasks.
4. **Deferred Hybrid Decoder**: Probe $\mathbf{z}^*$ with simple linear heads; wire into complex autoregressive token decoders only after $\mathbf{z}^*$ provably carries the required invariant state information.
5. **Kill Criterion Enforcement**: Continuous assessment against the 2-week baseline threshold.

---

## 2. Directory Layout & Module Responsibilities

```text
contractive-latent-reasoning/
├── IMPLEMENTATION_PLAN.md         # Full engineering blueprint and specification
├── README.md                      # Project setup, quickstart, and reproduction guide
├── pyproject.toml                 # Package specification and dependencies
├── requirements.txt               # Pinned Python dependencies
├── configs/                       # Hydra / YAML experiment configurations
│   ├── base.yaml                  # Global training, logging, and hardware defaults
│   ├── phase1_parity.yaml         # Config for N-step parity task
│   ├── phase1_graph.yaml          # Config for graph connectivity task
│   ├── phase1_permutation.yaml    # Config for S_n group permutation task
│   ├── phase2_contraction.yaml    # Contraction ablations and spectral penalty tuning
│   └── phase3_hybrid.yaml         # Hybrid latent-to-token decoder integration
├── data/
│   ├── README.md                  # Synthetic benchmark data documentation
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── parity.py              # Long-horizon parity generator (up to N=128 bits)
│   │   ├── graph_connectivity.py  # Path-finding and reachability on random graphs
│   │   └── permutation_groups.py  # Word problem solver over symmetric groups S_n
│   └── synthetic/                 # Local directory for cached synthetic datasets
├── src/
│   ├── __init__.py
│   ├── dynamics/                  # Continuous vector field & ODE dynamics
│   │   ├── __init__.py
│   │   ├── energy.py              # E_theta(z; x) architectures (ICNN, MLP, Quadratic)
│   │   ├── vector_field.py        # Vector field f(z; x) = -∇_z E - D·z
│   │   ├── contraction.py         # Demidovich projectors, spectral hooks, MonDEQ blocks
│   │   └── solvers.py             # torchdiffeq wrappers, adaptive RK4, adjoint sensitivity
│   ├── models/                    # Neural network components
│   │   ├── __init__.py
│   │   ├── encoder.py             # x -> condition context vector c(x)
│   │   ├── latent_reasoner.py     # Continuous ODE reasoning block
│   │   ├── decoder.py             # Readout head (linear classifier or autoregressive decoder)
│   │   └── hybrid_model.py        # End-to-end integrated model pipeline
│   ├── training/                  # Optimization and loop execution
│   │   ├── __init__.py
│   │   ├── loss.py                # Task loss + spectral penalty + Lyapunov regularizer
│   │   ├── trainer.py             # PyTorch training engine with W&B logging
│   │   └── scheduler.py           # Curriculum scheduler for integration time T
│   ├── diagnostics/               # Numerical validation tools
│   │   ├── __init__.py
│   │   ├── jacobian_analyzer.py   # Power-iteration estimator for lambda_max(Sym(J))
│   │   ├── trajectory_tracker.py  # Multi-seed trajectory divergence tracker ||z_1(t) - z_2(t)||
│   │   └── energy_visualizer.py   # Phase-space vector field and contour plotting
│   └── utils/                     # General utilities
│       ├── __init__.py
│       └── metrics.py             # Convergence rates, accuracy, solver step counter
├── experiments/                   # Executable training and ablation scripts
│   ├── run_phase1_proof_of_concept.py
│   ├── run_phase2_contraction_ablation.py
│   └── run_phase3_hybrid_evaluation.py
├── tests/                         # Pytest automated test suite
│   ├── __init__.py
│   ├── test_data_generators.py
│   ├── test_vector_field.py
│   ├── test_contraction_conditions.py
│   ├── test_solvers.py
│   └── test_end_to_end.py
├── notebooks/                     # Exploratory research notebooks
│   ├── 01_phase_space_diagnostics.ipynb
│   └── 02_trajectory_convergence_demo.ipynb
└── docs/                          # Technical specifications and math notes
    ├── architecture_spec.md
    └── mathematical_reference.md
```

---

## 3. Phase 1 — Proof of Concept: Algorithmic Invariants

### 3.1 Benchmark Selection: Depth-Limited Failure Modes
Standard constant-depth transformers without CoT fail on problems requiring sequential state tracking (circuit class $\text{TC}^0$). We select three synthetic tasks with strict algebraic definitions:

1. **Long-Horizon Parity ($N$-Bit)**:
   - Input: Binary sequence $\mathbf{x} \in \{-1, +1\}^N$, with sequence length $N \in [16, 128]$.
   - Target: Parity $y = \prod_{i=1}^N x_i \in \{-1, +1\}$.
   - Target Failure Mode: Fixed-depth transformers fail as $N > 2 \times \text{layers}$.
2. **Directed Graph Connectivity ($k$-Hop Reachability)**:
   - Input: Edge list for a directed graph $G=(V, E)$ ($|V|=32, 64$) and node pair $(u, v)$.
   - Target: Boolean reachability $y \in \{0, 1\}$ with path length $k \in [4, 16]$.
   - Target Failure Mode: Requires iterative pointer chasing; transformers without scratchpads struggle at high $k$.
3. **Word Problem on Symmetric Groups $S_n$**:
   - Input: Sequence of permutations $g_1, g_2, \dots, g_L \in S_5$.
   - Target: Composite permutation $\pi = g_1 \circ g_2 \circ \dots \circ g_L$.
   - Target Failure Mode: Non-abelian group composition with high compositional depth.

### 3.2 Model Architecture (Latent ODE Block)
- **Conditioning Encoder**: Small Transformer or bidirectional GRU mapping input sequence $\mathbf{x}$ to context vector $\mathbf{c} \in \mathbb{R}^{d_c}$.
- **Latent Reasoner**:
  - State dimension: $\mathbf{z} \in \mathbb{R}^{d_z}$ (default $d_z = 128$).
  - Initial state: $\mathbf{z}_0 = \mathbf{0}$ or projected from $\mathbf{c}$.
  - Vector field $\mathbf{f}_\theta(\mathbf{z}, \mathbf{c})$: 2-layer MLP with Swish/GELU activations:
    $$\frac{d\mathbf{z}}{dt} = \mathbf{f}_\theta(\mathbf{z}, \mathbf{c}) = \mathbf{W}_2 \phi(\mathbf{W}_1 \mathbf{z} + \mathbf{U} \mathbf{c} + \mathbf{b}_1) + \mathbf{b}_2 - \gamma \mathbf{z}$$
- **ODE Integrator**: `torchdiffeq.odeint` with explicit Runge-Kutta 4 (`rk4`) or adaptive `dopri5` over integration interval $t \in [0, T]$, where $T \in [1.0, 10.0]$.
- **Readout Head**: Linear projection $\hat{y} = \text{Linear}(\mathbf{z}(T))$.

### 3.3 Training Loop & Success Metrics
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$).
- **Gradient Computation**: Adjoint sensitivity method (`torchdiffeq.odeint_adjoint`) for $\mathcal{O}(1)$ memory scaling with respect to integration steps.
- **Success Criteria**:
  - $\ge 99.5\%$ test accuracy on $N=64$ bit parity (where standard Transformer baseline gets $\sim 50\%$).
  - Stable training loss without gradient explosion or NaN trajectories.

---

## 4. Phase 2 — Contractive Constraint Enforcement

### 4.1 Mathematical Formulation of the Demidovich Bound
A continuous dynamical system $\dot{\mathbf{z}} = \mathbf{f}(\mathbf{z})$ is strictly contractive with respect to the Euclidean metric if:

$$\lambda_{\max}\left(\frac{\mathbf{J}_\mathbf{f}(\mathbf{z}) + \mathbf{J}_\mathbf{f}^T(\mathbf{z})}{2}\right) \le -\kappa < 0 \quad \forall \mathbf{z} \in \mathbb{R}^d$$

Under this condition, for any two solutions $\mathbf{z}_1(t), \mathbf{z}_2(t)$:
$$\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\|_2 \le e^{-\kappa t} \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\|_2$$

### 4.2 Engineering Approaches for Enforcing Contraction

We evaluate two complementary enforcement mechanisms:

#### Approach A: Input-Convex Potential + Damping (Exact by Construction)
We parametrize the vector field as the negative gradient of a strictly convex potential function plus linear damping:
$$\mathbf{f}_\theta(\mathbf{z}; \mathbf{c}) = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{c}) - \mathbf{D}\mathbf{z}$$
- $E_\theta(\mathbf{z}; \mathbf{c})$ is structured as an **Input Convex Neural Network (ICNN)** (Amos et al., 2017):
  $$E_\theta(\mathbf{z}; \mathbf{c}) = g_K(\mathbf{z}), \quad g_{l+1} = \sigma(\mathbf{W}_l^{(z)} g_l + \mathbf{W}_l^{(c)} \mathbf{c} + b_l), \quad \mathbf{W}_l^{(z)} \ge 0$$
- Non-negative weights $\mathbf{W}_l^{(z)} = \text{softplus}(\mathbf{V}_l)$ ensure convexity ($\nabla^2_\mathbf{z} E \succeq 0$).
- Damping matrix $\mathbf{D} = \text{diag}(d_1, \dots, d_d) + \kappa \mathbf{I}$ with $d_i \ge 0$.
- Because $\nabla^2_\mathbf{z} E \succeq 0$ and $\mathbf{D} \succeq \kappa \mathbf{I}$, the Jacobian satisfies:
  $$\mathbf{J}_\mathbf{f}(\mathbf{z}) = -\nabla^2_\mathbf{z} E - \mathbf{D} \implies \text{Sym}(\mathbf{J}_\mathbf{f}) \preceq -\kappa \mathbf{I} < 0$$
- *Guarantee*: Contraction is guaranteed for all parameters without penalty hyperparameter tuning.

#### Approach B: Direct Vector Field with Monotone/Lipschitz Projection (High Expressivity)
For general vector fields $\mathbf{f}_\theta(\mathbf{z}, \mathbf{c}) = \mathbf{W}_2 \phi(\mathbf{W}_1 \mathbf{z}) - \mathbf{D}\mathbf{z}$:
- Apply **Spectral Normalization** on $\mathbf{W}_1$ and $\mathbf{W}_2$ such that $\|\mathbf{W}_1\|_2 \le \sigma_1, \|\mathbf{W}_2\|_2 \le \sigma_2$.
- Add an explicit spectral contraction loss penalty during training:
  $$\mathcal{L}_{\text{spectral}}(\theta) = \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}}\left[ \max\left(0, \lambda_{\max}(\text{Sym}(\mathbf{J}_\mathbf{f}(\mathbf{z}))) + \kappa \right)^2 \right]$$
- Approximate $\lambda_{\max}(\text{Sym}(\mathbf{J}_\mathbf{f}))$ during training using 3 iterations of **Hessian-Free Power Iteration**:
  $$\mathbf{v} \leftarrow \frac{\text{Sym}(\mathbf{J})\mathbf{v}}{\|\text{Sym}(\mathbf{J})\mathbf{v}\|_2}, \quad \text{where } \mathbf{J}\mathbf{v} = \left. \frac{d}{d\epsilon} \mathbf{f}(\mathbf{z} + \epsilon \mathbf{v}) \right|_{\epsilon=0}$$
  evaluated via forward-mode automatic differentiation (PyTorch `torch.func.jvp`).

### 4.3 Numerical Verification Suite
1. **Multi-Seed Trajectory Convergence Test**:
   - For a given input $\mathbf{x}$, sample $K=10$ random initial conditions $\mathbf{z}_0^{(k)} \sim \mathcal{N}(0, \mathbf{I})$.
   - Integrate each trajectory to $T=10.0$.
   - Assert: $\max_{j, k} \|\mathbf{z}_j(T) - \mathbf{z}_k(T)\|_2 < 10^{-4}$.
2. **Jacobian Eigenspectrum Profiler**:
   - Sample $M=500$ points along the integrated path.
   - Compute full explicit Jacobians via `torch.func.jacfwd`.
   - Assert: $\max_m \lambda_{\max}\left(\frac{\mathbf{J}_m + \mathbf{J}_m^T}{2}\right) \le -\kappa + \text{tol}$.

---

## 5. Phase 3 — Hybrid Interface: Continuous Thought to Discrete Readout

```
+-------------------------------------------------------------------------+
|                               Input x                                   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  Conditioning Encoder (Transformer / MLP)               |
|                               c(x) in R^{d_c}                           |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    Continuous Latent Reasoner (ODE)                     |
|                                                                         |
|   z_0 = 0 (or h(c))                                                     |
|   dz/dt = -∇_z E_θ(z; c) - D·z                                          |
|   Demidovich Bound: λ_max(Sym(J)) <= -κ < 0                             |
|   Integration: t in [0, T] via RK4 / Dopri5                             |
|                                                                         |
|                                  z*(x)                                  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    Latent Projection Bridge (Linear / MLP)              |
|                             e_thought in R^{d_model}                    |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   Autoregressive Decoder (Transformer)                  |
|                                                                         |
|   Prefix Tokens:   [BOS, <THOUGHT_EMBEDDING>, x_1, x_2, ...]            |
|   Direct Readout:  P(y_1, ..., y_M | z*(x), x)                          |
|   (Bypasses token-by-token chain-of-thought derivation)                 |
+-------------------------------------------------------------------------+
```

### 5.1 Interface Specification
- **Latent Thought Vector**: $\mathbf{z}^* = \mathbf{z}(T) \in \mathbb{R}^{d_z}$.
- **Bridge Network**: Projector $\mathbf{P} \in \mathbb{R}^{d_{\text{model}} \times d_z}$ maps the contracted state into the decoder's token embedding space:
  $$\mathbf{e}_{\text{thought}} = \mathbf{W}_p \mathbf{z}^* + \mathbf{b}_p$$
- **Decoder Conditioning**: $\mathbf{e}_{\text{thought}}$ is prepended as a soft prompt prefix:
  $$\mathbf{H}_0 = [\mathbf{e}_{\text{thought}}, \mathbf{e}_{\text{token}_1}, \mathbf{e}_{\text{token}_2}, \dots]$$

### 5.2 Training Regimes
We implement and compare two training strategies:
1. **End-to-End Joint Training**:
   - Backpropagate token cross-entropy loss directly through the decoder, through the bridge, and through the ODE using `odeint_adjoint`.
   - Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}}(y, \hat{y}) + \lambda_c \mathcal{L}_{\text{contraction}} + \lambda_e \|\nabla_\mathbf{z} E(\mathbf{z}^*)\|^2$.
2. **Two-Stage Decoupled Training**:
   - *Stage 1*: Train Encoder + Latent Reasoner on abstract state targets or direct task classification (e.g., parity/graph labels).
   - *Stage 2*: Freeze the Latent Reasoner; train only the bridge and the discrete token decoder to verbalize the contracted state $\mathbf{z}^*$.

---

## 6. Comprehensive Evaluation Plan

### 6.1 Baseline Comparisons
All baselines matched to identical parameter counts ($\sim 5\text{M}$ params for Phase 1/2 synthetic; $\sim 50\text{M}$ for Phase 3):
1. **Standard Transformer (Direct)**: Transformer encoder-decoder predicting answer directly without scratchpad.
2. **Autoregressive CoT Transformer**: Transformer trained to emit step-by-step intermediate tokens before outputting the final answer.
3. **Recurrent Transformer (Universal Transformer / R-Transformer)**: Discrete step recurrence in latent space: $\mathbf{z}_{k+1} = \mathbf{z}_k + \text{TransformerBlock}(\mathbf{z}_k, \mathbf{c})$.
4. **Deep Equilibrium Model (DEQ)**: Root-finding equilibrium model using quasi-Newton methods (Broyden) without contraction guarantees.

### 6.2 Key Ablation Matrix
- **Contraction vs. Free ODE**: Compare strictly contractive ODE (ICNN + Damping) against an unconstrained Neural ODE ($\dot{\mathbf{z}} = \text{MLP}(\mathbf{z})$). Measure stability under initial condition noise $\mathbf{z}_0 + \epsilon$.
- **Test-Time Compute Scaling**: For a fixed trained model, sweep integration horizon $T \in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]$ and solver tolerances $\text{rtol}, \text{atol} \in [10^{-2}, 10^{-7}]$. Measure if performance scales monotonically with solver compute without emitting tokens.
- **Noise Robustness**: Inject additive Gaussian noise $\mathbf{z}(t_i) \leftarrow \mathbf{z}(t_i) + \mathcal{N}(0, \sigma^2 \mathbf{I})$ at midpoint $t_i = T/2$. Evaluate recovery back to $\mathbf{z}^*$.

### 6.3 Performance & Cost Metrics
- **Task Accuracy**: Exact match on answer token.
- **Compute Efficiency**: Number of Function Evaluations (NFE) vs. FLOPs.
- **Memory Footprint**: Peak VRAM during training and inference compared to KV-cache growth in CoT.

---

## 7. Risks, Failure Modes & Mitigations

| Risk | Mathematical Cause | Engineering Mitigation |
| :--- | :--- | :--- |
| **Expressivity Bottleneck** | A strictly contractive vector field has a unique global equilibrium; it cannot support multimodality (e.g. creative/branching tasks). | Restrict contraction to be **input-conditioned**: the unique attractor $\mathbf{z}^*(\mathbf{x})$ is unique *given $\mathbf{x}$*, while varying non-linearly with $\mathbf{x}$. For inherently multimodal tasks, use a mixture of contractive latent fields. |
| **Solver Stiffness & Explosion of NFEs** | If eigenvalues of $\mathbf{J}$ have huge negative real parts, explicit solvers (RK4) require extremely small step sizes ($h < 2/|\lambda_{\max}|$). | Use implicit or semi-implicit A-stable solvers (e.g., `implicit_adams`, `radau`), or regularize the condition number $\kappa(\mathbf{J})$ to prevent stiffness. |
| **Vanishing Gradients in Adjoint State** | Integrating backwards through $e^{-\kappa t}$ flow corresponds to forward explosion $e^{+\kappa t}$ in the backward adjoint ODE. | Bound integration time $T$; use checkpointed reverse integration or direct equilibrium backprop (implicit function theorem: $\frac{d\mathbf{z}^*}{d\theta} = -(\mathbf{J}_{\mathbf{z}^*})^{-1} \frac{\partial \mathbf{f}}{\partial \theta}$). |

---

## 8. Milestones & Timeline

```
Milestone 1: Proof-of-Concept Verification (Weeks 1-2)
├── Build data generators (parity, graph connectivity, permutation).
├── Implement baseline Neural ODE block in PyTorch with torchdiffeq.
└── Achieve >99% on parity N=64; demonstrate failure of direct Transformer baseline.

Milestone 2: Contractive Framework & Diagnostics Suite (Weeks 3-4)
├── Implement ICNN energy potential + strictly positive damping D.
├── Build automated Demidovich Jacobian analyzer and multi-seed convergence tests.
└── Run ablation: Unconstrained ODE vs. Contractive ODE under perturbation.

Milestone 3: Hybrid Architecture & Test-Time Scaling (Weeks 5-6)
├── Implement Latent-to-Decoder projection bridge.
├── Evaluate test-time compute scaling: Accuracy vs. Integration Time T and NFE.
└── Compare against autoregressive CoT in terms of latency and memory.

Milestone 4: Manuscript & Benchmark Release (Weeks 7-8)
├── Compile empirical benchmark report (Accuracy, FLOPs, Memory, Noise-robustness).
└── Package reproducible codebase and open-source models.
```

### Minimal Viable Result (MVR) for First Paper/Report
Demonstrating that on algorithmic tasks where standard transformers fail without CoT, a **contractive latent ODE with fixed parameters solves the task with 100% precision, scales performance purely by increasing integration time $T$, and exhibits zero error degradation when subjected to mid-trajectory state noise.**
