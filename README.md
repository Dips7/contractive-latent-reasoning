# Contractive Latent Dynamical Reasoning (CLR)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A continuous-time reasoning architecture that replaces discrete autoregressive chain-of-thought (CoT) with an algebraically constrained, contractive dynamical system.

---

## Key Mathematical Idea

Instead of emitting thousands of unverified tokens where error accumulates multiplicatively ($e^{-\sum \epsilon_t}$), we formulate reasoning as an energy-minimizing flow in a continuous latent space $\mathbf{z} \in \mathbb{R}^d$:

$$\frac{d\mathbf{z}}{dt} = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{x}) - \mathbf{D}\mathbf{z}, \quad \mathbf{D} \succ 0$$

By bounding the symmetric Jacobian of the vector field:
$$\lambda_{\max}\left(\frac{\mathbf{J} + \mathbf{J}^T}{2}\right) \le -\kappa < 0$$
all trajectories converge exponentially to a single, unique fixed point $\mathbf{z}^*(\mathbf{x})$, guaranteeing robustness against hallucinations and enabling continuous test-time compute scaling by integrating longer in time ($T$).

---

## Directory Layout

```text
contractive-latent-reasoning/
├── IMPLEMENTATION_PLAN.md    # Detailed research & engineering master plan
├── README.md                 # Project overview and quickstart
├── pyproject.toml            # Python packaging metadata
├── requirements.txt          # Pinned dependencies
├── configs/                  # Hydra/YAML experiment configurations
├── data/                     # Synthetic benchmark generators
│   └── generators/           # Parity, Graph Connectivity, S_n Groups
├── src/                      # Core package source code
│   ├── dynamics/             # Energy potentials, vector fields, contraction hooks, solvers
│   ├── models/               # Encoders, ODE reasoners, decoders, hybrid models
│   ├── training/             # Loss functions, trainers, integration schedulers
│   ├── diagnostics/          # Jacobian eigenspectrum, multi-seed convergence tracker
│   └── utils/                # Metrics and helpers
├── experiments/              # Executable experiment entry points
├── tests/                    # Unit and regression test suite
├── notebooks/                # Phase space and trajectory visualization
└── docs/                     # Architectural and mathematical specifications
```

---

## Quickstart

### 1. Installation

```bash
cd contractive-latent-reasoning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite

```bash
pytest tests/
```

### 3. Run Experiments & Audits

- **Phase 1 Proof-of-Concept (N-bit Parity)**:
  ```bash
  python experiments/run_phase1_proof_of_concept.py --config configs/phase1_parity.yaml
  ```
- **Phase 2 Contraction Diagnostics & Eigenspectrum**:
  ```bash
  python experiments/run_phase2_contraction_ablation.py --config configs/phase2_contraction.yaml
  ```
- **Multi-Hop Graph Reachability & Continuous Test-Time Scaling**:
  ```bash
  python experiments/run_graph_reachability_experiment.py
  ```
- **Perturbation & Self-Healing Stress Test**:
  ```bash
  python experiments/run_perturbation_stress_test.py
  ```
- **Real-World Cora Academic Network Benchmark (2,708 papers, 5,429 citations)**:
  ```bash
  python experiments/run_cora_real_world_experiment.py
  ```
- **Phase 3 Hybrid Interface (Continuous Thought to Token Decoder)**:
  ```bash
  python experiments/run_phase3_hybrid_reasoning.py
  ```
- **Parity Generalization & Overlap Audit**:
  ```bash
  python experiments/run_parity_generalization_audit.py
  ```

---

## Verified Empirical Results (Post-Audit Benchmarks)

All results below are reproducible and persisted as verifiable JSON artifacts in `experiments/results/`:

| Benchmark | Baseline | CLR (Contractive ODE) | Key Mechanism & Finding |
| :--- | :--- | :--- | :--- |
| **Real-World Cora Citation Network** (2,708 papers, 5,429 citations) | 68.67% (Direct MLP) / 60.67% (GNN-2) / 74.67% (GNN-4) / 87.67% (GNN-6) / 86.67% (GNN-8) | **100.00%** (T=4.0, isolated seed 42) | Continuous flow outperforms discrete message passing even when horizon ceiling is 100% (GNN-6: 87.7%, GNN-8: 86.7% due to layer-wise optimization degradation and over-smoothing). 0.0% deep-hop accuracy on GNN-2/4 explained by majority-class default (89.3% and 75.3% predicted unreachable). Monotonic $T$-scaling: 90.3% ($T=0.5$) $\to$ 96.7% ($T=1.0$) $\to$ 98.7% ($T=2.0$) $\to$ 100.0% ($T\ge 4.0$). All 6 checkpoints saved in `checkpoints/`. |
| **Multi-Hop Reachability** (16 nodes) | 54.50% (Constant depth) | **94.67%** | Monotonic compute scaling: $T=0.5$ (74.67%) $\to T=20.0$ (**96.33%**). True spectral bound $\lambda_{\max} \le \|A_{\text{norm}}\|_2 - d_{\min} = -0.387 < 0$. |
| **Midpoint Noise Perturbation** ($\Delta t = 5.0$) | Drops 100% $\to$ 72.33% (at 500% noise) | **92.67% – 95.83%** (Synthetic) / **92.67%** ($\sigma=10^{-7}$) $\to$ **50.00%** ($\sigma \ge 10^{-6}$) (Cora) | Exponential contraction decays perturbations by **286× – 307×** in synthetic regimes. On Cora (2,708 nodes), self-healing is refuted: a graded transition occurs at $\sigma \in [10^{-7}, 10^{-6}]$ as additive background noise fills the unreachable log-norm floor ($-16.40 > -17.5$), driving predictions to chance (50.0%). |
| **Hybrid Thought-to-Token** (Two-Stage Decoder) | — | **99.00%** Token Acc (100% Exact Sentences) | Autoregressively generates sentences from $\mathbf{z}^*$; monotonic test-time scaling: $T=0.5$ (75.0%) $\to T=10.0$ (**95.5%**). |
| **Demidovich Contraction Bound** | $\lambda_{\max} > 0$ (unconstrained) | $\lambda_{\max}(\text{Sym}(\mathbf{J})) \le \mathbf{-0.535}$ (Synthetic) / $\mathbf{-1.890}$ (Cora, empirical) | Strictly contractive: Cora analytic bound $\le -0.981 < 0$ and empirical 43,328-dim power iteration yields $\lambda_{\max}(\text{Sym}(J)) = -1.890 < 0$. |
| **Parity Generalization Audit** | 46.50% ($N=8$) / 48.00% ($N=32$) | 100% ($N=8$, recall) / 49.50% ($N=32$, chance) | Discloses $N=8$ memorization vs $N=32$ held-out boundary; documents ICNN potential expressivity limit on non-convex parity (`docs/mathematical_reference.md` §7). |

---

## Citation

```bibtex
@article{clr2026,
  title={Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Energy Contraction},
  author={Research Team},
  journal={Working Manuscript},
  year={2026}
}
```
