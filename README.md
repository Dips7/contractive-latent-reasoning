# Reviewer Replication Suite: Continuous Graph Latent Dynamics

**Manuscript**: *Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures*  
**Package Type**: Standalone Reviewer Replication Suite & Research Repository

---

## 1. Overview & Research Scope

This repository provides the complete, self-contained replication suite accompanying the manuscript. It contains the dataset generation logic, model definitions, training harnesses, and evaluation tools required to reproduce the synthetic benchmark suite, architectures, and loss certificates described in the paper.

### Research Contributions & Empirical Scope:
1. **Analytical Sensitivity Analysis**: Proves that under continuous-time Demidovich contraction ($\mu_2(J) \le -\alpha < 0$), the spatial block of the equilibrium input-sensitivity Jacobian decays exponentially as $\mathcal{O}\left(\frac{1}{\alpha}\left(\frac{\beta}{\beta+\alpha}\right)^k\right)$ along shortest paths of length $k$, induced by shortest-path nilpotency under elementwise activations.
2. **Audited Synthetic Benchmark Suite**: Formulates and audits 36 graph instances (7,230 reachability queries) across directed chains, bottlenecks, and acyclic digraphs with graph-instance-level test partitioning, zero replacement sampling, zero internal duplicates, and BFS-verified ground truth.
3. **Factorial Damping Sweep**: Evaluates 72 trained models across nominal damping parameters, demonstrating that continuous contractive models achieve 100.0% multi-hop recall on small bounded-degree graphs while test cross-entropy losses scale monotonically across nominal damping settings under the evaluated protocol.
4. **Multiscale Skip-Hop Contractive Architecture**: Introduces a multiscale architecture ($P_1 = P, P_2 = P^2$) satisfying uniform Demidovich contraction while accelerating the spatial influence bound to $\mathcal{O}(\rho^{\lceil k/2 \rceil})$, achieving 47.7% to 69.3% reductions in held-out test cross-entropy loss relative to single-scale baselines.

> [!IMPORTANT]
> **Replication Scope & Package Notice**:  
> The workflow and scripts in this repository are provided for independent reviewer replication. During package quality assurance (QA), **individual components were verified** in an isolated clean-extraction environment:
> 1. Complete regeneration of the 36-instance benchmark suite across 3 digraph families (`dataset_generator.py`).
> 2. Cryptographic SHA-256 verification of all 36 graph instances and the dataset manifest against research release digests (`verify_dataset_integrity.py`).
> 3. Independent audit of all 7,230 reachability queries and operator norm bounds ($\|P\|_2 \le 1.0$) against BFS ground truth (`audit_benchmark_data.py`).
> 4. Forward and backward gradient pass execution across all 5 model architectures (`models.py`).
>
> **Full retraining of all 117 models (72 damping sweep + 45 architectural comparison models) was not rerun during packaging QA.** The 117 model evaluations reported in the manuscript and supplement represent completed empirical runs from the original research study, audited under recorded loss certificates. Binary checkpoint files (`*.pt`) are omitted from the standalone archive to adhere to clean code release standards; executing Steps 5 and 6 retrains all models from scratch to regenerate checkpoints locally under `./results/*/checkpoints/`, which Step 7 then evaluates to generate held-out test loss certificates.

---

## 2. System Requirements & Environment

### Pinned Dependencies (`requirements.txt`):
- **Python**: `3.9.6` (CPython 3.9+)
- **PyTorch**: `torch==2.8.0`
- **NumPy**: `numpy==2.0.2`
- **NetworkX**: `networkx==3.2.1`

### Hardware Requirements & Expected Runtimes:
- **Compute**: Standard CPU or CUDA/MPS GPU (all default settings execute on standard multi-core CPU).
- **RAM**: $\ge 8$ GB.
- **Estimated Full Runtime**:
  - Steps 1–4 (Dataset generation, integrity check & audit): $< 30$ seconds.
  - Step 5 (72-model damping parameter sweep): $\approx 10$ minutes on an 8-core CPU.
  - Step 6 (45-model architectural comparison): $\approx 8$ minutes on an 8-core CPU.
  - Step 7 (Held-out test loss evaluation & certificate generation): $< 30$ seconds.

---

## 3. Directory Structure

```text
contractive-latent-reasoning/
├── README.md                                  # This replication guide
├── requirements.txt                           # Pinned dependencies (torch, networkx, numpy)
└── stability_investigation/                   # Core research investigation
    ├── reviewer_code_bundle/                  # Standalone self-contained reviewer replication suite
    │   ├── README.md                          # Standalone execution guide
    │   ├── requirements.txt                   # Environment specification
    │   ├── benchmark_manifest.json            # Dataset metadata & SHA-256 instance digests
    │   ├── dataset_generator.py               # Step 1: Benchmark suite generation
    │   ├── verify_dataset_integrity.py        # Step 2: Cryptographic dataset verification
    │   ├── audit_benchmark_data.py            # Step 3: Graph reachability & operator norm audit
    │   ├── models.py                          # Step 4: Model architecture definitions
    │   ├── run_damping_intervention.py        # Step 5: 72-model damping parameter sweep
    │   ├── run_architectural_comparison.py    # Step 6: 45-model architectural comparison
    │   └── evaluate_and_persist_test_loss.py  # Step 7: Test loss evaluation & certificates
    ├── data/                                  # Pre-generated benchmark datasets
    │   ├── benchmark_audit_certificate.json   # Full BFS ground truth audit certificate
    │   └── synthetic/                         # 36 benchmark digraph instances (.pt)
    ├── results/                               # Empirical results, checkpoints & certificates
    │   ├── damping_intervention/              # Damping sweep checkpoints & loss certificates
    │   └── architectural_comparison/          # Architectural comparison checkpoints & certificates
    ├── scripts/                               # Core execution and verification scripts
    ├── src/                                   # Extended model architectures and generators
    ├── audit/                                 # Provenance & baseline integrity certificates
    ├── ARCHITECTURAL_COMPARISON_REPORT.md     # 45-model comparative evaluation report
    ├── BENCHMARK_AUDIT_REPORT.md              # 36-instance dataset audit report
    ├── CLAIM_REGISTER.md                      # Formal empirical and theoretical claim registry
    ├── DAMPING_INTERVENTION_REPORT.md         # 72-model damping parameter sweep report
    ├── PROOF_OBLIGATIONS.md                   # Mathematical proof obligations & status
    ├── SENSITIVITY_ANALYSIS.md                # Resolvent series derivation & sensitivity bounds
    ├── STUDY_PROTOCOL.md                      # Pre-registered study protocol & methodology
    └── THEORY_NOTE.md                         # Contraction theory & spectral bounds
```

---

## 4. Step-by-Step Reviewer Reproduction Workflow

### Step 0: Clone & Install Dependencies
```bash
git clone -b stability-investigation https://github.com/Dips7/contractive-latent-reasoning.git
cd contractive-latent-reasoning
pip install -r requirements.txt
cd stability_investigation/reviewer_code_bundle
```

### Step 1: Regenerate the 36-Instance Benchmark Suite
Generates 36 graph instances across directed chains, bottlenecks, and acyclic digraphs:
```bash
python3 dataset_generator.py
```
*Expected Output*: Generated 36/36 graph files in `./data/synthetic/` and wrote `benchmark_manifest.json`.

### Step 2: Cryptographic SHA-256 Dataset Verification
Verifies instance byte-sizes and cryptographic SHA-256 digests:
```bash
python3 verify_dataset_integrity.py
```
*Expected Output*: `36/36 instance hashes matched. Verification PASSED.`

### Step 3: Independent Graph Reachability & Operator Norm Audit
Audits all 7,230 query reachability labels against NetworkX BFS and verifies $\|P\|_2 \le 1.0$:
```bash
python3 audit_benchmark_data.py
```
*Expected Output*: `BFS ground-truth verification: 7230/7230 queries matched (0 errors). Maximum spectral norm: 1.000000 <= 1.0. Audit PASSED.`

### Step 4: Model Forward & Backward Pass Verification
Verifies model instantiation and gradient backpropagation for all 5 architectures:
```bash
python3 -c "import torch; import models; m = models.MultiscaleSkipHop(1, 8, 2, 0.05, 1.5, 0.5, 0.5); x = torch.randn(10, 1); A = torch.eye(10); out = m(x, A); out.sum().backward(); print('Model check passed successfully.')"
```

### Step 5: Execute 72-Model Damping Parameter Sweep
Retrains 72 models across nominal damping settings ($d_{\min} \in \{0.50, 0.80, 1.05, 1.20, 1.50, 2.00\}$), undamped baselines, and discrete GNNs:
```bash
python3 run_damping_intervention.py
```

### Step 6: Execute 45-Model Architectural Comparison Study
Trains the 45-model architectural benchmark suite across all 3 topology families and 5 model configurations:
```bash
python3 run_architectural_comparison.py
```

### Step 7: Evaluate Held-Out Test Loss & Generate Certificates
Evaluates test cross-entropy loss across all trained models:
```bash
python3 evaluate_and_persist_test_loss.py
```
*Expected Output*: `test_loss_evaluation_certificate.json written. All test evaluations completed.`

---

## 5. Architectural Specifications

| Model Class | Vector Field Formulation | Damping Bound | Spatial Decay Rate |
| :--- | :--- | :--- | :--- |
| **Continuous Undamped** | $f(z) = \sigma(P z W) - z$ | $\mu_2(J) \le 0$ | Marginal ($\rho = 1.0$) |
| **Continuous Damped** | $f(z) = \sigma(P z W) - D z$ ($d_{\min} \ge 1.05$) | $\mu_2(J) \le -0.05$ | $\mathcal{O}(\rho^k)$, $\rho \approx 0.952$ |
| **Multiscale Skip-Hop** | $f(z) = \sigma(w_1 P z W_1 + w_2 P^2 z W_2) - D z$ | $\mu_2(J) \le -0.05$ | $\mathcal{O}(\rho^{\lceil k/2 \rceil})$, $\rho \approx 0.952$ |
| **Discrete GNN-6** | $h^{(l+1)} = \text{ReLU}(P h^{(l)} W)$ ($L=6$) | N/A (discrete) | Depth-bounded ($L=6$) |
