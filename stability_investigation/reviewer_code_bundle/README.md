# Reviewer Replication Suite: Continuous Graph Latent Dynamics (Documented & Component-Checked)

**Manuscript Title**: Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures  
**Target Journal**: *Neurocomputing* (Elsevier)  
**Package Type**: Standalone Anonymized Reviewer Code & Replication Archive

---

## 1. Overview, Scope & System Requirements

This archive provides the standalone, self-contained Python replication suite accompanying the manuscript. It contains the complete source code, dataset generation logic, model definitions, training harnesses, and evaluation tools required to reproduce the synthetic benchmark suite, architectures, and loss certificates described in the paper. All scripts resolve paths strictly relative to the archive directory (`Path(__file__).resolve().parent`), supporting execution in any extracted location.

> [!IMPORTANT]
> **Replication Scope & Verification Notice**:  
> The workflow and scripts in this archive are provided for independent replication. During package quality assurance (QA), **individual components were verified** in an isolated clean-extraction environment—specifically:
> 1. Complete regeneration of the 36-instance benchmark suite across 3 digraph families (`dataset_generator.py`).
> 2. Cryptographic SHA-256 verification of all 36 graph instances and the dataset manifest against research release digests (`verify_dataset_integrity.py`).
> 3. Independent audit of all 7,230 reachability queries and operator norm bounds ($\|P\|_2 \le 1.0$) against BFS ground truth (`audit_benchmark_data.py`).
> 4. Forward and backward gradient pass execution across all 5 model architectures (`models.py`).
>
> **Full end-to-end retraining of all 117 models (72 damping sweep + 45 architectural comparison models) was not rerun during package QA.** The 117 model evaluations reported in the manuscript and supplement (Table 1, Tables S1–S3) represent completed empirical runs from the original research study, audited under recorded loss certificates. Binary checkpoint files (`*.pt`) are omitted from this archive to adhere to clean code release standards and journal upload limits; executing Steps 5 and 6 retrains all models from scratch to regenerate checkpoints locally under `./results/*/checkpoints/`, which Step 7 then evaluates to generate held-out test loss certificates.

### Reference Execution Environment (Pinned Dependencies):
- **Python**: `3.9.6` (Apple Clang / CPython 3.9+)
- **PyTorch**: `torch==2.8.0`
- **NumPy**: `numpy==2.0.2`
- **NetworkX**: `networkx==3.2.1`

### Hardware Requirements & Expected Runtimes:
- **Compute**: Standard CPU or CUDA/MPS GPU (all default settings execute on a standard multi-core CPU).
- **RAM**: $\ge 8$ GB.
- **Estimated Full Runtime**:
  - Step 1–4 (Dataset generation, integrity check & audit): $< 30$ seconds.
  - Step 5 (Full 72-model damping parameter sweep): $\approx 10$ minutes on 8-core CPU.
  - Step 6 (Full 45-model architectural comparison): $\approx 8$ minutes on 8-core CPU.
  - Step 7 (Held-out test loss evaluation & certificate generation): $< 30$ seconds.

---

## 2. Archive Directory Layout

When unzipped, this archive contains the following self-contained source files:

```text
anonymized_code_and_scripts/
├── README.md                           <- This complete replication guide
├── requirements.txt                    <- Pinned dependency specification
├── benchmark_manifest.json             <- Ground-truth dataset manifest (SHA-256: 8d5b02c5...)
├── models.py                           <- Consolidated model architectures (ODE, Multiscale, Readout, Discrete)
├── dataset_generator.py                <- Synthetic benchmark dataset generation script
├── verify_dataset_integrity.py         <- Cryptographic dataset integrity verifier
├── audit_benchmark_data.py             <- Automated query and operator integrity auditor
├── run_damping_intervention.py         <- 72-model descriptive damping sweep training harness
├── run_architectural_comparison.py     <- 45-model architectural comparison training harness
└── evaluate_and_persist_test_loss.py   <- Checkpoint test loss evaluation & certificate generator
```

---

## 3. Step-by-Step Replication Workflow

Follow these steps in sequence to reproduce all data, models, and results from scratch:

### Step 1: Environment Installation
Install the required packages in your preferred virtual environment:
```bash
pip install -r requirements.txt
```

### Step 2: Regenerate the Audited Synthetic Benchmark Suite
Generate the 36 synthetic graph instances across `chain_ladder`, `bottleneck`, and `random_dag` with strict multi-instance partitioning (5 Train, 2 Val, 5 Test per family):
```bash
python3 dataset_generator.py
```
- **Output Directory**: `./data/synthetic/{chain_ladder, bottleneck, random_dag}/`
- **Generated Manifest**: `./data/synthetic/benchmark_manifest.json` and `./benchmark_manifest.json`

### Step 3: Verify Dataset Cryptographic Integrity
Validate all 36 generated graph files against the recorded research release SHA-256 digests:
```bash
python3 verify_dataset_integrity.py
```
- **Expected Console Output**: `VERIFICATION SUMMARY: 36/36 graph instance files matched. OVERALL DATASET INTEGRITY: PASSED (100% BYTE-FOR-BYTE MATCH)`.

### Step 4: Run the Automated Dataset & Query Audit
Validate all 7,230 query pairs for 100% agreement with BFS ground truth, zero duplicate queries, zero split overlap, and operator spectral norm bounds ($\|P\|_2 \le 1.0$):
```bash
python3 audit_benchmark_data.py
```
- **Expected Console Output**: `Audit completed: Status = PASSED | Total queries: 7230`.
- **Generated Report**: `./BENCHMARK_AUDIT_REPORT.md`
- **Generated Certificate**: `./benchmark_audit_certificate.json`

### Step 5: Train the 72-Model Descriptive Damping Parameter Sweep
Train the 72 models (3 families $\times$ 8 configurations $\times$ 3 PRNG seeds: 42, 101, 777) across nominal damping settings $d_{\min} \in \{0.50, 0.80, 1.05, 1.20, 1.50, 2.00\}$, undamped ODE, and discrete GNN6:
```bash
python3 run_damping_intervention.py
```
- **Checkpoints Saved To**: `./results/damping_intervention/checkpoints/*.pt`
- **Audit JSON Saved To**: `./results/damping_intervention/damping_intervention_results.json`

### Step 6: Train the 45-Model Architectural Comparison Sweep
Train the 45 models (3 families $\times$ 5 architectures $\times$ 3 PRNG seeds) benchmarking the baseline damped ODE, Candidate A (margin tuned), Candidate B (multiscale skip-hop), Candidate C (degree readout), and discrete GNN6:
```bash
python3 run_architectural_comparison.py
```
- **Checkpoints Saved To**: `./results/architectural_comparison/checkpoints/*.pt`
- **Audit JSON Saved To**: `./results/architectural_comparison/architectural_comparison_results.json`

### Step 7: Evaluate Held-Out Test Loss and Generate Certificates
Evaluate all saved model checkpoints on held-out test graph instances and generate reproducible loss certificates:
```bash
python3 evaluate_and_persist_test_loss.py
```
- **Output Certificate 1**: `./results/architectural_comparison/test_loss_evaluation_certificate.json`
- **Output Certificate 2**: `./results/damping_intervention/damping_test_loss_certificate.json`
- **Verifies**:
  - `chain_ladder`: Baseline ($0.001051$) $\to$ Candidate B ($0.000494$, -53.0%)
  - `bottleneck`: Baseline ($0.001366$) $\to$ Candidate B ($0.000715$, -47.7%)
  - `random_dag`: Baseline ($0.008138$) $\to$ Candidate B ($0.002501$, -69.3%)
  - Confirms 100.0% test classification accuracy, specificity, and recall across all models (benchmark ceiling effect).

---

## 4. Key Implementation Details in `models.py`

- **`ContinuousGraphODE`**: Single-scale contractive graph ODE with SVD weight projections ($\|W_1\|_2 \le 1.0, \|W_2\|_2 \le 1.0$) and diagonal coordinate damping ($D \ge d_{\min} I_d > 1.0$). Solved via explicit RK4 (`explicit_rk4_step`) over $T=4.0$ ($h=0.25$).
- **`MultiscaleContractiveODE` (Candidate B)**: Certified multiscale vector field integrating $P$ and $P^2$ with convex combination $\gamma_1 + \gamma_2 = 1.0$, satisfying uniform Demidovich stability under identical damping bounds ($\mu_2(J) \le 1.0 - d_{\min} = -\alpha < 0$).
- **`StructuralReadoutODE` (Candidate C)**: Exploratory structural baseline appending $[\ln(1 + d_{\text{out}}(s)), \ln(1 + d_{\text{in}}(t))]$ to readout MLP (subject to disclosed open obligation PO-C2).
- **`DiscreteGNN`**: 6-layer discrete message-passing baseline with bipartite normalized propagation operator $P = N^\top$.
