# Cover Letter

> [!IMPORTANT]
> **AUTHOR CONFIRMATION REQUIRED BEFORE FORMAL SUBMISSION**  
> This cover letter is a pre-submission template. Submitting authors must confirm all statements of originality, absence of competing interests, institutional approval, and sign the letter before transmitting to *Neurocomputing*.

To:  
The Editors-in-Chief and Editorial Board  
*Neurocomputing* (Elsevier)

**Date**: November 2026  
**Subject**: Submission of Original Research Article: *"Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures"*

Dear Editors,

We are pleased to submit our original research manuscript entitled **"Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures"** for consideration as a Full Research Paper in *Neurocomputing*.

### Background & Research Problem
Continuous-depth graph neural networks parameterized by ordinary differential equations (Neural ODEs) provide an elegant mathematical framework for relational learning, offering rigorous dynamical stability guarantees via Demidovich contraction theory. Despite these theoretical guarantees, earlier exploratory studies on transductive citation networks observed an unexpected empirical collapse in positive classification recall on reachability queries beyond path length 2. This observation raised a fundamental question for the continuous graph machine learning community: *Does enforcing continuous-time contraction inherently preclude long-range relational information propagation?*

### Key Contributions & Calibrated Epistemic Scope
In this manuscript, we investigate this question through formal sensitivity analysis, audited synthetic benchmarks with strict experimental controls, and stability-preserving architectural designs:

1. **Analytical Sensitivity Operator & Shortest-Path Nilpotency**: We clearly separate temporal trajectory contraction ($\mu_2(J) \le -\alpha$) from spatial equilibrium sensitivity ($J_{\text{eq}} = \partial z^* / \partial x$). Leveraging the exact Neumann series identity for the linearized equilibrium resolvent $(I_{nd} - \mathcal{T})^{-1} = \sum_{m=0}^\infty \mathcal{T}^m$, we prove that the $(v, s)$ spatial block at directed distance $k$ is bounded above by $\|\frac{\partial z^*_v}{\partial x_s}\|_2 \le \frac{1}{\alpha}(\frac{\beta}{\beta+\alpha})^k$ via shortest-path nilpotency under elementwise activations. We explicitly distinguish this asymptotic upper bound on linearized equilibrium sensitivity from empirical representation decay or classification failure.
2. **Audited Synthetic Benchmark Suite (7,230 Queries across 36 Digraphs)**: To evaluate continuous dynamics under simplified, known graph structures with strict experimental controls, we formulate and audit a synthetic benchmark suite across directed chains, bottlenecks, and layered DAGs, enforcing multi-instance test partitioning (zero query pair overlap), zero replacement sampling, and BFS-verified ground truth.
3. **Descriptive Parameter Sweep & Benchmark Ceiling**: Across 72 trained models sweeping nominal damping settings, continuous contractive models achieve **100.0% recall across all evaluated queries** (with 6-hop queries supported on chains and bottlenecks), while seed-averaged validation and test losses scale monotonically with nominal damping. We report this association descriptively under the evaluated protocol, avoiding ungrounded causal claims and explicitly disclosing that the benchmark's classification metrics are at ceiling across all evaluated models.
4. **Certified Multiscale Skip-Hop Architecture**: We introduce a multiscale continuous vector field integrating $P$ and $P^2$. We prove that it satisfies uniform Demidovich contraction under identical damping bounds ($\mu_2(J) \le -\alpha < 0$) while accelerating the spatial influence upper bound to $\mathcal{O}(\rho^{\lceil k/2 \rceil})$. Across the benchmark suite, the multiscale architecture achieves 47.7% to 69.3% reductions in held-out test cross-entropy loss relative to the single-scale baseline.

### Compliance and Disclosures
In accordance with *Neurocomputing* author guidelines:
- This manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere.
- Its publication is approved by all authors and tacitly or explicitly by the responsible authorities where the work was carried out.
- The manuscript adheres to double-blind peer review formatting, with all identifying author details and affiliations placed in the separate Title Page (`01_Title_Page.md`).
- We have prepared standalone files including Research Highlights (`03_Highlights.md`, $\le 85$ characters per bullet), Declarations (`06_Declarations.md`), an anonymized Reviewer Code Archive (`anonymized_code_and_scripts.zip`), and comprehensive Supplementary Material (`05_Supplementary_Material.md`) containing full proofs and audited per-run loss certificates.
- All code, datasets, and checkpoint SHA-256 digests are fully preserved and traceable.

Thank you for your time, consideration, and management of the peer-review process.

Sincerely,

**Dipesh Gurung**, M.Sc. (Corresponding Author)  
Research Associate, Research Department  
Lord Buddha Education Foundation (LBEF Campus)  
Opposite to Maitidevi Temple, Kathmandu 44600, Nepal  
Email: `dipesh.gurung@lbef.edu.np` / `dips.grg7@gmail.com`  
Telephone: +977-1-4424412  
*On behalf of the co-authors*: Binod Bhattarai and Prof. Dr. R.N. Thakur
