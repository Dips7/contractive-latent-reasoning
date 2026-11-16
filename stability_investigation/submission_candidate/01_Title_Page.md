# Title Page

**Manuscript Title**: Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures  
**Target Journal**: *Neurocomputing* (Elsevier)  
**Article Type**: Full Length Article (Original Research)

---

## Authors

**Dipesh Gurung**$^{a,*}$, **Binod Bhattarai**$^b$, **Prof. Dr. R.N. Thakur**$^a$

## Affiliations

$^a$ Research Department, Lord Buddha Education Foundation (LBEF Campus), Opposite to Maitidevi Temple, Kathmandu 44600, Nepal  
$^b$ Department of Computer Science and Engineering, School of Sciences, Noida International University (NIU), Greater Noida, Uttar Pradesh 203201, India  

## Corresponding Author Details

* **Full Name**: Dipesh Gurung, M.Sc.
* **Position / Role**: Research Associate
* **Department**: Research Department
* **Institution**: Lord Buddha Education Foundation (LBEF Campus)
* **Postal Address**: Opposite to Maitidevi Temple, Kathmandu 44600, Nepal
* **Primary Email**: `dipesh.gurung@lbef.edu.np`
* **Personal Email**: `dips.grg7@gmail.com`
* **Telephone**: +977-1-4424412
* **ORCID**: [0009-0009-0335-2267](https://orcid.org/0009-0009-0335-2267)
* **GitHub Profile**: [https://github.com/Dips7](https://github.com/Dips7)

## Co-Author Details

* **Binod Bhattarai** (Ph.D. Scholar)
  * Department of Computer Science and Engineering, School of Sciences, Noida International University (NIU), Greater Noida, Uttar Pradesh 203201, India
  * Email: `binod25@gmail.com`
  * ORCID: [0009-0006-6691-5264](https://orcid.org/0009-0006-6691-5264)
* **Prof. Dr. R.N. Thakur** (Dean, Academics)
  * Academics, Lord Buddha Education Foundation (LBEF Campus), Kathmandu 44600, Nepal
  * Email: `rn.thakur@lbef.edu.np`
  * ORCID: [0000-0003-3911-4358](https://orcid.org/0000-0003-3911-4358)

---

## Abstract

Continuous-depth graph neural ODEs offer guaranteed dynamical stability under Demidovich contraction. However, prior exploratory evaluations on transductive citation networks (CiteSeer and PubMed across path lengths 1–6)—subject to protocol limitations including query overlaps and unisolated initializations—showed contractive models collapsing to 0.00% positive reachability recall beyond path length 2, while a discrete baseline on CiteSeer maintained non-zero recall. We investigate the mechanisms and empirical properties governing long-range signal propagation in contractive graph vector fields. First, separating temporal contraction from spatial sensitivity, we derive an equilibrium sensitivity upper bound via an exact linearized Neumann resolvent identity $(I_{nd} - \mathcal{T})^{-1} = \sum_{m=0}^\infty \mathcal{T}^m$. We prove that the $(v, s)$ spatial block is bounded above by $\|\frac{\partial z^*_v}{\partial x_s}\|_2 \le \frac{1}{\alpha} \left(\frac{\beta}{\beta + \alpha}\right)^k$ via shortest-path nilpotency under elementwise activations. We emphasize that this formula establishes an asymptotic upper bound rather than a tight rate, a proof of vanishing classification accuracy, or an established failure cause on complex graphs. Second, on an audited synthetic benchmark suite of 36 graph instances and 7,230 query pairs across chains, bottlenecks, and acyclic digraphs with strict split controls, a descriptive parameter sweep across 72 models shows continuous contractive models achieving 100.0% recall across evaluated queries (with 6-hop queries supported on chains and bottlenecks), while seed-averaged losses scale monotonically with nominal damping without establishing causal mechanisms. Finally, we introduce a certified Multiscale Skip-Hop Contractive Architecture ($P, P^2$) that preserves Demidovich stability while reducing held-out test cross-entropy loss by 47.7% to 69.3% on these ceiling-saturated benchmarks.

## Keywords

Continuous-time Graph Neural Networks; Neural Ordinary Differential Equations; Demidovich Contraction; Dynamical Stability; Long-Range Graph Reachability; Over-Squashing

---

## Declarations Summary

* **Declaration of Competing Interests**: The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
* **Data and Code Availability**: Anonymized evaluation scripts and model source files are provided in the accompanying reviewer code archive (`anonymized_code_and_scripts.zip`). Extended proofs and full loss records are in `05_Supplementary_Material.pdf`. Benchmark datasets (7,230 queries across 36 graphs) and checkpoint certificates are audited under recorded SHA-256 digests. Upon acceptance, code and data will be permanently archived under a public repository DOI.
