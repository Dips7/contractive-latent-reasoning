# Submission Readiness Report: Neurocomputing Submission Package

**Manuscript Title**: Stability and Long-Range Information Propagation in Continuous-Time Graph Neural Dynamics: Sensitivity Analysis, Audited Benchmarks, and Certified Multiscale Architectures  
**Target Journal**: *Neurocomputing* (Elsevier)  
**Article Type**: Full Research Paper  
**Governing Branch**: `stability_investigation/`  
**Candidate Directory**: `stability_investigation/submission_candidate/`  
**Date**: November 2026  
**Final Submission Readiness Status**: **PRE-SUBMISSION CANDIDATE (AUTHOR REVIEW REQUIRED)**  
*(CRITICAL NOTICE: This package is NOT ready for direct journal submission. It represents an audited pre-submission staging package. Formal submission requires resolving author-input placeholders, author confirmations, and reviewing open proof obligation PO-C2.)*

---

## 1. Executive Summary & Audit Outcome

This report synthesizes the complete editorial, evidence-alignment, mathematical verification, automated layout inspection, and packaging process for the manuscript targeted at *Neurocomputing* (Elsevier). All review items have been systematically reconciled:

### Key Audit Reconciliations:
- **No Unsupported Significance Claims**: Removed any mention of "statistical significance" from `EVIDENCE_BOUNDARY_MATRIX.md` (CLM-06). Replaced with explicit descriptive characterization (seed-averaged cross-entropy loss reduction across $N=3$ seeds; no formal hypothesis test conducted).
- **Exact Abstract Synchronization**: Synchronized the abstract in `01_Title_Page.md` with `neurocomputing_manuscript.md` byte-for-byte. The abstract is **245 words** ($\le 250$ word limit) and contains all required caveats (citation protocol limitations, exact Neumann identity vs. spatial sensitivity upper bound, ceiling effect on synthetic digraphs, and descriptive damping scaling without causal claims).
- **Calibrated Cover Letter**: In `02_Cover_Letter.md`, replaced overstated phrasing ("resolve this question" $\to$ "investigate this question through formal sensitivity analysis, audited synthetic benchmarks with strict experimental controls, and stability-preserving architectural designs").
- **Accurately Scoped Highlights**: Revised Highlight 3 to explicitly name the benchmark and its ceiling effect: *"Small synthetic benchmarks show 100% multi-hop reachability recall at ceiling"* (77 characters, $\le 85$ character Elsevier limit).
- **Documented and Component-Checked Reviewer Code Archive (Study Runs vs. Clean-Room Audit)**: Packaged a standalone reviewer code archive (`anonymized_code_and_scripts.zip`, 10 files) with dependencies pinned to the reference execution environment (`Python 3.9.6`, `torch==2.8.0`, `numpy==2.0.2`, `networkx==3.2.1`). All script paths are anchored strictly to the archive root (`Path(__file__).resolve().parent`). To prevent reader confusion regarding study completion:
  - *Original Empirical Study*: The 72 damping sweep models and 45 architectural comparison models (totaling 117 model runs across 3 seeds) were fully executed and completed in the original investigation, producing the authoritative checkpoints, test losses, and cryptographic audit certificates (`test_loss_evaluation_certificate.json`, `damping_test_loss_certificate.json`) presented in Manuscript Table 1 and Supplementary Tables S1–S3.
  - *Clean-Room Package Testing*: During package QA, the reviewer archive was verified for component integrity—synthetic dataset regeneration, 36-instance cryptographic hash matching, BFS query/norm auditing, and forward/backward gradient passes across all 5 architectures. To maintain a lightweight source archive adhering to code submission standards, binary checkpoint weights (`*.pt`) are not bundled, and repeating the multi-epoch retraining of all 117 models was not re-executed during the temporary clean-room verification pass. The reviewer archive workflow is therefore classified as **documented and component-checked** rather than end-to-end verified.
- **Accurate 31-Page Automated Layout QA**: Documented the dual-layer inspection in `VISUAL_QA_RECORD.md` as **Automated PDF Layout and Structural QA**, providing programmatic bounding-box and margin measurements across **all 31 pages** (all 16 manuscript pages and all 15 supplement pages at 150 DPI), certifying zero horizontal clipping (side margins $\ge 60.5$ pt) and full 9-column table rendering.
- **Baseline Immutability**: All 25 frozen baseline artifacts verified unchanged via SHA-256 digests (Exit code: 0).
- **Candidate C Boundary**: Proof obligation PO-C2 (feature-only baseline) is prominently disclosed as **OPEN / UNMET**; node degree correlation with depth is disclosed; results are strictly quarantined from Candidate B.

---

## 2. Phase-by-Phase Verification Summary

- **Phase 0 (Provenance & Scope)**: Verified all 25 baseline files in `BASELINE_MANIFEST.json`. Generated `evidence_provenance_hashes.json` (SHA-256 digests of all 11 supporting evidence artifacts). Created isolated directory `submission_candidate/`.
- **Phase 1 (Evidence Boundary Freezing)**: Created `EVIDENCE_BOUNDARY_MATRIX.md` mapping claims CLM-01 through CLM-07. Replaced non-strict inequalities with "bounded above by" ($\le$). Established hard epistemic distinctions between exact resolvents and sensitivity bounds, continuous loss and ceiling metrics, and descriptive associations and causal attributions.
- **Phase 2 (Candidate C Resolution)**: Added prominent disclosure of open proof obligation PO-C2 in §4.3, §6.2 (Key Observation 4), Table 1, and Supplementary Section S4.1. Quarantined Candidate C from Candidate B.
- **Phase 3 (Manuscript Calibration)**: Calibrated title to "Audited Benchmarks", streamlined abstract to 245 words, de-causalized all section headings, and formatted as an anonymized peer-review draft.
- **Phase 4 (Mathematical & Numerical Audit)**: Reconciled $1/\alpha$ tail prefactor cancellation $\frac{1}{d_{\min}(1-\rho)} = \frac{1}{\alpha}$. Audited all Table 1 values against `test_loss_evaluation_certificate.json` and `damping_test_loss_certificate.json` with sample standard deviations ($ddof=1$).
- **Phase 5 (Standalone Submission Files)**: Authored `01_Title_Page.md`, `02_Cover_Letter.md`, `03_Highlights.md` ($\le 85$ chars/bullet), `05_Supplementary_Material.md` (34 KB extended proofs and full loss records), `06_Declarations.md`, and packaged self-contained `anonymized_code_and_scripts.zip`.
- **Phase 6 (Rendering & Packaging)**: Rendered standalone HTML preprints and publication PDFs (`neurocomputing_manuscript.pdf`, `05_Supplementary_Material.pdf`). Conducted 31-page automated layout and bounding-box inspection in `VISUAL_QA_RECORD.md`.
- **Phase 7 (Audit Reconcilement & Sign-Off)**: Package classified as `PRE-SUBMISSION CANDIDATE (AUTHOR REVIEW REQUIRED)` with explicit author action checklist.

---

## 3. Package File Manifest and Cryptographic Checksums

The table below reflects the exact cryptographic SHA-256 digests and file sizes for all items in `stability_investigation/submission_candidate/`:

| Filename | Type | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| `01_Title_Page.md` | Submission Meta | 4,722 | `d0d164de151f2769803676712fe0eaa341fc5c1d38cbbfce674e3d277bf20494` |
| `02_Cover_Letter.md` | Editorial Letter | 5,670 | `53f52bc85fd1d9213fbfeda8ebeb24c3d3d538000e1630be476795d83bdf483c` |
| `03_Highlights.md` | Journal Highlights | 577 | `ebad8d799edd5e8510ffa3f72933a241b92a460389e3a9826dfcf5fba3694e7a` |
| `05_Supplementary_Material.html` | Rendered Supplement HTML | 67,613 | `b7f1ef65930c56524f1c3826d3f71856581ef946dc20aec44b070397ac843947` |
| `05_Supplementary_Material.md` | Technical Supplement | 34,269 | `ad9371fce98f251970ff6a420e5c119f2cecf15aab81013cafd5a3883b263b36` |
| `05_Supplementary_Material.pdf` | Formatted Supplement PDF | 2,130,265 | `bab17e2b6cbc773c1905f5d1c2c6583998c15ff9fddb7612312f00008fda09e2` |
| `06_Declarations.md` | Legal & Ethical Declarations | 3,972 | `a62d1cbebaf65a3c80fcf594e416d311c85608dd28b247ab8869455839b69d9c` |
| EVIDENCE_BOUNDARY_MATRIX.md | Epistemic Matrix | 8,648 | `7c1f939c152dc7b1555cf814cb2d9e94b6340a3ce93f8310cb14018f5bab3c7e` |
| `SUBMISSION_READINESS_REPORT.md` | Readiness Audit Report | 12,772 | *(Recorded in package_manifest.json)* |
| `VISUAL_QA_RECORD.md` | 31-Page Automated Layout & Structural QA Record | 9,072 | `f44ae1492bc9df8b9668a5f035b217147c0be298ad685d700bd7b89efe8dbfc4` |
| `anonymized_code_and_scripts.zip` | Reviewer Code Archive (ZIP) | 27,584 | `d4975130906df17e79659a5713cbecb38de0e98b72ee3bad9466965bc27b61c7` |
| `evidence_provenance_hashes.json` | Provenance Audit Hashes | 2,123 | `f31dbac23974f9aa66c5e342fadfa8bd15c0b75aa4ecfc4aab9526e5b75e7148` |
| `neurocomputing_manuscript.html` | Rendered Manuscript HTML | 69,564 | `c1dbdf007ff5b6638c71dc4a9a6eb4ced38cc4b8009f9adf72baca65a649e564` |
| `neurocomputing_manuscript.md` | Blinded Manuscript | 43,968 | `2e3fb39f7f98873d785e1676fc1a777df7bae9781f6907dd4e41411cb7a10689` |
| `neurocomputing_manuscript.pdf` | Formatted Manuscript PDF | 1,902,756 | `212cd7d53079e90bad8787a1924537403746d692274e8f7e1cb495c3f47eb08e` |
| `package_manifest.json` | Package Cryptographic Index | 2,564 | *(Root JSON manifest indexing all 15 candidate files)* |

---

## 4. Automated Layout and Structural QA Summary

As documented in `VISUAL_QA_RECORD.md`:
- **Audited Scope**: Complete programmatic layout and structural validation covering all 31 rendered pages (16 manuscript + 15 supplement).
- **Raster Printable Bounding Boxes (150 DPI)**:
  - *Manuscript PDF (`neurocomputing_manuscript.pdf`)*: 16 pages, 612.0 x 792.0 pt. Left margins range from 88.8 to 93.6 pt; right margins from 60.5 to 94.6 pt. Minimum horizontal clearance is **60.5 pt** (0.84 in), certifying zero text or table clipping outside printable boundaries.
  - *Supplementary PDF (`05_Supplementary_Material.pdf`)*: 15 pages, 612.0 x 792.0 pt. Left margins range from 105.1 to 105.6 pt; right margins from 85.4 to 109.9 pt. Minimum clearance is **85.4 pt** (1.18 in).
- **Structural Integrity**: Table 1 renders across pages 13-14 with all 9 columns intact. Tables S1–S3 in the supplement render complete numerical loss records with aligned sample standard deviations. Mathematical display equations are fully vectorized with MathJax fonts and zero raw LaTeX artifacts.

---

## 5. Author Action Checklist Prior to Formal Submission

The submitting authors review the following items before uploading to Elsevier Editorial Manager:

1. **Author Identification & Institutional Metadata**:
   - [x] In `01_Title_Page.md`: Fully populated with verified author directory records for Dipesh Gurung (Corresponding Author), Binod Bhattarai, and Prof. Dr. R.N. Thakur.
   - [x] In `02_Cover_Letter.md`: Signed by corresponding author Dipesh Gurung on behalf of all co-authors.
2. **Declarations & CRediT Attribution**:
   - [x] In `06_Declarations.md` (§1): Conflict of interest declaration formulated.
   - [x] In `06_Declarations.md` (§2): Specific CRediT contributor roles assigned and populated.
   - [x] In `06_Declarations.md` (§3): Public repository URL assigned (`https://github.com/Dips7/contractive-latent-reasoning`).
   - [x] In `06_Declarations.md` (§4): Non-funding statement confirmed (Option A).
   - [x] In `06_Declarations.md` (§5): Institutional facilities acknowledgments populated (LBEF Campus & NIU).
   - [ ] In `06_Declarations.md` (§6): Confirm manuscript originality and co-author approval in Elsevier Editorial Manager during upload.
3. **Open Proof Obligation PO-C2 Decision**:
   - Proof obligation PO-C2 (evaluating an isolated feature-only classification baseline without ODE latent states) remains OPEN / UNMET. The current package transparently discloses this as an exploratory readout caveat. If authors wish to make affirmative claims for degree features, PO-C2 must be executed prior to submission; otherwise, the current qualified disclosure must be retained.
4. **Journal File Upload Mapping**:
   - *Title Page with Author Details*: Upload `01_Title_Page.md` (or converted docx/pdf).
   - *Blind / Anonymized Manuscript*: Upload `neurocomputing_manuscript.pdf`.
   - *Highlights*: Upload `03_Highlights.md`.
   - *Cover Letter*: Upload `02_Cover_Letter.md`.
   - *Supplementary Material*: Upload `05_Supplementary_Material.pdf`.
   - *Reviewer Code Archive*: Upload `anonymized_code_and_scripts.zip`.
   - *Declarations*: Upload `06_Declarations.md`.

---

## 6. Certification Sign-Off

**Audit Verification**: 100% Passed  
**Baseline Contamination**: 0.0% (25/25 baseline files verified unchanged)  
**Package State**: Staged and Audited  
**Submission Status**: **PRE-SUBMISSION CANDIDATE (AUTHOR REVIEW REQUIRED)**
