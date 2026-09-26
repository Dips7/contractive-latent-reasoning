"""
Build Publication Submission Package for:
"Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction"

Authors:
- Dipesh Gurung (Corresponding Author: dipesh.gurung@lbef.edu.np)
- Binod Bhattarai (binod25@gmail.com)
- Dr. R N Thakur (rn.thakur@lbef.edu.np)

Outputs generated in submission_package/:
1. 01_Cover_Letter.docx
2. 02_Title_Page.docx
3. 03_Highlights.docx
4. 04_Manuscript_Full.docx (with native OMML block & inline equations and high-res figures)
"""

import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree
from latex2mathml.converter import convert as l2m
import mathml2omml

BASE_DIR = Path(__file__).resolve().parents[2]
PKG_DIR = BASE_DIR / "submission_package"
FIG_DIR = BASE_DIR / "paper" / "figures"
PKG_DIR.mkdir(parents=True, exist_ok=True)

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def clean_latex_fallback(latex_expr: str) -> str:
    """Fallback text transformation converting LaTeX math syntax to clean Unicode."""
    replacements = [
        (r'\mathbf{z}_v', '𝐳_v'),
        (r'\mathbf{z}', '𝐳'),
        (r'\mathbf{x}', '𝐱'),
        (r'\mathbf{h}_v', '𝐡_v'),
        (r'\mathbf{h}', '𝐡'),
        (r'\mathbf{J}', '𝐉'),
        (r'\mathbf{D}', '𝐃'),
        (r'\mathbf{W}_1', '𝐖₁'),
        (r'\mathbf{W}_2', '𝐖₂'),
        (r'\mathbf{W}', '𝐖'),
        (r'\mathbf{A}_{\text{norm}}', '𝐀_norm'),
        (r'\mathbf{A}', '𝐀'),
        (r'\mathbf{s}', '𝐬'),
        (r'\mathbf{I}', '𝐈'),
        (r'\mathbf{0}', '𝟎'),
        (r'\delta \mathbf{z}', 'δ𝐳'),
        (r'\delta', 'δ'),
        (r'\lambda_{\max}', 'λ_max'),
        (r'\lambda', 'λ'),
        (r'\kappa', 'κ'),
        (r'\sigma', 'σ'),
        (r'\epsilon_t', 'ε_t'),
        (r'\epsilon', 'ε'),
        (r'\theta', 'θ'),
        (r'\alpha', 'α'),
        (r'\Phi_t', 'Φ_t'),
        (r'\mathbb{R}^d', 'ℝᵈ'),
        (r'\mathbb{R}', 'ℝ'),
        (r'\text{Sym}', 'Sym'),
        (r'\text{diag}', 'diag'),
        (r'\text{norm}', 'norm'),
        (r'\text{mid}', 'mid'),
        (r'\le', '≤'),
        (r'\ge', '≥'),
        (r'\approx', '≈'),
        (r'\ne', '≠'),
        (r'\cdot', '·'),
        (r'\times', '×'),
        (r'\nabla_\mathbf{z}^2', '∇_𝐳²'),
        (r'\nabla_\mathbf{z}', '∇_𝐳'),
        (r'\nabla', '∇'),
        (r'\succeq', '⪰'),
        (r'\preceq', '⪯'),
        (r'\implies', '⟹'),
        (r'\in', '∈'),
        (r'\to', '→'),
        (r'\forall', '∀'),
        (r'\|', '‖'),
        (r'\{', '{'),
        (r'\}', '}'),
        (r'\dots', '…'),
        (r'\quad', '  '),
        (r'10^{-12}', '10⁻¹²'),
        (r'10^{-8}', '10⁻⁸'),
        (r'10^{-7}', '10⁻⁷'),
        (r'10^{-6}', '10⁻⁶'),
        (r'10^{-4}', '10⁻⁴'),
        (r'10^{-2}', '10⁻²'),
        (r'2^{N-1}', '2ᴺ⁻¹'),
    ]
    s = latex_expr
    for src, dst in replacements:
        s = s.replace(src, dst)
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathbb\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\math[a-zA-Z]+\{([^}]+)\}', r'\1', s)
    s = s.replace('\\', '')
    return s


def latex_to_inline_omml(latex_code: str) -> etree._Element:
    """Converts a LaTeX inline equation to an OpenXML OMML inline element."""
    mathml = l2m(latex_code)
    omml = mathml2omml.convert(mathml)
    if omml.startswith("<m:oMath>"):
        wrapped = omml.replace("<m:oMath>", f'<m:oMath xmlns:m="{MATH_NS}">', 1)
    else:
        wrapped = f'<m:oMath xmlns:m="{MATH_NS}">{omml}</m:oMath>'
    return etree.fromstring(wrapped)


def latex_to_omml_para(latex_code: str) -> etree._Element:
    """Converts a LaTeX display equation to an OpenXML OMML paragraph element."""
    mathml = l2m(latex_code)
    omml = mathml2omml.convert(mathml)
    wrapped = f'<m:oMathPara xmlns:m="{MATH_NS}">{omml}</m:oMathPara>'
    return etree.fromstring(wrapped)


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Sets specific borders on a table cell using OpenXML."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    borders = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    for border_name, border_style in borders.items():
        if border_style:
            b = OxmlElement(f'w:{border_name}')
            b.set(qn('w:val'), border_style.get('val', 'single'))
            b.set(qn('w:sz'), str(border_style.get('sz', 4)))
            b.set(qn('w:space'), '0')
            b.set(qn('w:color'), border_style.get('color', 'auto'))
            tcBorders.append(b)
        else:
            b = OxmlElement(f'w:{border_name}')
            b.set(qn('w:val'), 'none')
            tcBorders.append(b)
    tcPr.append(tcBorders)


def style_document(doc):
    """Applies clean academic styling (Times New Roman, 1-inch margins)."""
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(11.5)
    font.color.rgb = RGBColor(0x11, 0x11, 0x11)


def add_p(doc, text="", bold_prefix=None, space_after=6, space_before=0, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT):
    """
    Adds a styled paragraph with automated inline LaTeX-to-OMML parsing.
    Text enclosed in $...$ is dynamically converted to native Word OMML math objects.
    """
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.bold = True
        r_bold.font.name = 'Times New Roman'
        r_bold.font.size = Pt(11.5)
        
    if text:
        tokens = re.split(r'(\$[^\$]+\$)', text)
        for token in tokens:
            if token.startswith('$') and token.endswith('$'):
                latex_expr = token[1:-1].strip()
                try:
                    omml_elem = latex_to_inline_omml(latex_expr)
                    p._p.append(omml_elem)
                except Exception as e:
                    clean_text = clean_latex_fallback(latex_expr)
                    r_fallback = p.add_run(clean_text)
                    r_fallback.font.name = 'Times New Roman'
                    r_fallback.font.size = Pt(11.5)
                    r_fallback.italic = True
            else:
                if token:
                    r_text = p.add_run(token)
                    r_text.font.name = 'Times New Roman'
                    r_text.font.size = Pt(11.5)
                    if italic:
                        r_text.italic = True
    return p


def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12.5)
    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    return p


def add_h3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11.5)
    return p


def add_eq(doc, latex_code):
    """Appends an OMML display equation as a centered native Word equation."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    elem = latex_to_omml_para(latex_code)
    p._p.append(elem)
    return p


# -----------------------------------------------------------------------------
# 1. BUILD COVER LETTER
# -----------------------------------------------------------------------------
def build_cover_letter():
    doc = docx.Document()
    style_document(doc)

    add_p(doc, "September 26, 2026", space_after=12)
    add_p(doc, "To:\nThe Editor-in-Chief,\nJournal Submission Office", space_after=12)

    add_p(doc, "Subject: Submission of Original Research Article titled \"Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction\"", bold_prefix="Dear Editor-in-Chief,\n\n", space_after=10)

    add_p(doc, "We are pleased to submit our original research article titled \"Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction\" for consideration for publication in your prestigious journal.")

    add_p(doc, r"This work addresses a primary bottleneck in modern reasoning architectures: the error compounding, quadratic memory overhead, and runaway drift inherent to discrete autoregressive Chain-of-Thought (CoT) token generation. We introduce Contractive Latent Dynamical Reasoning (CLR), a continuous-time framework that formulates multi-step reasoning as an algebraically constrained dynamical system evolving in continuous latent space. By establishing strict Demidovich contraction ($\lambda_{\max}(\text{Sym}(J)) \le -\kappa < 0$), CLR mathematically guarantees exponential convergence to a unique problem-conditioned equilibrium.")

    add_p(doc, "Key scientific and empirical findings presented in this manuscript include:")
    
    add_p(doc, "1. Formal Expressivity Boundary: We prove why damped Input-Convex Neural Network (ICNN) potential flows fail on non-convex combinatorial parity (verifying our pre-registered kill criterion at chance), and demonstrate how spectrally normalized non-potential flows resolve this tension.", space_after=4)
    add_p(doc, r"2. Real-World Benchmark Superiority: On the Cora citation network (2,708 papers, 5,429 citations), CLR achieves 100.00% multi-hop reachability, decisively outperforming discrete Graph Neural Networks ($K \in \{2, 4, 6, 8\}$) where discrete depth saturation and over-smoothing limit accuracy to 87.67% ($K=6$) and 86.67% ($K=8$).", space_after=4)
    add_p(doc, "3. Physical Refutation of Self-Healing: Under a unified numerical harness, we refute naive claims of 'self-healing' under global additive noise on graphs, revealing the exact log-norm energy floor mechanism responsible for threshold crossing.", space_after=4)
    add_p(doc, "4. Large-Scale Numerical Verification: We design an autograd-based shifted power iteration that verifies Demidovich contraction across all 43,328 dimensions in 0.22 seconds.", space_after=10)

    add_p(doc, "This manuscript represents original work and is not currently under consideration for publication elsewhere. All authors (Dipesh Gurung, Binod Bhattarai, and Dr. R N Thakur) have read, contributed to, and approved the final submitted version. The complete submission package comprises the Cover Letter, Title Page, Highlights, Full Manuscript, and comprehensive Supplementary Material document detailing mathematical proofs, the 43,328-dimension shifted power iteration algorithm, complete hyperparameter configurations, and zero-leakage dataset protocols. Source code, test suites (24 passed unit tests), and model checkpoints are publicly available at: https://github.com/Dips7/contractive-latent-reasoning.")

    add_p(doc, "Thank you very much for your time, consideration, and handling of our manuscript. We look forward to hearing from you.")

    add_p(doc, "Sincerely,\n\nDipesh Gurung (Corresponding Author)\nDepartment of Information Technology\nLord Buddha Education Foundation, Kathmandu 44600, Nepal\nEmail: dipesh.gurung@lbef.edu.np | Tel: +977-1-4424412\nORCID: 0009-0009-0335-2267", space_after=0)

    out_path = PKG_DIR / "01_Cover_Letter.docx"
    doc.save(out_path)
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# 2. BUILD TITLE PAGE
# -----------------------------------------------------------------------------
def build_title_page():
    doc = docx.Document()
    style_document(doc)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(18)
    r_title = p_title.add_run("Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.name = "Times New Roman"

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(8)
    
    r1 = p_auth.add_run("Dipesh Gurung")
    r1.bold = True
    r1.font.name = "Times New Roman"
    p_auth.add_run("1,*,\t").font.name = "Times New Roman"
    
    r2 = p_auth.add_run("Binod Bhattarai")
    r2.bold = True
    r2.font.name = "Times New Roman"
    p_auth.add_run("2,\t").font.name = "Times New Roman"
    
    r3 = p_auth.add_run("Dr. R N Thakur")
    r3.bold = True
    r3.font.name = "Times New Roman"
    p_auth.add_run("1").font.name = "Times New Roman"

    # Affiliations
    add_p(doc, "1 Department of Information Technology, Lord Buddha Education Foundation (LBEF Campus), Kathmandu 44600, Nepal\n2 Department of Computer Science and Engineering, School of Engineering and Technology, Noida International University, Greater Noida, Uttar Pradesh 203201, India\n* Corresponding Author: dipesh.gurung@lbef.edu.np", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)

    # Corresponding Author Box
    add_h2(doc, "Corresponding Author Details")
    add_p(doc, "Dipesh Gurung", bold_prefix="Full Name: ", space_after=2)
    add_p(doc, "Assistant Professor & Researcher", bold_prefix="Academic Position: ", space_after=2)
    add_p(doc, "Department of Information Technology, Lord Buddha Education Foundation (LBEF Campus)", bold_prefix="Department / Faculty: ", space_after=2)
    add_p(doc, "Opposite to Maitidevi Temple, Kathmandu 44600, Nepal", bold_prefix="Postal Address: ", space_after=2)
    add_p(doc, "dipesh.gurung@lbef.edu.np (Primary) / dips.grg7@gmail.com", bold_prefix="E-mail: ", space_after=2)
    add_p(doc, "+977-1-4424412", bold_prefix="Telephone: ", space_after=2)
    add_p(doc, "0009-0009-0335-2267", bold_prefix="ORCID: ", space_after=14)

    # Co-Author Details
    add_h2(doc, "Co-Author Credentials & ORCIDs")
    add_p(doc, "Binod Bhattarai, Assistant Professor & Ph.D. Candidate, Noida International University, India. Email: binod25@gmail.com. ORCID: 0009-0006-6691-5264.", bold_prefix="Binod Bhattarai: ", space_after=3)
    add_p(doc, "Prof. Dr. R.N. Thakur, Professor & Dean, Lord Buddha Education Foundation, Kathmandu, Nepal. Email: rn.thakur@lbef.edu.np.", bold_prefix="Dr. R N Thakur: ", space_after=14)

    # Abstract (exact 1 paragraph, ~196 words)
    add_h2(doc, "Abstract")
    abstract_text = (
        r"Scaling test-time reasoning in neural architectures predominantly relies on discrete autoregressive Chain-of-Thought rollouts, "
        r"which suffer from compounding drift and unbounded context consumption in open-ended domains. We propose Contractive Latent Dynamical "
        r"Reasoning (CLR), a continuous-time framework formulating multi-step reasoning as a strictly contractive dynamical system evolving in "
        r"continuous latent space. By constraining the vector field's symmetric Jacobian under Demidovich contraction ($\lambda_{\max}(\text{Sym}(J)) \le -\kappa < 0$), "
        r"CLR guarantees exponential convergence to a unique problem-conditioned equilibrium, eliminating runaway hallucinations while enabling "
        r"monotonic continuous test-time scaling. Investigating the expressivity-contraction boundary, we prove that while damped input-convex "
        r"potential flows minimize a convex surrogate and fail on non-convex combinatorial parity (meeting our pre-registered kill criterion at "
        r"chance), spectrally bounded non-potential neural vector fields provide expressive relational routing while preserving Demidovich "
        r"contraction. On the Cora citation network (2,708 papers, 5,429 citations), CLR achieves 100.00% multi-hop reachability with robust "
        r"decision margins, outperforming discrete Graph Neural Networks where depth saturation and over-smoothing limit accuracy to 87.67% ($K=6$) "
        r"and 86.67% ($K=8$). Furthermore, we refute naive self-healing on graphs by demonstrating how global noise fills the near-zero energy floor "
        r"of unreachable nodes, and empirically verify Demidovich contraction across 43,328 dimensions."
    )
    add_p(doc, abstract_text, space_after=12)

    # Keywords (5-7)
    add_h2(doc, "Keywords")
    add_p(doc, "Contractive Dynamical Systems; Latent Space Reasoning; Demidovich Contraction; Neural Ordinary Differential Equations; Test-Time Compute Scaling; Graph Neural Networks; Non-Convex Optimization.", space_after=12)

    out_path = PKG_DIR / "02_Title_Page.docx"
    doc.save(out_path)
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# 3. BUILD HIGHLIGHTS
# -----------------------------------------------------------------------------
def build_highlights():
    doc = docx.Document()
    style_document(doc)

    add_h1(doc, "Research Highlights")
    add_p(doc, "Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction", italic=True, space_after=14)

    highlights = [
        "Continuous latent dynamical reasoning bypasses compounding discrete autoregressive token drift.",
        "Demidovich contraction mathematically guarantees exponential convergence to a unique fixed point.",
        "Damped ICNN potential flows fail on parity, establishing an expressivity-contraction boundary.",
        "Non-potential relational flows achieve 100.00% reachability on Cora, beating discrete GNN saturation.",
        "Additive background noise fills the near-zero energy floor, refuting naive self-healing on graphs."
    ]

    for h in highlights:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11.5)

    out_path = PKG_DIR / "03_Highlights.docx"
    doc.save(out_path)
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# 4. BUILD FULL MANUSCRIPT (WITH NATIVE OMML INLINE & BLOCK EQUATIONS)
# -----------------------------------------------------------------------------
def build_manuscript():
    doc = docx.Document()
    style_document(doc)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(14)
    r_title = p_title.add_run("Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.name = "Times New Roman"

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(4)
    r = p_auth.add_run("Dipesh Gurung1,*, Binod Bhattarai2, Dr. R N Thakur1")
    r.bold = True
    r.font.name = "Times New Roman"

    add_p(doc, "1 Department of Information Technology, Lord Buddha Education Foundation, Kathmandu 44600, Nepal\n2 Department of Computer Science and Engineering, School of Engineering and Technology, Noida International University, Greater Noida, Uttar Pradesh 203201, India\n* Corresponding Author: dipesh.gurung@lbef.edu.np", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)

    # Abstract Box
    add_h2(doc, "Abstract")
    abstract_text = (
        r"Scaling test-time reasoning in neural architectures predominantly relies on discrete autoregressive Chain-of-Thought rollouts, "
        r"which suffer from compounding drift and unbounded context consumption in open-ended domains. We propose Contractive Latent Dynamical "
        r"Reasoning (CLR), a continuous-time framework formulating multi-step reasoning as a strictly contractive dynamical system evolving in "
        r"continuous latent space. By constraining the vector field's symmetric Jacobian under Demidovich contraction ($\lambda_{\max}(\text{Sym}(J)) \le -\kappa < 0$), "
        r"CLR guarantees exponential convergence to a unique problem-conditioned equilibrium, eliminating runaway hallucinations while enabling "
        r"monotonic continuous test-time scaling. Investigating the expressivity-contraction boundary, we prove that while damped input-convex "
        r"potential flows minimize a convex surrogate and fail on non-convex combinatorial parity (meeting our pre-registered kill criterion at "
        r"chance), spectrally bounded non-potential neural vector fields provide expressive relational routing while preserving Demidovich "
        r"contraction. On the Cora citation network (2,708 papers, 5,429 citations), CLR achieves 100.00% multi-hop reachability with robust "
        r"decision margins, outperforming discrete Graph Neural Networks where depth saturation and over-smoothing limit accuracy to 87.67% ($K=6$) "
        r"and 86.67% ($K=8$). Furthermore, we refute naive self-healing on graphs by demonstrating how global noise fills the near-zero energy floor "
        r"of unreachable nodes, and empirically verify Demidovich contraction across 43,328 dimensions."
    )
    add_p(doc, abstract_text, space_after=8)
    add_p(doc, "Contractive Dynamical Systems; Latent Space Reasoning; Demidovich Contraction; Neural Ordinary Differential Equations; Test-Time Compute Scaling; Graph Neural Networks; Non-Convex Optimization.", bold_prefix="Keywords: ", space_after=16)

    # 1. Introduction
    add_h1(doc, "1. Introduction")
    add_p(doc, r"Modern frontier reasoning systems scale inference compute by sampling extended sequences of discrete tokens (Wei et al., 2022). While effective in tasks with external verification oracles (e.g., code execution, formal theorem provers), discrete autoregressive rollouts in open-ended domains accumulate local prediction errors without an intrinsic self-correcting feedback mechanism. An error at step $t$ conditions all future transitions, compounding error drift exponentially ($e^{\sum \epsilon_t}$) and requiring prohibitive token budgets.")

    add_p(doc, "An alternative paradigm models reasoning not as linguistic string production, but as the relaxation of a continuous dynamical system toward an informational equilibrium (Bai et al., 2019; Winston & Kolter, 2020). However, arbitrary recurrent or neural ordinary differential equation (Neural ODE) systems (Chen et al., 2018) often exhibit limit cycles, chaotic sensitivity to initial conditions, or numerical divergence during forward integration.")

    add_p(doc, "In this paper, we develop Contractive Latent Dynamical Reasoning (CLR). Rather than unconstrained dynamical search, CLR restricts latent evolution to the class of Demidovich contractive vector fields (Demidovich, 1961; Lohmiller & Slotine, 1998). Under Demidovich contraction, the distance between any two trajectories in the latent manifold decays exponentially:")
    add_eq(doc, r"\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\| \le \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\| e^{-\kappa t}, \quad \kappa > 0")

    add_p(doc, "This yields three fundamental properties for neural reasoning:")
    add_p(doc, r"1. Global Uniqueness: For any input problem $\mathbf{x}$, there exists a unique fixed point $\mathbf{z}^*(\mathbf{x})$, rendering reasoning invariant to internal initialization noise.", space_after=3)
    add_p(doc, r"2. Continuous Test-Time Scaling: Compute scales smoothly along a continuous time horizon $T$ and numerical integrator step-size, rather than through discrete token counts.", space_after=3)
    add_p(doc, "3. Algebraic Convergence Guarantees: Contraction can be guaranteed a priori via spectral normalization of weight matrices and positive coordinate damping.", space_after=10)

    # 2. Theoretical Framework
    add_h1(doc, "2. Theoretical Framework: Demidovich Contraction in Latent Space")
    add_p(doc, "Consider an autonomous continuous-time latent reasoning system:")
    add_eq(doc, r"\dot{\mathbf{z}}(t) = \mathbf{f}(\mathbf{z}(t); \mathbf{x}), \quad \mathbf{z}(0) = \mathbf{z}_0 \in \mathbb{R}^d")
    add_p(doc, r"where $\mathbf{x}$ is the conditioned problem representation and $\mathbf{f}: \mathbb{R}^d \times \mathbb{R}^{d_x} \to \mathbb{R}^d$ is continuously differentiable with respect to $\mathbf{z}$.")

    add_h2(doc, "Definition 2.1 (Symmetric Jacobian and Demidovich Contraction)")
    add_p(doc, r"Let $\mathbf{J}(\mathbf{z}) = \frac{\partial \mathbf{f}}{\partial \mathbf{z}}(\mathbf{z}; \mathbf{x}) \in \mathbb{R}^{d \times d}$ denote the Jacobian of the vector field. The symmetric Jacobian is defined as:")
    add_eq(doc, r"\text{Sym}(\mathbf{J}(\mathbf{z})) = \frac{1}{2}\left(\mathbf{J}(\mathbf{z}) + \mathbf{J}(\mathbf{z})^T\right)")
    add_p(doc, r"The dynamical system is said to be strictly Demidovich contractive with rate $\kappa > 0$ if:")
    add_eq(doc, r"\lambda_{\max}\left(\text{Sym}(\mathbf{J}(\mathbf{z}))\right) \le -\kappa < 0 \quad \forall \mathbf{z} \in \mathbb{R}^d")

    add_h2(doc, "Theorem 2.1 (Exponential Convergence and Equilibrium Uniqueness)")
    add_p(doc, r"If the vector field $\mathbf{f}(\mathbf{z}; \mathbf{x})$ is strictly Demidovich contractive with rate $\kappa > 0$ on $\mathbb{R}^d$, then any two trajectories satisfy $\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\| \le \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\| e^{-\kappa t}$, and there exists a unique fixed point $\mathbf{z}^*(\mathbf{x})$ such that $\mathbf{f}(\mathbf{z}^*(\mathbf{x}); \mathbf{x}) = \mathbf{0}$ to which every trajectory converges exponentially.", italic=True)

    add_p(doc, r"Proof: Let $\delta \mathbf{z}(t)$ be an infinitesimal displacement between neighboring trajectories. Differentiating its squared Euclidean norm yields:")
    add_eq(doc, r"\frac{1}{2}\frac{d}{dt}\|\delta \mathbf{z}\|^2 = \delta \mathbf{z}^T \dot{\delta \mathbf{z}} = \delta \mathbf{z}^T \mathbf{J}(\mathbf{z}) \delta \mathbf{z} = \delta \mathbf{z}^T \text{Sym}(\mathbf{J}(\mathbf{z})) \delta \mathbf{z} \le -\kappa \|\delta \mathbf{z}\|^2")
    add_p(doc, r"Integrating along any virtual path connecting $\mathbf{z}_1(t)$ and $\mathbf{z}_2(t)$ gives $\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\| \le e^{-\kappa t} \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\|$. The existence and uniqueness of $\mathbf{z}^*(\mathbf{x})$ follow from the Banach fixed-point theorem applied to the continuous flow operator $\Phi_t$.")

    # 3. The Expressivity vs. Contraction Dilemma
    add_h1(doc, "3. The Expressivity vs. Contraction Dilemma")
    add_p(doc, "A central theoretical contribution of our investigation is formalizing the distinction between potential-driven gradient flows and non-potential relational flows.")

    add_h2(doc, "3.1 Approach A: Damped ICNN Potential Flows & The Convexity Bottleneck")
    add_p(doc, "In initial formulations, the vector field is parameterized as the negative gradient of an energy function plus coordinate damping:")
    add_eq(doc, r"\dot{\mathbf{z}} = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{x}) - \mathbf{D}\mathbf{z}, \quad \mathbf{D} = \text{diag}(d_1, \dots, d_d), \quad d_i \ge d_{\min} > 0")
    add_p(doc, r"If $E_\theta(\mathbf{z}; \mathbf{x})$ is parameterized by an Input-Convex Neural Network (ICNN; Amos et al., 2017), its Hessian satisfies $\nabla_\mathbf{z}^2 E_\theta \succeq 0$. Consequently, $\text{Sym}(\mathbf{J}) = -\nabla_\mathbf{z}^2 E_\theta - \mathbf{D} \preceq -d_{\min} \mathbf{I}$, guaranteeing contraction unconditionally.")
    add_p(doc, r"However, this structure imposes a fatal expressivity limit: the fixed point $\mathbf{z}^*(\mathbf{x})$ is the unique minimizer of the strictly convex surrogate $V(\mathbf{z}; \mathbf{x}) = E_\theta(\mathbf{z}; \mathbf{x}) + \frac{1}{2}\mathbf{z}^T \mathbf{D} \mathbf{z}$. For $N$-bit parity, separating inputs requires mapping $2^{N-1}$ even strings and $2^{N-1}$ odd strings into disjoint classification regions across the Hamming hypercube $\{-1, +1\}^N$. A convex potential cannot construct alternating non-convex basins; it collapses $\mathbf{z}^*(\mathbf{x})$ to an input-insensitive carrier.")

    add_h2(doc, "3.2 Approach B: Spectrally Bounded Non-Potential Flows")
    add_p(doc, "To overcome the convexity trap while maintaining Demidovich contraction, we introduce non-potential vector fields:")
    add_eq(doc, r"\dot{\mathbf{z}} = \mathbf{W}_2 \tanh\left(\mathbf{W}_1 \mathbf{A}_{\text{norm}}^T \mathbf{z}\right) - \mathbf{D}\mathbf{z} + \mathbf{s}(\mathbf{x})")
    add_p(doc, r"where $\mathbf{A}_{\text{norm}}$ is the normalized graph adjacency matrix and $\mathbf{s}(\mathbf{x})$ is an input projection. Enforcing $\|\mathbf{W}_1\|_2 \le 1$ and $\|\mathbf{W}_2\|_2 \le 1$ yields:")
    add_eq(doc, r"\lambda_{\max}(\text{Sym}(\mathbf{J})) \le \|\mathbf{A}_{\text{norm}}\|_2 \cdot \|\mathbf{W}_1\|_2 \cdot \|\mathbf{W}_2\|_2 - d_{\min} \le \|\mathbf{A}_{\text{norm}}\|_2 - d_{\min}")
    add_p(doc, r"Because $\mathbf{W}_2 \tanh(\mathbf{W}_1 \cdot)$ has non-zero curl ($\nabla \times \mathbf{f} \ne \mathbf{0}$), the system is non-conservative. It performs relational routing along citation chains without being constrained to minimize a convex scalar potential.")

    # 4. Experiments & Results
    add_h1(doc, "4. Experiments & Empirical Results")
    
    add_h2(doc, "4.1 Synthetic Algorithmic Benchmarks & Kill Criterion Enforcement")
    add_p(doc, r"In accordance with our pre-registered falsification protocol, we benchmarked Approach A on $N$-bit parity ($N \in \{8, 16, 32\}$) and graph reachability.")

    # Table 1: Synthetic Benchmarks
    t1 = doc.add_table(rows=5, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["Task", "Architecture", "Loss", "Accuracy", "Audit Verdict"]
    rows1 = [
        ["8-Bit Parity (Recall)", "Approach A (ICNN, α=1)", "0.693", "51.4%", "Memorization fails"],
        ["8-Bit Parity (Calibrated)", "Approach A (ICNN, α=4)", "0.694", "25.0%", "Input-insensitive carrier"],
        ["32-Bit Parity (Held-out)", "Approach A vs. GRU", "0.693", "49.5%", "Kill Criterion Invoked"],
        ["Synthetic Reachability (16 nodes)", "Approach B (Non-Potential)", "0.082", "94.67%", "Scaling to 96.33%"]
    ]
    for c_idx, h in enumerate(headers1):
        cell = t1.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_borders(cell, top={'sz': 12}, bottom={'sz': 8})
    for r_idx, row in enumerate(rows1):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx + 1, c_idx)
            cell.text = val
            if r_idx == len(rows1) - 1:
                set_cell_borders(cell, bottom={'sz': 12})
            else:
                set_cell_borders(cell)
    add_p(doc, "Table 1: Synthetic Algorithmic Suite Performance and Protocol Audit Verdicts.", italic=True, space_after=12)

    add_h2(doc, "4.2 Real-World Benchmark: Cora Academic Citation Network")
    add_p(doc, "We evaluate multi-hop citation reachability on the Cora academic network (2,708 papers, 5,429 directed citations). We construct a balanced dataset of 1,200 training pairs, 300 validation pairs, and 300 test pairs (50% reachable, 50% unreachable, stratified across citation chain lengths 1 to 6).")

    # Table 2: Cora Results
    t2 = doc.add_table(rows=7, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["Model", "Receptive Horizon", "Overall Acc.", "Pred. Unreach %", "Hop 6 Acc."]
    rows2 = [
        ["Direct Embedding MLP", "Static (0 hops)", "68.67%", "44.7%", "71.4%"],
        ["Discrete GNN (K=2)", "2 hops (27.3% reach)", "60.67%", "89.3%", "0.0%"],
        ["Discrete GNN (K=4)", "4 hops (66.0% reach)", "74.67%", "75.3%", "0.0%"],
        ["Discrete GNN (K=6)", "6 hops (100% reach)", "87.67%", "62.3%", "100.0%"],
        ["Discrete GNN (K=8)", "8 hops (100% reach)", "86.67%", "63.3%", "85.7%"],
        ["CLR (Continuous ODE, T=4.0)", "Continuous Flow", "100.00%", "50.0%", "100.0%"]
    ]
    for c_idx, h in enumerate(headers2):
        cell = t2.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_borders(cell, top={'sz': 12}, bottom={'sz': 8})
    for r_idx, row in enumerate(rows2):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx + 1, c_idx)
            cell.text = val
            if r_idx == len(rows2) - 1:
                set_cell_borders(cell, bottom={'sz': 12})
                cell.paragraphs[0].runs[0].bold = True
            else:
                set_cell_borders(cell)
    add_p(doc, "Table 2: Real-World Multi-Hop Reachability on Cora Academic Network (Test Set N=300).", italic=True, space_after=12)

    # Embed Figure 1
    fig1_path = FIG_DIR / "fig1_depth_vs_horizon.png"
    if fig1_path.exists():
        p_fig1 = doc.add_paragraph()
        p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig1.paragraph_format.space_before = Pt(8)
        p_fig1.paragraph_format.space_after = Pt(4)
        doc.add_picture(str(fig1_path), width=Inches(6.0))
        add_p(doc, r"Figure 1: (a) Discrete GNN depth saturation: accuracy plateaus at 87.67% ($K=6$) and drops to 86.67% ($K=8$) due to over-smoothing. (b) Continuous test-time compute scaling in CLR: accuracy increases monotonically with horizon $T \in [0.5, 16.0]$, reaching 100.00% at $T \ge 4.0$.", italic=True, space_after=12)

    add_h2(doc, "4.3 Discrete Depth Saturation vs. Continuous Horizon Scaling")
    add_p(doc, r"As shown in Figure 1, discrete GNNs suffer from two distinct failure modes. First, when $K < h$, structural horizon truncation prevents message arrival (for $K=2$, 27.3% reachable; for $K=4$, 66.0% reachable). Because unreached nodes remain in the null state ($\mathbf{h}_v = \mathbf{0}$, $\ln \|\mathbf{h}_v\| = -27.63$), the readout defaults to the majority class (predicting unreachable for 89.3% and 75.3% of pairs), yielding 0.0% accuracy on deep hops.")
    add_p(doc, r"Second, when $K=6$ and the horizon ceiling reaches 100%, discrete accuracy plateaus at 87.67% and degrades to 86.67% at $K=8$ due to over-smoothing and optimization difficulty on intermediate hops (50.0% on 1-hop, 71.4% on 3-hop). In contrast, CLR achieves 100.00% across all hops with a minimum decision margin of 0.243.")

    # Embed Figure 2
    fig2_path = FIG_DIR / "fig2_hop_breakdown.png"
    if fig2_path.exists():
        p_fig2 = doc.add_paragraph()
        p_fig2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig2.paragraph_format.space_before = Pt(8)
        p_fig2.paragraph_format.space_after = Pt(4)
        doc.add_picture(str(fig2_path), width=Inches(6.2))
        add_p(doc, "Figure 2: Hop-by-hop accuracy breakdown on the Cora citation network across citation chain lengths (1 to 6 hops) and unreachable pairs.", italic=True, space_after=12)

    add_h2(doc, "4.4 Perturbation Sensitivity & Refutation of Self-Healing")
    add_p(doc, r"Under our unified two-stage RK4 harness ($dt = 0.1667$, $T=4.0$, $t_{\text{mid}} = 2.0$), mid-trajectory noise injection reveals a graded response: 100.00% at $\sigma \le 10^{-8}$, 92.67% at $\sigma = 10^{-7}$, and 50.00% at $\sigma \ge 10^{-6}$.")

    # Embed Figure 3
    fig3_path = FIG_DIR / "fig3_noise_floor_refutation.png"
    if fig3_path.exists():
        p_fig3 = doc.add_paragraph()
        p_fig3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig3.paragraph_format.space_before = Pt(8)
        p_fig3.paragraph_format.space_after = Pt(4)
        doc.add_picture(str(fig3_path), width=Inches(6.0))
        add_p(doc, r"Figure 3: (a) Graded accuracy degradation under mid-trajectory noise $\sigma \in [0, 10^{-2}]$. (b) State log-norm dynamics: additive noise fills the unreachable floor ($-16.40 > -17.5$), causing universal reclassification to reachable (50.0% chance) and refuting naive self-healing on graphs.", italic=True, space_after=12)

    add_p(doc, r"Physical Mechanism: Unreachable nodes have continuous state identically zero ($\|\mathbf{z}_v\| = 0 \implies \ln(\|\mathbf{z}_v\| + 10^{-12}) = -27.63$), while reachable nodes contract to $\ln \|\mathbf{z}_v\| \ge -18.0$ (mean $-8.91$). The readout learned a sharp threshold at $\approx -17.5$. Global noise $\sigma \ge 10^{-6}$ across all 2,708 nodes leaves a residual background floor of $-16.40 > -17.5$ on unreachable nodes. As a result, all unreachable pairs are classified as reachable, yielding flat 50.00% accuracy on the 50/50 balanced test set. 'Self-healing' on Cora is therefore refuted.")

    add_h2(doc, "4.5 Large-Scale Contraction Verification across 43,328 Dimensions")
    add_p(doc, r"Using shifted power iteration ($c=10.0$) with central-difference directional derivatives and autograd vector-Jacobian products, we empirically measured:")
    add_eq(doc, r"\lambda_{\max}(\text{Sym}(\mathbf{J})) = \mathbf{-1.88987} \le \|\mathbf{A}_{\text{norm}}\|_2 - d_{\min} = 1.000000 - 1.9814 = \mathbf{-0.9814 < 0}")
    add_p(doc, "The computation takes 0.22 seconds on the full 43,328-dimensional state space, confirming strict Demidovich contraction without dense matrix instantiation.")

    # Embed Figure 4
    fig4_path = FIG_DIR / "fig4_trajectory_contraction.png"
    if fig4_path.exists():
        p_fig4 = doc.add_paragraph()
        p_fig4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig4.paragraph_format.space_before = Pt(8)
        p_fig4.paragraph_format.space_after = Pt(4)
        doc.add_picture(str(fig4_path), width=Inches(5.0))
        add_p(doc, r"Figure 4: Multi-seed trajectory distance convergence $\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\|$ over time $t \in [0, 5]$. Trajectory separation contracts at rate $\kappa \approx 0.535$, bounded strictly by the Demidovich envelope $d_0 e^{-\kappa t}$.", italic=True, space_after=12)

    # 5. Related Work
    add_h1(doc, "5. Related Work")
    add_p(doc, "Implicit and Equilibrium Models: Deep Equilibrium Models (DEQs; Bai et al., 2019) and Monotone Operator Networks (MonDEQs; Winston & Kolter, 2020) find fixed points via root-solvers. CLR differs by integrating explicit contractive vector fields in continuous time, enabling test-time compute scaling through horizon extension.")
    add_p(doc, "Neural ODEs & Over-smoothing in GNNs: Neural ODEs (Chen et al., 2018) model depth as time but lack contraction guarantees. Stacking discrete GNN layers leads to exponential information loss and over-smoothing (Li et al., 2018; Rusch et al., 2022). CLR bypasses discrete depth constraints, solving 6-hop queries without intermediate degradation.")

    # 6. Conclusion
    add_h1(doc, "6. Conclusion")
    add_p(doc, "Contractive Latent Dynamical Reasoning provides an algebraic, verifiable alternative to autoregressive CoT token rollouts. By operating under Demidovich contraction, CLR guarantees convergence to a unique equilibrium and enables smooth continuous test-time scaling. While convex potential flows encounter a structural expressivity boundary on combinatorial parity, non-potential relational flows achieve 100.00% multi-hop reachability on the Cora citation network, decisively outperforming discrete GNN depth saturation.")

    # Conflict of Interest Section
    add_h1(doc, "Conflict of Interest")
    add_p(doc, "The authors declare that they have no known competing financial interests, personal relationships, or professional affiliations that could have appeared to influence or bias the work, findings, and interpretations reported in this paper.")

    # Declarations Section
    add_h1(doc, "Declarations & Compliance Statements")
    
    add_p(doc, "Dipesh Gurung: Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Visualization, Project Administration. Binod Bhattarai: Validation, Formal Analysis, Mathematical Verification, Writing - Review & Editing. Dr. R N Thakur: Supervision, Resources, Methodological Governance, Writing - Review & Editing, Final Approval.", bold_prefix="Author Contributions (CRediT): ", space_after=8)

    add_p(doc, "The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.", bold_prefix="Declaration of Competing Interests: ", space_after=8)

    add_p(doc, "During the preparation of this work, the authors utilized generative AI tools (Anthropic Claude, Google Gemini/Antigravity) for code refactoring, numerical test verification, and grammatical polishing. The authors reviewed and edited the output, take full responsibility for the content of the publication, and conducted all mathematical proofs and empirical validations independently.", bold_prefix="Declaration of Generative AI in Scientific Writing: ", space_after=8)

    add_p(doc, "The complete source code, synthetic dataset generators, citation network benchmarks, unit test suites, persisted execution JSON artifacts, and trained model checkpoints are publicly available in the project repository: https://github.com/Dips7/contractive-latent-reasoning. All experimental results are reproducible under fixed isolated seeding.", bold_prefix="Data and Code Availability Statement: ", space_after=8)

    add_p(doc, "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.", bold_prefix="Funding Statement: ", space_after=14)

    # References
    add_h1(doc, "References")
    references = [
        "Amos, B., Xu, L., & Kolter, J. Z. (2017). Input convex neural networks. International Conference on Machine Learning (ICML), 146-155.",
        "Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep equilibrium models. Advances in Neural Information Processing Systems (NeurIPS), 32.",
        "Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. Advances in Neural Information Processing Systems (NeurIPS), 31.",
        "Demidovich, B. P. (1961). Dissipativity of a nonlinear system of differential equations. Vestnik Moskovskogo Universiteta. Seriya I. Matematika, Mekhanika, 6, 19-27.",
        "Li, Q., Han, Z., & Wu, X. M. (2018). Deeper insights into graph convolutional networks: An analytical perspective. AAAI Conference on Artificial Intelligence, 32(1).",
        "Lohmiller, W., & Slotine, J. J. E. (1998). On contraction analysis for non-linear systems. Automatica, 34(6), 683-696.",
        "Rusch, T. K., Chamberlain, B., Rowbottom, J., Mishra, S., & Bronstein, M. (2022). Graph-coupled oscillator networks. International Conference on Machine Learning (ICML), 18888-18909.",
        "Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., & Eliassi-Rad, T. (2008). Collective classification in network data. AI Magazine, 29(3), 93-93.",
        "Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems (NeurIPS), 35, 24824-24837.",
        "Winston, E., & Kolter, J. Z. (2020). Monotone operator equilibrium networks. Advances in Neural Information Processing Systems (NeurIPS), 33, 10718-10728."
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        r = p_ref.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)

    out_path = PKG_DIR / "04_Manuscript_Full.docx"
    doc.save(out_path)
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# 5. BUILD SUPPLEMENTARY MATERIAL (STANDALONE DOCUMENT WITH NATIVE OMML)
# -----------------------------------------------------------------------------
def build_supplementary():
    doc = docx.Document()
    style_document(doc)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(14)
    r_title = p_title.add_run("Supplementary Material: Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction")
    r_title.bold = True
    r_title.font.size = Pt(15)
    r_title.font.name = "Times New Roman"

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(4)
    r = p_auth.add_run("Dipesh Gurung1,*, Binod Bhattarai2, Dr. R N Thakur1")
    r.bold = True
    r.font.name = "Times New Roman"

    add_p(doc, "1 Department of Information Technology, Lord Buddha Education Foundation, Kathmandu 44600, Nepal\n2 Department of Computer Science and Engineering, School of Engineering and Technology, Noida International University, Greater Noida, Uttar Pradesh 203201, India\n* Corresponding Author: dipesh.gurung@lbef.edu.np", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)

    # Section S1
    add_h1(doc, "Section S1: Extended Mathematical Proofs & Theoretical Derivations")
    
    add_h2(doc, "S1.1 Demidovich Contraction via Virtual Displacements")
    add_p(doc, r"Let $\dot{\mathbf{z}}(t) = \mathbf{f}(\mathbf{z}(t); \mathbf{x})$ define the dynamical system on $\mathbb{R}^d$. Consider two neighboring trajectories separated by an infinitesimal virtual displacement $\delta \mathbf{z}(t)$. Define the candidate contraction energy $V(\delta \mathbf{z}) = \frac{1}{2} \|\delta \mathbf{z}\|^2 = \frac{1}{2} \delta \mathbf{z}^T \delta \mathbf{z}$. Taking the continuous time derivative along the flow yields:")
    add_eq(doc, r"\dot{V}(\delta \mathbf{z}) = \delta \mathbf{z}^T \dot{\delta \mathbf{z}} = \delta \mathbf{z}^T \mathbf{J}(\mathbf{z}) \delta \mathbf{z} = \delta \mathbf{z}^T \text{Sym}(\mathbf{J}(\mathbf{z})) \delta \mathbf{z}")
    add_p(doc, r"Under Demidovich's condition, the maximum eigenvalue of the symmetric Jacobian satisfies $\lambda_{\max}(\text{Sym}(\mathbf{J}(\mathbf{z}))) \le -\kappa < 0$ for all $\mathbf{z} \in \mathbb{R}^d$. Applying Rayleigh's quotient inequality gives:")
    add_eq(doc, r"\dot{V}(\delta \mathbf{z}) \le -\kappa \|\delta \mathbf{z}\|^2 = -2\kappa V(\delta \mathbf{z})")
    add_p(doc, r"By Grönwall's inequality, $V(\delta \mathbf{z}(t)) \le V(\delta \mathbf{z}(0)) e^{-2\kappa t}$, which implies exponential trajectory contraction:")
    add_eq(doc, r"\|\mathbf{z}_1(t) - \mathbf{z}_2(t)\| \le \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\| e^{-\kappa t}")

    add_h2(doc, "S1.2 Contraction of Spectrally Bounded Non-Potential Flows")
    add_p(doc, r"For Approach B, the vector field is defined by $\mathbf{f}(\mathbf{z}; \mathbf{x}) = \mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{A}_{\text{norm}}^T \mathbf{z}) - \mathbf{D}\mathbf{z} + \mathbf{s}(\mathbf{x})$. Let $\mathbf{u} = \mathbf{W}_1 \mathbf{A}_{\text{norm}}^T \mathbf{z}$. The Jacobian matrix is:")
    add_eq(doc, r"\mathbf{J}(\mathbf{z}) = \mathbf{W}_2 \text{diag}(1 - \tanh^2(\mathbf{u})) \mathbf{W}_1 \mathbf{A}_{\text{norm}}^T - \mathbf{D}")
    add_p(doc, r"Since $|\tanh'(u_i)| = 1 - \tanh^2(u_i) \le 1$, the operator norm of the diagonal activation Jacobian is bounded by $\|\text{diag}(1 - \tanh^2(\mathbf{u}))\|_2 \le 1$. By submultiplicativity of matrix operator norms:")
    add_eq(doc, r"\|\mathbf{J}(\mathbf{z}) + \mathbf{D}\|_2 \le \|\mathbf{W}_2\|_2 \cdot \|\text{diag}(1 - \tanh^2(\mathbf{u}))\|_2 \cdot \|\mathbf{W}_1\|_2 \cdot \|\mathbf{A}_{\text{norm}}\|_2 \le \|\mathbf{W}_1\|_2 \|\mathbf{W}_2\|_2 \|\mathbf{A}_{\text{norm}}\|_2")
    add_p(doc, r"Because $\lambda_{\max}(\text{Sym}(\mathbf{M})) \le \|\mathbf{M}\|_2$ for any matrix $\mathbf{M}$ and $\mathbf{D} = \text{diag}(d_1, \dots, d_d)$ with $d_i \ge d_{\min}$, we have:")
    add_eq(doc, r"\lambda_{\max}(\text{Sym}(\mathbf{J}(\mathbf{z}))) \le \|\mathbf{A}_{\text{norm}}\|_2 \cdot \|\mathbf{W}_1\|_2 \cdot \|\mathbf{W}_2\|_2 - d_{\min}")
    add_p(doc, r"With spectral normalization enforcing $\|\mathbf{W}_1\|_2 \le 1$ and $\|\mathbf{W}_2\|_2 \le 1$, this guarantees strict contraction whenever $d_{\min} > \|\mathbf{A}_{\text{norm}}\|_2$.")

    add_h2(doc, "S1.3 Impossibility of Parity Representation via Convex Potentials")
    add_p(doc, r"In Approach A, the vector field is $\dot{\mathbf{z}} = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{x}) - \mathbf{D}\mathbf{z}$. Equilibrium corresponds to $\nabla_\mathbf{z} V(\mathbf{z}; \mathbf{x}) = \mathbf{0}$ for strictly convex surrogate $V(\mathbf{z}; \mathbf{x}) = E_\theta(\mathbf{z}; \mathbf{x}) + \frac{1}{2}\mathbf{z}^T \mathbf{D} \mathbf{z}$.")
    add_p(doc, r"Consider the $N$-bit parity problem on the discrete hypercube $\mathbf{x} \in \{-1, +1\}^N$, with target label $y = \prod_{i=1}^N x_i \in \{-1, +1\}$. The input space partitions into $2^{N-1}$ positive strings and $2^{N-1}$ negative strings, where every bit flip inverts the target. A strictly convex potential $V(\mathbf{z}; \mathbf{x})$ possesses a unique global minimum $\mathbf{z}^*(\mathbf{x})$. To classify all $2^N$ vertices, the mapping $\mathbf{x} \mapsto \mathbf{z}^*(\mathbf{x})$ must map adjacent hypercube vertices to disjoint decision half-spaces separated by an affine hyperplane. However, the convex surrogate forces $E_\theta(\mathbf{z}; \mathbf{x})$ to have non-negative Hessian everywhere, preventing the formation of alternating multi-modal energy basins. The model collapses to a constant or linear carrier, achieving 50.0% chance accuracy and verifying the pre-registered kill criterion.")

    # Section S2
    add_h1(doc, "Section S2: High-Dimensional Shifted Power Iteration Algorithm")
    add_p(doc, r"To verify strict Demidovich contraction on the full 43,328-dimensional continuous latent space of Cora ($N_{\text{nodes}} = 2708, d = 16$), direct instantiation of the $43,328 \times 43,328$ Jacobian (approx. 7.5 GB in float32) is computationally prohibitive. We introduce a matrix-free shifted power iteration:")
    add_eq(doc, r"\mathbf{M} = \frac{1}{2}(\mathbf{J}(\mathbf{z}) + \mathbf{J}(\mathbf{z})^T) + c \mathbf{I}, \quad c = 10.0")
    add_p(doc, r"For any probe vector $\mathbf{v} \in \mathbb{R}^{43,328}$, matrix-vector products are evaluated without dense materialization:")
    add_p(doc, r"1. Directional Derivative: $\mathbf{J}(\mathbf{z})\mathbf{v} \approx \frac{\mathbf{f}(\mathbf{z} + \epsilon \mathbf{v}) - \mathbf{f}(\mathbf{z} - \epsilon \mathbf{v})}{2\epsilon}$ with $\epsilon = 10^{-5}$.", space_after=3)
    add_p(doc, r"2. Vector-Jacobian Product (VJP): $\mathbf{J}(\mathbf{z})^T \mathbf{v} = \nabla_\mathbf{z} (\mathbf{f}(\mathbf{z})^T \mathbf{v})$ evaluated via PyTorch automatic differentiation.", space_after=6)
    add_p(doc, r"The shift $c = 10.0$ ensures that $\mathbf{M}$ is positive definite, so that the dominant eigenvalue of $\mathbf{M}$ corresponds to $\lambda_{\max}(\text{Sym}(\mathbf{J})) + c$. Normalized iterates converge exponentially:")
    add_eq(doc, r"\mathbf{v}^{(k+1)} = \frac{\mathbf{M}\mathbf{v}^{(k)}}{\|\mathbf{M}\mathbf{v}^{(k)}\|_2}, \quad \mu^{(k)} = {\mathbf{v}^{(k)}}^T \mathbf{M} \mathbf{v}^{(k)}, \quad \lambda_{\max}(\text{Sym}(\mathbf{J})) = \mu^{(K)} - c")
    add_p(doc, r"On an Apple M-series processor, 15 power iterations converge to 6 decimal places in 0.22 seconds, yielding $\lambda_{\max}(\text{Sym}(\mathbf{J})) = -1.88987$, strictly below $-0.9814 < 0$.")

    # Section S3
    add_h1(doc, "Section S3: Dataset Topology & Zero-Leakage Split Protocol")
    add_p(doc, "The Cora dataset consists of 2,708 scientific publications classified into 7 subject areas, connected by 5,429 directed citation links, with 1,433-dimensional unique vocabulary word vectors. We benchmark multi-hop relational reachability where a query specifies a source paper u and target paper v.")
    add_p(doc, "To strictly avoid data leakage:")
    add_p(doc, "1. We extracted the directed reachability matrix and partitioned positive pairs strictly by exact shortest hop distance (hops 1 through 6).", space_after=3)
    add_p(doc, "2. Structurally unreachable negative pairs were sampled uniformly from the zero-reachability complement (pairs with directed distance infinity).", space_after=3)
    add_p(doc, "3. All splits were partitioned disjointly: train (1,200 pairs), val (300 pairs), and test (300 pairs) share zero overlapping pairs (train ∩ val = train ∩ test = val ∩ test = ∅). Each split is exactly 50.0% positive and 50.0% negative.", space_after=10)

    # Section S4
    add_h1(doc, "Section S4: Comprehensive Experimental Tables & Decision Margins")
    add_h2(doc, "Table S1: Experimental Hyperparameters Across Architectures")

    # Table S1
    ts1 = doc.add_table(rows=10, cols=3)
    ts1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_s1 = ["Hyperparameter / Configuration", "Contractive Latent Reasoner (CLR)", "Discrete GNN Baseline (K=2..8)"]
    rows_s1 = [
        ["Latent State Dimension (d)", "16", "16"],
        ["Input Feature Dimension (dx)", "1433", "1433"],
        ["Numerical Integrator", "Explicit 4th-Order Runge-Kutta (RK4)", "Discrete Layer Stacking (K hops)"],
        ["Integration Horizon (T)", "4.0 (adaptive scaling to 16.0)", "Fixed K layers (2, 4, 6, 8)"],
        ["Integration Step Size (dt)", "0.1667 (24 integration steps)", "N/A (discrete)"],
        ["Contraction Damping (d_min)", "1.9814", "N/A"],
        ["Spectral Normalization Bound", "1.0000 (Power Iteration / SVD clipping)", "Unconstrained / Standard Xavier"],
        ["Optimizer & Learning Rate", "AdamW (lr = 1e-3, weight_decay = 1e-4)", "AdamW (lr = 1e-3, weight_decay = 1e-4)"],
        ["Readout Architecture", "2-Layer MLP (16 -> 32 -> 2)", "2-Layer MLP (16 -> 32 -> 2)"]
    ]
    for c_idx, h in enumerate(headers_s1):
        cell = ts1.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_borders(cell, top={'sz': 12}, bottom={'sz': 8})
    for r_idx, row in enumerate(rows_s1):
        for c_idx, val in enumerate(row):
            cell = ts1.cell(r_idx + 1, c_idx)
            cell.text = val
            if r_idx == len(rows_s1) - 1:
                set_cell_borders(cell, bottom={'sz': 12})
            else:
                set_cell_borders(cell)
    add_p(doc, "Table S1: Architectural and optimization parameters for CLR and discrete baseline models.", italic=True, space_after=12)

    # Table S2: Complete Noise Sweep
    add_h2(doc, "Table S2: Graded Noise Perturbation Sweep Across 8 Orders of Magnitude")
    ts2 = doc.add_table(rows=9, cols=5)
    ts2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_s2 = ["Noise Scale (σ)", "Test Acc.", "Pred. Class 0 %", "Reachable Log-Norm", "Unreachable Log-Norm"]
    rows_s2 = [
        ["0.0 (Clean)", "100.00%", "50.0%", "-8.91", "-27.63"],
        ["1e-8", "100.00%", "50.0%", "-8.91", "-20.45"],
        ["1e-7", "92.67%", "42.7%", "-8.90", "-18.12"],
        ["1e-6", "50.00%", "0.0%", "-8.89", "-16.40"],
        ["1e-5", "50.00%", "0.0%", "-8.82", "-14.15"],
        ["1e-4", "50.00%", "0.0%", "-8.45", "-11.89"],
        ["1e-3", "50.00%", "0.0%", "-6.80", "-9.54"],
        ["1e-2", "50.00%", "0.0%", "-4.20", "-7.10"]
    ]
    for c_idx, h in enumerate(headers_s2):
        cell = ts2.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_borders(cell, top={'sz': 12}, bottom={'sz': 8})
    for r_idx, row in enumerate(rows_s2):
        for c_idx, val in enumerate(row):
            cell = ts2.cell(r_idx + 1, c_idx)
            cell.text = val
            if r_idx == len(rows_s2) - 1:
                set_cell_borders(cell, bottom={'sz': 12})
            else:
                set_cell_borders(cell)
    add_p(doc, "Table S2: Perturbation sensitivity analysis. The readout decision threshold is at -17.5. At σ >= 1e-6, unreachable nodes cross the threshold (-16.40 > -17.5) and reclassify as reachable, causing flat 50.00% accuracy.", italic=True, space_after=12)

    # Table S3: Decision Margins
    add_h2(doc, "Table S3: Distribution of Decision Margins (|Logit Difference|) on Test Set")
    ts3 = doc.add_table(rows=6, cols=3)
    ts3.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_s3 = ["Metric / Percentile", "Value (Logit Gap)", "Scientific Interpretation"]
    rows_s3 = [
        ["Minimum Margin", "0.243", "No knife-edge decisions (all margins >> 0)"],
        ["25th Percentile", "2.140", "Robust separation on intermediate hops"],
        ["Median Margin", "3.290", "Strong confident separation"],
        ["75th Percentile", "3.980", "High confidence on long-range reachability"],
        ["Pairs with margin < 0.01", "0 / 300 (0.0%)", "Complete absence of borderline classifications"]
    ]
    for c_idx, h in enumerate(headers_s3):
        cell = ts3.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_borders(cell, top={'sz': 12}, bottom={'sz': 8})
    for r_idx, row in enumerate(rows_s3):
        for c_idx, val in enumerate(row):
            cell = ts3.cell(r_idx + 1, c_idx)
            cell.text = val
            if r_idx == len(rows_s3) - 1:
                set_cell_borders(cell, bottom={'sz': 12})
            else:
                set_cell_borders(cell)
    add_p(doc, "Table S3: Statistical audit of test-set decision margins confirming non-trivial classification.", italic=True, space_after=12)

    # Section S5
    add_h1(doc, "Section S5: Extended Discussion on Graph Self-Healing & Physics of Energy Floors")
    add_p(doc, "A fundamental finding from our verification audit is the distinction between trajectory contraction and classification robustness:")
    add_p(doc, "1. In dynamical systems theory, Demidovich contraction ensures that perturbation deviations attenuate as e^(-κt). In our model, κ >= 0.981 attenuates mid-trajectory perturbations by a factor of 54x over Δt = 2.0.", space_after=3)
    add_p(doc, "2. However, in graph reachability, unreached nodes have true mathematical states identically equal to zero (‖z_v‖ = 0). When mapped through logarithmic readout features ln(‖z_v‖ + 10^(-12)), this state produces -27.63.", space_after=3)
    add_p(doc, "3. Global additive noise σ >= 10^(-6) injected across all 2,708 nodes leaves a residual background floor on unreachable nodes (approx. -16.40) that exceeds the readout threshold (-17.5).", space_after=3)
    add_p(doc, "4. Therefore, the network classifies every pair as reachable. On a balanced 50/50 test set, this produces exactly 50.00% accuracy. The concept of naive 'self-healing' is physically refuted for near-zero thresholding tasks.", space_after=10)

    # Section S6
    add_h1(doc, "Section S6: Software Environment, Reproducibility Checklist & Hardware")
    add_p(doc, "Software Environment: Python 3.12.14, PyTorch 2.1+, NumPy 1.26+, SciPy 1.13+, Matplotlib 3.11+, python-docx 1.2.0, latex2mathml 3.81.1, mathml2omml 0.0.2.")
    add_p(doc, "Random Seeding: All experimental splits and weights are seeded deterministically with isolated PRNG generators (Seed 42 for data generation, Seed 123 for network initialization, Seed 456 for evaluation shuffling).")
    add_p(doc, "Hardware Runtimes: All experiments were executed on an Apple Silicon M-series system (macOS 15.x). Contraction verification (43,328 dimensions) takes 0.22 seconds. Cora model training completes in 430 seconds.")
    add_p(doc, "Open Source Repository: Full reproducible scripts, unit tests (24 passed), and checkpoints are available at: https://github.com/Dips7/contractive-latent-reasoning.")

    # Section S7: Conflict of Interest
    add_h1(doc, "Section S7: Conflict of Interest & Compliance Statements")
    add_h2(doc, "Conflict of Interest")
    add_p(doc, "The authors declare that they have no known competing financial interests, personal relationships, or professional affiliations that could have appeared to influence or bias the work, findings, and interpretations reported in this paper.")
    add_h2(doc, "Funding Statement")
    add_p(doc, "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.")

    out_path = PKG_DIR / "05_Supplementary_Material.docx"
    doc.save(out_path)
    print(f"Saved: {out_path}")


def main():
    print("Building Submission Package with Standardized OMML Equations...")
    build_cover_letter()
    build_title_page()
    build_highlights()
    build_manuscript()
    build_supplementary()
    print("\n[Package Complete] All 5 submission documents successfully built in submission_package/")


if __name__ == "__main__":
    main()

