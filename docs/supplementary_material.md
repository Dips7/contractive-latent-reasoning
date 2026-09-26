# Supplementary Material
## Contractive Latent Dynamical Reasoning: Bypassing Autoregressive Rollouts via Operator-Norm Contraction

**Authors**: Dipesh Gurung$^{1,*}$, Binod Bhattarai$^{2}$, Dr. R N Thakur$^{1}$  
$^{1}$ Department of Information Technology, Lord Buddha Education Foundation, Kathmandu 44600, Nepal  
$^{2}$ Department of Computer Science and Engineering, School of Engineering and Technology, Noida International University, Greater Noida, Uttar Pradesh 203201, India  
$^*$ Corresponding Author: `dipesh.gurung@lbef.edu.np`

---

## Section S1: Extended Mathematical Proofs & Theoretical Derivations

### S1.1 Demidovich Contraction via Virtual Displacements
Let $\dot{\mathbf{z}}(t) = \mathbf{f}(\mathbf{z}(t); \mathbf{x})$ define an autonomous dynamical system on $\mathbb{R}^d$. Consider two neighboring trajectories separated by an infinitesimal virtual displacement $\delta \mathbf{z}(t)$. Define the quadratic contraction energy:
$$V(\delta \mathbf{z}) = \frac{1}{2} \|\delta \mathbf{z}\|^2 = \frac{1}{2} \delta \mathbf{z}^T \delta \mathbf{z}$$

Differentiating with respect to time along the continuous trajectory yields:
$$\dot{V}(\delta \mathbf{z}) = \delta \mathbf{z}^T \dot{\delta \mathbf{z}} = \delta \mathbf{z}^T \mathbf{J}(\mathbf{z}) \delta \mathbf{z} = \delta \mathbf{z}^T \text{Sym}(\mathbf{J}(\mathbf{z})) \delta \mathbf{z}$$

Under Demidovich's condition, the maximum eigenvalue of the symmetric Jacobian is strictly negative and uniformly bounded away from zero:
$$\lambda_{\max}(\text{Sym}(\mathbf{J}(\mathbf{z}))) \le -\kappa < 0 \quad \forall \mathbf{z} \in \mathbb{R}^d$$

Applying Rayleigh's quotient inequality:
$$\dot{V}(\delta \mathbf{z}) \le -\kappa \|\delta \mathbf{z}\|^2 = -2\kappa V(\delta \mathbf{z})$$

Integrating via Grönwall's inequality yields exponential decay:
$$V(\delta \mathbf{z}(t)) \le V(\delta \mathbf{z}(0)) e^{-2\kappa t} \implies \|\mathbf{z}_1(t) - \mathbf{z}_2(t)\| \le \|\mathbf{z}_1(0) - \mathbf{z}_2(0)\| e^{-\kappa t}$$

### S1.2 Contraction of Spectrally Bounded Non-Potential Flows
In Approach B, the neural vector field is parameterized as:
$$\dot{\mathbf{z}} = \mathbf{W}_2 \tanh\left(\mathbf{W}_1 \mathbf{A}_{\text{norm}}^T \mathbf{z}\right) - \mathbf{D}\mathbf{z} + \mathbf{s}(\mathbf{x})$$

Let $\mathbf{u} = \mathbf{W}_1 \mathbf{A}_{\text{norm}}^T \mathbf{z}$. The Jacobian matrix of this vector field is:
$$\mathbf{J}(\mathbf{z}) = \mathbf{W}_2 \text{diag}\left(1 - \tanh^2(\mathbf{u})\right) \mathbf{W}_1 \mathbf{A}_{\text{norm}}^T - \mathbf{D}$$

Because $|\tanh'(u_i)| = 1 - \tanh^2(u_i) \le 1$, the operator 2-norm of the diagonal activation Jacobian satisfies $\|\text{diag}(1 - \tanh^2(\mathbf{u}))\|_2 \le 1$. By submultiplicativity of induced matrix norms:
$$\|\mathbf{J}(\mathbf{z}) + \mathbf{D}\|_2 \le \|\mathbf{W}_2\|_2 \cdot \|\text{diag}(1 - \tanh^2(\mathbf{u}))\|_2 \cdot \|\mathbf{W}_1\|_2 \cdot \|\mathbf{A}_{\text{norm}}\|_2 \le \|\mathbf{W}_1\|_2 \|\mathbf{W}_2\|_2 \|\mathbf{A}_{\text{norm}}\|_2$$

Because $\lambda_{\max}(\text{Sym}(\mathbf{M})) \le \|\mathbf{M}\|_2$ for any real matrix $\mathbf{M}$ and $\mathbf{D} = \text{diag}(d_1, \dots, d_d)$ with $d_i \ge d_{\min}$:
$$\lambda_{\max}(\text{Sym}(\mathbf{J}(\mathbf{z}))) \le \|\mathbf{A}_{\text{norm}}\|_2 \cdot \|\mathbf{W}_1\|_2 \cdot \|\mathbf{W}_2\|_2 - d_{\min}$$

With spectral normalization clipping $\|\mathbf{W}_1\|_2 \le 1$ and $\|\mathbf{W}_2\|_2 \le 1$:
$$\lambda_{\max}(\text{Sym}(\mathbf{J})) \le \|\mathbf{A}_{\text{norm}}\|_2 - d_{\min}$$
Strict Demidovich contraction is guaranteed whenever the damping coefficient satisfies $d_{\min} > \|\mathbf{A}_{\text{norm}}\|_2$.

### S1.3 Impossibility of Parity Representation via Convex Potentials
In Approach A, the dynamical flow is defined as the negative gradient of an Input-Convex Neural Network (ICNN) plus coordinate damping:
$$\dot{\mathbf{z}} = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{x}) - \mathbf{D}\mathbf{z}$$

The unique fixed point $\mathbf{z}^*(\mathbf{x})$ corresponds to the global minimizer of the strictly convex surrogate:
$$V(\mathbf{z}; \mathbf{x}) = E_\theta(\mathbf{z}; \mathbf{x}) + \frac{1}{2}\mathbf{z}^T \mathbf{D} \mathbf{z}$$

For $N$-bit parity on the Boolean hypercube $\mathbf{x} \in \{-1, +1\}^N$, the target label is $y = \prod_{i=1}^N x_i$. The input domain partitions into $2^{N-1}$ positive strings and $2^{N-1}$ negative strings, where adjacent Hamming neighbors always have opposite parity labels. A strictly convex potential possesses no saddle points or local minima other than its unique global minimum. Therefore, the mapping $\mathbf{x} \mapsto \mathbf{z}^*(\mathbf{x})$ cannot construct non-convex, alternating decision regions. The learned potential collapses to an input-insensitive carrier, achieving 50.0% chance accuracy and invoking the pre-registered kill criterion.

---

## Section S2: High-Dimensional Shifted Power Iteration Algorithm

To verify strict Demidovich contraction on the full 43,328-dimensional state space ($N_{\text{nodes}} = 2708, d = 16$), explicitly constructing the $43,328 \times 43,328$ Jacobian matrix requires over 7.5 GB of RAM in float32. We introduce a matrix-free shifted power iteration:
$$\mathbf{M} = \frac{1}{2}\left(\mathbf{J}(\mathbf{z}) + \mathbf{J}(\mathbf{z})^T\right) + c \mathbf{I}, \quad c = 10.0$$

For any probe vector $\mathbf{v} \in \mathbb{R}^{43,328}$, matrix-vector products are evaluated without matrix materialization:
1. **Directional Derivative**: $\mathbf{J}(\mathbf{z})\mathbf{v} \approx \frac{\mathbf{f}(\mathbf{z} + \epsilon \mathbf{v}) - \mathbf{f}(\mathbf{z} - \epsilon \mathbf{v})}{2\epsilon}$ with $\epsilon = 10^{-5}$.
2. **Vector-Jacobian Product (VJP)**: $\mathbf{J}(\mathbf{z})^T \mathbf{v} = \nabla_\mathbf{z} (\mathbf{f}(\mathbf{z})^T \mathbf{v})$ computed via reverse-mode PyTorch autograd.

Iteration updates:
$$\mathbf{v}^{(k+1)} = \frac{\mathbf{M} \mathbf{v}^{(k)}}{\|\mathbf{M} \mathbf{v}^{(k)}\|_2}, \quad \mu^{(k)} = {\mathbf{v}^{(k)}}^T \mathbf{M} \mathbf{v}^{(k)}$$
$$\lambda_{\max}(\text{Sym}(\mathbf{J})) = \mu^{(K)} - c$$

Empirical result on the full Cora network:
$$\lambda_{\max}(\text{Sym}(\mathbf{J})) = \mathbf{-1.88987} \le \|\mathbf{A}_{\text{norm}}\|_2 - d_{\min} = 1.000000 - 1.9814 = \mathbf{-0.9814 < 0}$$
Runtime: **0.22 seconds** across 15 iterations.

---

## Section S3: Dataset Topology & Zero-Leakage Split Protocol

The Cora citation graph contains 2,708 documents across 7 subject classes with 5,429 directed citation edges and 1,433-dimensional word vectors.
- **Stratified Partitioning**: Positive reachability pairs are stratified across exact shortest-path hop distances 1 through 6.
- **Unreachable Pair Sampling**: Negative pairs are sampled uniformly from the complement of directed reachability ($d(u, v) = \infty$).
- **Disjoint Split Partitioning**:
  - Training: 1,200 pairs (50% positive / 50% negative)
  - Validation: 300 pairs (50% positive / 50% negative)
  - Test: 300 pairs (50% positive / 50% negative)
  - $\text{train} \cap \text{val} = \text{train} \cap \text{test} = \text{val} \cap \text{test} = \emptyset$ (strictly zero data leakage).

---

## Section S4: Comprehensive Experimental Tables & Decision Margins

### Table S1: Architectural & Training Hyperparameters
| Hyperparameter / Configuration | Contractive Latent Reasoner (CLR) | Discrete GNN Baselines ($K=2..8$) |
| :--- | :--- | :--- |
| Latent State Dimension ($d$) | 16 | 16 |
| Input Feature Dimension ($d_x$) | 16 (one-hot source impulse; 1433-dim raw Cora BoW not consumed) | 16 (one-hot source impulse; 1433-dim raw Cora BoW not consumed) |
| Numerical Integrator | Explicit 4th-Order Runge-Kutta (RK4) | Discrete Layer Stacking ($K$ hops) |
| Integration Horizon ($T$) | 4.0 (scaling to 16.0) | Fixed $K \in \{2, 4, 6, 8\}$ |
| Integration Step Size ($dt$) | 0.1667 (24 integration steps) | N/A (discrete) |
| Contraction Damping ($d_{\min}$) | 1.9814 | N/A |
| Spectral Normalization Bound | 1.0000 (Power Iteration / SVD clipping) | Standard Xavier initialization |
| Optimizer & Learning Rate | AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$) | AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$) |
| Readout Network | 2-Layer MLP ($16 \to 32 \to 2$) | 2-Layer MLP ($16 \to 32 \to 2$) |

### Table S2: Graded Noise Perturbation Sweep Across 8 Orders of Magnitude
| Noise Scale ($\sigma$) | Test Accuracy | Predicted Unreachable % | Reachable State Log-Norm | Unreachable State Log-Norm |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 (Clean) | 100.00% | 50.0% | -8.91 | -27.63 |
| $10^{-8}$ | 100.00% | 50.0% | -8.91 | -21.02 |
| $10^{-7}$ | 92.67% | 42.7% | -8.91 | -18.72 |
| $10^{-6}$ | 50.00% | 0.0% | -8.89 | -16.40 |
| $10^{-5}$ | 50.00% | 0.0% | -8.82 | -14.15 |
| $10^{-4}$ | 50.00% | 0.0% | -8.45 | -11.89 |
| $10^{-3}$ | 50.00% | 0.0% | -6.80 | -9.54 |
| $10^{-2}$ | 50.00% | 0.0% | -4.20 | -7.10 |

### Table S3: Statistical Audit of Test-Set Decision Margins ($N=300$)
| Metric / Percentile | Value (Logit Gap) | Scientific Interpretation |
| :--- | :---: | :--- |
| Minimum Decision Margin | 0.243 | No knife-edge decisions (all margins $\gg 0$) |
| 25th Percentile | 2.140 | Robust separation on intermediate hops |
| Median Margin | 3.290 | Strong confident separation |
| 75th Percentile | 3.980 | High confidence on long-range reachability |
| Pairs with margin $< 0.01$ | 0 / 300 (0.0%) | Complete absence of borderline classifications |

---

## Section S5: The Physics of Graph Self-Healing & Energy Floor Elevation

While contractive dynamical systems exponentially attenuate deviations between trajectories ($\|\delta \mathbf{z}(t)\| \le \|\delta \mathbf{z}(0)\| e^{-\kappa t}$), graph reachability classification depends on an asymmetric boundary:
1. Reachable nodes converge to active equilibrium states with mean log-norm $-8.91$.
2. Unreachable nodes have mathematically identical zero states ($\|\mathbf{z}_v\| = 0$), mapping to $-27.63$ under $\ln(\|\mathbf{z}_v\| + 10^{-12})$.
3. The readout MLP places a sharp threshold at $\approx -17.5$.
4. Under uniform Gaussian perturbations $\sigma \ge 10^{-6}$ injected across all 2,708 graph nodes, the residual noise energy creates a non-zero background floor of $-16.40 > -17.5$ on unreachable nodes.
5. Consequently, all unreachable pairs are classified as reachable, yielding flat 50.00% accuracy on a balanced test set. This establishes that naive self-healing does not hold when decision boundaries rely on absolute zero-energy detection.

---

## Section S6: Software Environment, Reproducibility & Hardware Specifications

- **Software**: Python 3.12.14, PyTorch 2.1+, NumPy 1.26+, SciPy 1.13+, Matplotlib 3.11+, python-docx 1.2.0, latex2mathml 3.81.1, mathml2omml 0.0.2.
- **Random Seeding**: Seed 42 (data generation), Seed 123 (model weights), Seed 456 (evaluation).
- **Hardware & Benchmark Runtime**: Apple Silicon M-series (macOS 15.x). 0.22s for contraction verification, 430s for Cora model training.
- **Open Access Repository**: [https://github.com/Dips7/contractive-latent-reasoning](https://github.com/Dips7/contractive-latent-reasoning)

---

## Section S7: Conflict of Interest & Compliance Statements

### Conflict of Interest
The authors declare that they have no known competing financial interests, personal relationships, or professional affiliations that could have appeared to influence or bias the work, findings, and interpretations reported in this paper.

### Funding Statement
This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
