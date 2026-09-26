# System Architecture Specification

## Overview

The Contractive Latent Reasoning (CLR) system decouples the **depth of reasoning** from the **number of generated tokens**.

### Data Flow

```
Input Tokens / Vectors x
        │
        ▼
   [Encoder] ──────> Conditioning Context c(x) ∈ ℝ^{d_c}
                            │
                            ▼
               [Continuous Latent Block]
                     z(0) = 0
                     dz/dt = -∇_z E_θ(z; c) - D·z
                     t ∈ [0, T]
                            │
                            ▼
               Contracted Thought Vector z* ∈ ℝ^{d_z}
                            │
                            ▼
               [Projection Bridge Head]
                            │
                            ▼
               Output Tokens / Decision Class y*
```

### Module Contracts

1. **`SequenceEncoder`**:
   - Signature: `forward(x: Tensor) -> c: Tensor`
   - Input: `(B, L, D_in)` or `(B, L)`
   - Output: `(B, D_c)`

2. **`DampedGradientFlowField`**:
   - Signature: `forward(t: Tensor, z: Tensor) -> dz_dt: Tensor`
   - Requires `set_context(c: Tensor)` prior to ODE integration.
   - Computes: $\frac{d\mathbf{z}}{dt} = -\nabla_\mathbf{z} E_\theta(\mathbf{z}; \mathbf{c}) - \mathbf{D}\mathbf{z}$.

3. **`ODESolverWrapper`**:
   - Signature: `forward(vector_field, z0, t_span) -> trajectory: Tensor`
   - Output shape: `(len(t_span), B, D_z)`.
   - Terminal state $z^* = \text{trajectory}[-1]$.
