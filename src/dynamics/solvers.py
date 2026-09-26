"""ODE solver integration wrappers."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def explicit_rk4_step(f, t, y, dt):
    """Single step of explicit classic 4th order Runge-Kutta, supporting f(t, y) or f(y)."""
    try:
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)
    except TypeError:
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ode_rk4(
    vector_field,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    steps: Optional[int] = None,
    substeps_per_interval: int = 5,
) -> torch.Tensor:
    """
    Classic 4th order Runge-Kutta integrator strictly conforming to the
    (len(t_span), *y0.shape) contract matching torchdiffeq.odeint.
    
    Args:
        vector_field: Callable (t, y) -> dy/dt
        y0: Initial state tensor of arbitrary shape (*shape)
        t_span: 1D tensor or iterable of time points [t_0, t_1, ..., t_{K-1}]
        steps: Optional total number of internal RK4 sub-steps across all intervals
        substeps_per_interval: Number of internal RK4 sub-steps per interval [t_i, t_{i+1}]
            (used if `steps` is None, or overridden to guarantee accuracy)
            
    Returns:
        Tensor of shape (len(t_span), *y0.shape)
    """
    if len(t_span) == 0:
        raise ValueError("t_span must contain at least 1 time point.")
    if len(t_span) == 1:
        return y0.unsqueeze(0)

    n_intervals = len(t_span) - 1
    if steps is not None:
        substeps = max(1, steps // n_intervals)
    else:
        substeps = max(1, substeps_per_interval)

    trajectory = [y0]
    y = y0
    for i in range(n_intervals):
        t_start = t_span[i]
        t_end = t_span[i + 1]
        dt_interval = (t_end - t_start).item() if isinstance(t_end - t_start, torch.Tensor) else float(t_end - t_start)
        
        if dt_interval == 0.0:
            trajectory.append(y)
            continue
            
        dt = dt_interval / substeps
        t_curr = t_start.item() if isinstance(t_start, torch.Tensor) else float(t_start)
        
        for _ in range(substeps):
            t_tensor = torch.tensor(t_curr, device=y.device, dtype=y.dtype)
            y = explicit_rk4_step(vector_field, t_tensor, y, dt)
            t_curr += dt
            
        trajectory.append(y)

    return torch.stack(trajectory, dim=0)


class ODESolverWrapper(nn.Module):
    """
    Wrapper around numerical ODE integration (torchdiffeq with RK4 fallback).
    Defaults to adaptive 'dopri5' Runge-Kutta 4(5) Dormand-Prince method.
    When a fixed-step method ('rk4', 'euler') is selected, a safe step_size
    is explicitly supplied via options={'step_size': dt} to prevent single-step solves.
    """
    def __init__(
        self,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        use_adjoint: bool = True,
        step_size: Optional[float] = 0.05,
        options: Optional[dict] = None,
    ):
        super().__init__()
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self.step_size = step_size
        self.options = options or {}
        if step_size is not None and "step_size" not in self.options:
            self.options["step_size"] = step_size
        
        # Check torchdiffeq availability
        try:
            import torchdiffeq
            self.has_torchdiffeq = True
            self.odeint_fn = torchdiffeq.odeint_adjoint if use_adjoint else torchdiffeq.odeint
        except ImportError:
            self.has_torchdiffeq = False
            self.odeint_fn = None

    def forward(
        self,
        vector_field: nn.Module,
        z0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate dz/dt = f(t, z) from t_span[0] to t_span[-1].
        
        Returns:
            trajectory: (len(t_span), batch_size, latent_dim)
        """
        if self.has_torchdiffeq:
            opts = dict(self.options)
            # For fixed-step methods, ensure step_size is present in options
            if self.method in ["rk4", "euler", "midpoint", "explicit_adams", "implicit_adams"]:
                if "step_size" not in opts or opts["step_size"] is None:
                    opts["step_size"] = self.step_size or 0.05
            else:
                # Adaptive methods like dopri5 do not take step_size in options
                opts.pop("step_size", None)
            return self.odeint_fn(
                vector_field,
                z0,
                t_span,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                options=opts if len(opts) > 0 else None,
            )
        else:
            # Fallback internal RK4
            substeps = 10
            if "step_size" in self.options and self.options["step_size"] is not None:
                dt_target = self.options["step_size"]
                avg_interval = abs((t_span[-1] - t_span[0]).item()) / max(1, len(t_span) - 1)
                substeps = max(2, int(round(avg_interval / dt_target)))
            return solve_ode_rk4(vector_field, z0, t_span, substeps_per_interval=substeps)
