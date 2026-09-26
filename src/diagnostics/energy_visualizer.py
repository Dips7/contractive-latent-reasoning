"""2D and 3D Phase portrait visualizers."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def plot_2d_phase_portrait(
    vector_field: nn.Module,
    context_2d: torch.Tensor,
    z_range: float = 3.0,
    grid_res: int = 25,
    trajectories: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
):
    """
    Plots a 2D quiver plot of the vector field and overlays trajectory paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(-z_range, z_range, grid_res)
    y = np.linspace(-z_range, z_range, grid_res)
    X, Y = np.meshgrid(x, y)
    
    Z_grid = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)
    c_expand = context_2d.expand(Z_grid.shape[0], -1)
    
    vector_field.eval()
    vector_field.set_context(c_expand)
    t = torch.tensor(0.0)
    
    with torch.no_grad():
        dZ = vector_field(t, Z_grid).numpy()
        
    U = dZ[:, 0].reshape(grid_res, grid_res)
    V = dZ[:, 1].reshape(grid_res, grid_res)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.streamplot(X, Y, U, V, color="lightblue", density=1.2, arrowsize=1.2)
    
    if trajectories is not None:
        # trajectories: (T, num_trajectories, 2)
        traj_np = trajectories.detach().cpu().numpy()
        for i in range(traj_np.shape[1]):
            ax.plot(traj_np[:, i, 0], traj_np[:, i, 1], lw=2.0)
            ax.scatter(traj_np[0, i, 0], traj_np[0, i, 1], color="green", s=30)
            ax.scatter(traj_np[-1, i, 0], traj_np[-1, i, 1], color="red", marker="x", s=50)
            
    ax.set_title("Contractive Phase Portrait & Attractor Dynamics")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
