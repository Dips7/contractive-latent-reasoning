"""Integration time and curriculum schedulers."""

import torch


class IntegrationTimeScheduler:
    """
    Curriculum scheduler that gradually increases the integration horizon T during training.
    """
    def __init__(self, t_init: float = 1.0, t_final: float = 5.0, warmup_epochs: int = 20):
        self.t_init = t_init
        self.t_final = t_final
        self.warmup_epochs = warmup_epochs

    def get_t_span(self, epoch: int, device: torch.device) -> torch.Tensor:
        """Returns torch.tensor([0.0, current_T])."""
        if epoch >= self.warmup_epochs:
            current_t = self.t_final
        else:
            alpha = epoch / max(1, self.warmup_epochs)
            current_t = self.t_init + alpha * (self.t_final - self.t_init)
        return torch.tensor([0.0, current_t], device=device)
