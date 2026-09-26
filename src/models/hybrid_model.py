"""End-to-End Hybrid Reasoning Architecture."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .encoder import SequenceEncoder
from .latent_reasoner import ContractiveLatentReasoner
from .decoder import ClassificationDecoder, AutoregressiveDecoder


class HybridReasoningModel(nn.Module):
    """
    Unified end-to-end model:
    Input x -> Encoder -> Context c -> Contractive Latent ODE -> z* -> Decoder -> Output y
    """
    def __init__(
        self,
        input_dim: int = 1,
        context_dim: int = 128,
        latent_dim: int = 128,
        num_classes: int = 2,
        is_autoregressive: bool = False,
        vocab_size: int = 1000,
        solver_method: str = "dopri5",
        solver_step_size: Optional[float] = None,
    ):
        super().__init__()
        self.is_autoregressive = is_autoregressive
        
        # 1. Encoder
        self.encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_dim=context_dim,
            context_dim=context_dim,
            encoder_type="gru",
        )
        
        # 2. Continuous Latent Reasoner
        self.latent_reasoner = ContractiveLatentReasoner(
            latent_dim=latent_dim,
            context_dim=context_dim,
            solver_method=solver_method,
            step_size=solver_step_size,
        )
        
        # 3. Readout Decoder
        if is_autoregressive:
            self.decoder = AutoregressiveDecoder(
                vocab_size=vocab_size,
                model_dim=context_dim,
                latent_dim=latent_dim,
            )
        else:
            self.decoder = ClassificationDecoder(
                latent_dim=latent_dim,
                num_classes=num_classes,
            )

    def forward(
        self,
        x: torch.Tensor,
        t_span: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        """
        # Encode problem context
        c = self.encoder(x)
        
        # Continuous latent reasoning
        z_star, trajectory = self.latent_reasoner(c, t_span=t_span, z0=z0)
        
        # Readout
        if self.is_autoregressive:
            if target_tokens is None:
                raise ValueError("target_tokens must be provided for autoregressive decoder.")
            logits = self.decoder(z_star, target_tokens)
        else:
            logits = self.decoder(z_star)
            
        return {
            "logits": logits,
            "z_star": z_star,
            "trajectory": trajectory,
            "context": c,
        }
