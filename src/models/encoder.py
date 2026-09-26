"""Encoders mapping input x to conditioning context c(x)."""

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """
    Encodes sequence inputs (e.g., bit strings or token sequences) into a continuous context vector c.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        context_dim: int = 128,
        encoder_type: str = "gru",
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.proj = nn.Linear(hidden_dim * 2, context_dim)
        elif encoder_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, context_dim),
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, seq_len)
        Returns:
            c: Context vector (batch, context_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        if self.encoder_type == "gru":
            _, h_n = self.rnn(x)
            # Concatenate forward and backward final hidden states
            h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            c = self.proj(h_cat)
        elif self.encoder_type == "mlp":
            # Pool across sequence dimension
            h = self.mlp(x)
            c = torch.mean(h, dim=1)
        return c
