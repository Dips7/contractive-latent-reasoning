"""Decoder heads for reading out the contracted state z*."""

import torch
import torch.nn as nn


class ClassificationDecoder(nn.Module):
    """
    Direct linear or MLP classification readout from z*.
    """
    def __init__(self, latent_dim: int, num_classes: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_star: torch.Tensor) -> torch.Tensor:
        return self.head(z_star)


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive transformer decoder conditioned on soft thought prefix e_thought = Bridge(z*).
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        model_dim: int = 256,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_bridge = nn.Linear(latent_dim, model_dim)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def forward(self, z_star: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_star: Contracted latent state (batch, latent_dim)
            target_tokens: Ground-truth tokens (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = target_tokens.shape
        device = z_star.device
        
        # Thought vector prefix: (batch, 1, model_dim)
        thought_prefix = self.latent_bridge(z_star).unsqueeze(1)
        
        # Token embeddings
        tok_emb = self.token_embedding(target_tokens)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb
        
        # Concatenate thought prefix to sequence: (batch, 1 + seq_len, model_dim)
        full_seq = torch.cat([thought_prefix, x], dim=1)
        
        # Causal mask for autoregressive generation
        total_len = full_seq.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(total_len, device=device)
        
        out = self.decoder(tgt=full_seq, memory=thought_prefix, tgt_mask=causal_mask)
        # Project tokens back to vocabulary
        logits = self.lm_head(out[:, 1:, :]) # exclude prefix position
        return logits
