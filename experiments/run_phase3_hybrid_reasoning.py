"""
Phase 3 Master Experiment:
Hybrid Interface — Continuous Thought to Discrete Language Token Decoder.

Architecture:
1. Input problem x (Graph G, source u, target v)
2. Continuous Contractive Latent Flow: dz/dt = f(z; x) -> Contracted vector z*(x)
3. Projection Bridge: e_thought = W_p * z*(x) + b_p
4. Autoregressive Transformer Decoder:
   Conditioned on [e_thought], generates natural language tokens:
   "Path exists: YES" or "Path exists: NO"
   without intermediate chain-of-thought tokens!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from experiments.run_graph_reachability_experiment import (
    generate_balanced_graph_dataset,
    unpack_graph_features,
)
from src.dynamics.solvers import solve_ode_rk4
from src.utils import set_seed, save_experiment_results

# Vocabulary definitions
VOCAB = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "Path": 3,
    "exists": 4,
    ":": 5,
    "YES": 6,
    "NO": 7,
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def tokens_to_text(token_ids):
    words = [INV_VOCAB.get(idx, f"<{idx}>") for idx in token_ids]
    # Filter out special tokens for clean display
    clean_words = [w for w in words if w not in ["<PAD>", "<BOS>", "<EOS>"]]
    return " ".join(clean_words)


def build_target_sequences(labels: torch.Tensor):
    """
    Maps binary reachability labels {0, 1} to discrete token sequences:
    Label 1 -> [<BOS>, Path, exists, :, YES, <EOS>]
    Label 0 -> [<BOS>, Path, exists, :, NO, <EOS>]
    """
    seqs = []
    for y in labels:
        if y.item() == 1:
            seq = [VOCAB["<BOS>"], VOCAB["Path"], VOCAB["exists"], VOCAB[":"], VOCAB["YES"], VOCAB["<EOS>"]]
        else:
            seq = [VOCAB["<BOS>"], VOCAB["Path"], VOCAB["exists"], VOCAB[":"], VOCAB["NO"], VOCAB["<EOS>"]]
        seqs.append(seq)
    return torch.tensor(seqs, dtype=torch.long)


# ----------------------------------------------------------------------
# Full Hybrid Model: Continuous Latent Reasoner + Discrete Token Decoder
# ----------------------------------------------------------------------
class HybridContractiveLanguageModel(nn.Module):
    def __init__(
        self,
        num_nodes: int = 16,
        node_dim: int = 16,
        model_dim: int = 64,
        vocab_size: int = VOCAB_SIZE,
        num_decoder_layers: int = 2,
        min_damping: float = 1.5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.model_dim = model_dim
        self.min_damping = min_damping
        
        # Continuous Reasoner Parameters (Approach B)
        self.w1 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.w2 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.log_d = nn.Parameter(torch.ones(node_dim) * 0.0)
        
        # Thought Projection Bridge: z*(x) -> e_thought in R^{model_dim}
        self.thought_bridge = nn.Sequential(
            nn.Linear(node_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        
        # Discrete Token Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(32, model_dim)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def get_damping(self):
        return F.softplus(self.log_d) + self.min_damping

    def reason_continuous(self, x, t_span):
        """Runs the continuous ODE flow to reach contracted thought state z*."""
        batch_size = x.shape[0]
        device = x.device
        A, u_onehot, v_onehot = unpack_graph_features(x, self.num_nodes)
        
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / deg
        
        source = torch.zeros(batch_size, self.num_nodes, self.node_dim, device=device)
        source[:, :, 0] = u_onehot
        z0 = source.clone()
        d = self.get_damping()
        
        def vf(t, z):
            z_prop = torch.bmm(A_norm.transpose(1, 2), z)
            flow = self.w2(torch.tanh(self.w1(z_prop)))
            return flow - d * z + source
            
        t_final = t_span[-1].item() if isinstance(t_span[-1], torch.Tensor) else float(t_span[-1])
        steps = max(25, int(t_final * 10))
        traj = solve_ode_rk4(vf, z0, t_span, steps=steps)
        z_star = traj[-1] # (B, N, node_dim)
        
        # Extract target node v's representation
        v_state = (z_star * v_onehot.unsqueeze(-1)).sum(dim=1) # (B, node_dim)
        
        # Project through bridge to form thought prefix embedding
        e_thought = self.thought_bridge(v_state) # (B, model_dim)
        return e_thought

    def forward(self, x, target_tokens, t_span):
        """
        Teacher-forced training forward pass.
        target_tokens: (B, seq_len) including <BOS> ... <EOS>
        """
        batch_size, seq_len = target_tokens.shape
        device = x.device
        
        # 1. Continuous Thought Generation
        e_thought = self.reason_continuous(x, t_span) # (B, model_dim)
        thought_prefix = e_thought.unsqueeze(1) # (B, 1, model_dim)
        
        # 2. Token Embeddings (exclude the final <EOS> for teacher forcing input)
        input_tokens = target_tokens[:, :-1] # (B, seq_len - 1)
        in_len = input_tokens.shape[1]
        
        tok_emb = self.token_embedding(input_tokens)
        pos = torch.arange(in_len, device=device).unsqueeze(0).expand(batch_size, in_len)
        pos_emb = self.pos_embedding(pos)
        x_tok = tok_emb + pos_emb # (B, in_len, model_dim)
        
        # 3. Concatenate Thought Prefix at index 0: [thought_prefix, input_tokens]
        # full sequence length = 1 + in_len = seq_len
        full_seq = torch.cat([thought_prefix, x_tok], dim=1) # (B, seq_len, model_dim)
        
        # Causal attention mask
        total_len = full_seq.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(total_len, device=device)
        
        out = self.decoder(tgt=full_seq, memory=thought_prefix, tgt_mask=causal_mask)
        logits = self.lm_head(out) # (B, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, x, t_span, max_len: int = 8):
        """
        Autoregressively generate tokens conditioned on continuous thought state z*.
        """
        self.eval()
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Continuous Reasoning
        e_thought = self.reason_continuous(x, t_span)
        thought_prefix = e_thought.unsqueeze(1) # (B, 1, model_dim)
        
        # Start with <BOS>
        current_tokens = torch.full((batch_size, 1), VOCAB["<BOS>"], dtype=torch.long, device=device)
        generated_sequences = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        for step in range(max_len):
            in_len = current_tokens.shape[1]
            tok_emb = self.token_embedding(current_tokens)
            pos = torch.arange(in_len, device=device).unsqueeze(0).expand(batch_size, in_len)
            pos_emb = self.pos_embedding(pos)
            x_tok = tok_emb + pos_emb
            
            full_seq = torch.cat([thought_prefix, x_tok], dim=1)
            total_len = full_seq.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(total_len, device=device)
            
            out = self.decoder(tgt=full_seq, memory=thought_prefix, tgt_mask=causal_mask)
            logits = self.lm_head(out[:, -1, :]) # predict next token from last position
            next_token = torch.argmax(logits, dim=-1) # (B,)
            
            for b in range(batch_size):
                if not finished[b]:
                    tok = next_token[b].item()
                    generated_sequences[b].append(tok)
                    if tok == VOCAB["<EOS>"]:
                        finished[b] = True
                        
            if all(finished):
                break
                
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
            
        return generated_sequences


def main():
    print("=" * 85)
    print("PHASE 3: HYBRID INTERFACE — CONTINUOUS THOUGHT TO DISCRETE LANGUAGE DECODER")
    print("=" * 85)
    
    set_seed(42)
    device = torch.device("cpu")
    num_nodes = 16
    
    print("\n[Step 1] Generating Problem Graphs and Target Discrete Text Sequences...")
    x_train, y_train = generate_balanced_graph_dataset(num_samples=3000, num_nodes=num_nodes, seed=42)
    x_val, y_val = generate_balanced_graph_dataset(num_samples=600, num_nodes=num_nodes, seed=100)
    
    target_train = build_target_sequences(y_train)
    target_val = build_target_sequences(y_val)
    
    print("Sample Problem -> Target Sentence Pairs:")
    for i in range(3):
        print(f"  Sample {i}: Label={y_train[i].item()} -> Target Text: '{tokens_to_text(target_train[i].tolist())}'")
        
    train_loader = DataLoader(TensorDataset(x_train, target_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, target_val), batch_size=32, shuffle=False)
    
    # ------------------------------------------------------------------
    # Step 2: Initialize & Train Hybrid Model (Two-Stage Regime)
    # ------------------------------------------------------------------
    print("\n[Step 2] Training Hybrid Model via Two-Stage Regime (Stage 1: Latent Flow; Stage 2: Token Decoder)...")
    model = HybridContractiveLanguageModel(
        num_nodes=num_nodes,
        node_dim=16,
        model_dim=64,
        vocab_size=VOCAB_SIZE,
        num_decoder_layers=2,
        min_damping=1.5,
    ).to(device)
    
    start_time = time.time()
    t_train = torch.tensor([0.0, 4.0])
    
    # Stage 1: Train continuous reasoner on reachability
    print("  [Stage 1/2] Training Continuous Contractive Flow on Graph Reachability (5 epochs)...")
    aux_readout = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 2)).to(device)
    opt_stage1 = torch.optim.AdamW(
        list(model.w1.parameters()) + list(model.w2.parameters()) + [model.log_d] + list(aux_readout.parameters()),
        lr=8e-3, weight_decay=1e-4
    )
    train_loader_aux = DataLoader(TensorDataset(x_train, target_train, y_train), batch_size=32, shuffle=True)
    for ep in range(1, 6):
        model.train()
        tot_loss = 0.0
        corr = 0
        tot = 0
        for bx, _, by_lbl in train_loader_aux:
            opt_stage1.zero_grad()
            A, u_onehot, v_onehot = unpack_graph_features(bx, num_nodes)
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
            A_norm = A / deg
            src = torch.zeros(bx.size(0), num_nodes, 16, device=device)
            src[:, :, 0] = u_onehot
            d = model.get_damping()
            def vf(t, z):
                return model.w2(torch.tanh(model.w1(torch.bmm(A_norm.transpose(1, 2), z)))) - d * z + src
            traj = solve_ode_rk4(vf, src, t_train, steps=25)
            v_state = (traj[-1] * v_onehot.unsqueeze(-1)).sum(dim=1)
            logits_aux = aux_readout(v_state)
            loss = F.cross_entropy(logits_aux, by_lbl)
            loss.backward()
            opt_stage1.step()
            tot_loss += loss.item() * bx.size(0)
            corr += (logits_aux.argmax(-1) == by_lbl).sum().item()
            tot += bx.size(0)
        print(f"    Epoch {ep:02d} | Flow Reachability Loss: {tot_loss/tot:.4f} | Flow Reachability Acc: {corr/tot*100:.2f}%")
        
    # Stage 2: Train thought bridge & Transformer token decoder
    print("  [Stage 2/2] Training Thought Bridge & Discrete Token Decoder (5 epochs)...")
    opt_stage2 = torch.optim.AdamW(
        list(model.thought_bridge.parameters()) + list(model.token_embedding.parameters()) +
        list(model.pos_embedding.parameters()) + list(model.decoder.parameters()) + list(model.lm_head.parameters()),
        lr=4e-3, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<PAD>"])
    for ep in range(1, 6):
        model.train()
        tot_loss = 0.0
        tot_tokens = 0
        correct_tokens = 0
        for bx, by in train_loader:
            opt_stage2.zero_grad()
            logits = model(bx, by, t_train)
            loss = criterion(logits.view(-1, VOCAB_SIZE), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_stage2.step()
            tot_loss += loss.item() * by.size(0)
            preds = logits.argmax(-1)
            correct_tokens += (preds == by).sum().item()
            tot_tokens += by.numel()
        token_acc = correct_tokens / tot_tokens
        print(f"    Epoch {ep:02d} | Token LM Loss: {tot_loss/len(train_loader.dataset):.4f} | Token Acc: {token_acc*100:.2f}%")
               
    train_time = time.time() - start_time
    print(f"\nTwo-stage hybrid training completed in {train_time:.2f}s.")
    
    # ------------------------------------------------------------------
    # Step 3: Autoregressive Discrete Text Generation Test
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("[Step 3] Autoregressive Text Generation from Contracted Latent Thoughts (T=6.0):")
    print("=" * 85)
    
    t_eval = torch.tensor([0.0, 6.0])
    test_batch_x = x_val[:10]
    test_batch_y = y_val[:10]
    test_targets = target_val[:10]
    
    generated_ids = model.generate(test_batch_x, t_eval, max_len=8)
    
    exact_matches = 0
    for i in range(10):
        gen_text = tokens_to_text(generated_ids[i])
        tgt_text = tokens_to_text(test_targets[i].tolist())
        is_correct = (gen_text == tgt_text)
        if is_correct:
            exact_matches += 1
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"  Query {i:02d} | Target: '{tgt_text:<18}' | Generated: '{gen_text:<18}' | {status}")
        
    print(f"\nSample Exact Sentence Match: {exact_matches}/10 ({exact_matches*10}%)")
    
    # ------------------------------------------------------------------
    # Step 4: Test-Time Compute Scaling on Discrete Text Generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("[Step 4] Continuous Test-Time Scaling on Discrete Sentence Accuracy (600 Val Graphs):")
    print("=" * 85)
    print(f"{'Integration Horizon T':<25} | {'Exact Sentence Accuracy':<25} | {'Scaling Progression'}")
    print("-" * 85)
    
    val_sub_x = x_val[:200]
    val_sub_targets = target_val[:200]
    scaling_data = []
    
    for t_val in [0.5, 1.0, 2.0, 4.0, 6.0, 10.0]:
        t_span = torch.tensor([0.0, t_val])
        gen_ids = model.generate(val_sub_x, t_span, max_len=8)
        
        matches = 0
        for i in range(len(val_sub_x)):
            gen_text = tokens_to_text(gen_ids[i])
            tgt_text = tokens_to_text(val_sub_targets[i].tolist())
            if gen_text == tgt_text:
                matches += 1
        acc = (matches / len(val_sub_x)) * 100
        scaling_data.append({"T": t_val, "sentence_accuracy": float(acc)})
        bar = "█" * int(acc / 4)
        print(f"Horizon T = {t_val:<13.1f} | {acc:6.2f}%                    | {bar}")
        
    print("=" * 85)

    save_experiment_results(
        experiment_name="phase3_hybrid_reasoning",
        metrics={
            "exact_sentence_matches_sample": exact_matches,
            "test_time_scaling": scaling_data,
        },
        config={
            "num_nodes": num_nodes,
            "seed": 42,
        },
        seed=42,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
