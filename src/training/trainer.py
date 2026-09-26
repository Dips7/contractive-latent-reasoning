"""Trainer engine for contractive latent models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os


class Trainer:
    """
    Standard PyTorch training and evaluation loop.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip

    def train_epoch(
        self,
        dataloader: DataLoader,
        t_span: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x, y = batch
            elif isinstance(batch, dict):
                x, y = batch["x"], batch["y"]
                
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(x, t_span=t_span)
            v_field = getattr(self.model, "latent_reasoner", None)
            if v_field is not None:
                v_field = v_field.vector_field
                
            loss_dict = self.loss_fn(outputs, y, vector_field=v_field)
            loss = loss_dict["total_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Training loss diverged to non-finite value: {loss.item()}")
            
            loss.backward()
            if self.grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if not torch.isfinite(total_norm):
                    raise RuntimeError(f"Gradient norm exploded to non-finite value: {total_norm}")
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs["logits"], dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
        return {
            "loss": total_loss / max(1, total_samples),
            "accuracy": total_correct / max(1, total_samples),
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        t_span: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                elif isinstance(batch, dict):
                    x, y = batch["x"], batch["y"]
                    
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x, t_span=t_span)
                
                loss_dict = self.loss_fn(outputs, y)
                loss = loss_dict["total_loss"]
                
                total_loss += loss.item() * x.size(0)
                preds = torch.argmax(outputs["logits"], dim=-1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)
                
        return {
            "loss": total_loss / max(1, total_samples),
            "accuracy": total_correct / max(1, total_samples),
        }
