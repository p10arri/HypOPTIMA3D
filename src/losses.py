import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from loguru import logger

from src.utils.hyperbolic_function import HyperbolicSpace
from src.utils.enums import TrainingMode, Space, ModelVariant


def get_loss(cfg, training_mode: TrainingMode) -> nn.Module:
    space = Space(cfg.space.name)
    model_variant = ModelVariant(cfg.model.name)

    if training_mode == TrainingMode.CLASSIFICATION:
        logger.info("Using CrossEntropyLoss for Classification")
        return CrossEntropyLoss()
    
    elif training_mode == TrainingMode.CONTRASTIVE:
        logger.info(f"Using PairwiseCELoss for {space.value} space")
        # Ensure your PairwiseCELoss returns (loss, stats_dict)
        return PairwiseCELoss(space=space, curvature=cfg.space.curvature,thau=cfg.space.thau)

    elif training_mode == TrainingMode.SIMSIAM:
        logger.info("Using SimSiam Negative Cosine Similarity Loss")
        return HyperbolicSimSiamLoss(space=space, curvature=cfg.space.curvature,thau=cfg.space.thau)

class PairwiseCELoss(nn.Module): 
    """
    Supports Euclidean (Cosine) and Hyperbolic distances.
    """
    def __init__(self, space: Space, thau: float = 0.2, curvature: float = 0.1):
        super().__init__()
        self.space = space
        self.thau = thau
        self.c = curvature
        
        if self.space == Space.HYPERBOLIC:
            self.hyp_space = HyperbolicSpace(c=self.c)

    def _get_distance_matrix(self, x: torch.Tensor, y: torch.Tensor):
        if self.space == Space.HYPERBOLIC:
            # Hyperbolic distance is used as a dissimilarity measure
            # We negate it to use it in a similarity-based CrossEntropy
            return -self.hyp_space.distance(x, y)
        else:
            # Standard Cosine Similarity
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            return x_norm @ y_norm.t()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        bsize = x.shape[0]
        device = x.device
        
        target = torch.arange(bsize, device=device)
        # Mask out self-similarity (diagonal) with a large negative value
        eye_mask = torch.eye(bsize, device=device) * 1e9

        logits00 = self._get_distance_matrix(x, x) / self.thau - eye_mask
        logits01 = self._get_distance_matrix(x, y) / self.thau

        logits = torch.cat([logits01, logits00], dim=1)
        
        # Stability trick: subtract max logits
        logits = logits - logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        
        # stats = {
        #     "logits/min": logits01.min().item(),
        #     "logits/mean": logits01.mean().item(),
        #     "logits/max": logits01.max().item(),
        #     "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
        # }
        with torch.no_grad():
            # Full logits accuracy (matches the CrossEntropy objective)
            full_preds = logits.argmax(dim=1)
            acc = (full_preds == target).float().mean().item()
            
            # Positive pairs vs Negative pairs separation
            # logits01 diagonal= the positive pairs
            pos_sim = torch.diag(logits01) 
            
            stats = {
                "logits/min": logits01.min().item(),
                "logits/max": logits01.max().item(),
                "logits/acc": acc,
                "sim/pos_mean": pos_sim.mean().item(), # if value too high and logits/acc low -> COLLAPSE
                "sim/diff": (pos_sim.mean() - logits01.mean()).item(), # Contrastive Gap. Low -> model fails to distinguish the augmented view
            }
        
        return loss, stats



class HyperbolicSimSiamLoss(nn.Module):
    """
    Implementation of SimSiam loss adapted for different geometries.
    Note: For Euclidean, it typically uses negative cosine similarity.
    """
    def __init__(self, space: Space, curvature: float = 0.1, thau: float = 0.1):
        super().__init__()
        self.space = space
        self.c = curvature
        self.thau = thau
        
        if self.space == Space.HYPERBOLIC:
            self.hyp_space = HyperbolicSpace(c=self.c)

    def d_func(self, x: torch.Tensor, y: torch.Tensor):
        if self.space == Space.HYPERBOLIC:
            return self.hyp_space.distance(x, y) / self.thau
        else:
            # Standard SimSiam uses negative cosine similarity: -cos(p, z)
            # This is equivalent to minimizing the angle between them
            p_norm = F.normalize(x, dim=-1)
            z_norm = F.normalize(y, dim=-1)
            return -(p_norm * z_norm).sum(dim=-1)
    
    def forward(self, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Symmetrized loss
        loss = (self.d_func(p1, z2.detach()).mean() + self.d_func(p2, z1.detach()).mean()) * 0.5
        stats = {
            "p1/min": p1.min().item(),
            "p1/max": p1.max().item(),
            "p2/min": p2.min().item(),
            "p2/max": p2.max().item(),
        }

        return loss, stats

class CrossEntropyLoss(nn.Module):
    """Standard CE for supervised training with integrated stats logging."""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = self.ce_loss(outputs, targets)
        
        with torch.no_grad():
            preds = outputs.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            stats = {"loss": loss.item(), "accuracy": acc}

        return loss, stats