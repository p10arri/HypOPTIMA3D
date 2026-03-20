import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from loguru import logger

from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.sphere import Sphere

from src.utils.enums import TrainingMode, Space

def get_loss(cfg, training_mode: TrainingMode) -> nn.Module:
    space = Space(cfg.space.name)

    thau = getattr(cfg.space, "thau", 0.1)

    if training_mode == TrainingMode.SUPERVISED:
        # Cross entropy loss is invariant of the manifold
        logger.info("Using CrossEntropyLoss for Supervised Classification")
        return CrossEntropyLoss()
    
    elif training_mode == TrainingMode.CONTRASTIVE:
        logger.info(f"Using PairwiseCELoss for {space.value} space")
        return PairwiseCELoss(space=space, curvature=cfg.space.curvature, thau=thau)

    elif training_mode == TrainingMode.SIMSIAM:
        logger.info(f"Using SimSiam Negative Similarity Loss for {space.value}")
        return ProjectorSimSiamLoss(space=space, curvature=cfg.space.curvature, thau=thau)

class PairwiseCELoss(nn.Module): 
    """
    Contrastive Loss geometry aware.
    """
    def __init__(self, space: Space, thau: float = 0.1, curvature: float = 1.0):
        super().__init__()
        self.space = space
        self.thau = thau
        
        if self.space == Space.HYPERBOLIC:
            self.manifold = Lorentz(k=curvature)
        elif self.space == Space.SPHERICAL:
            self.manifold = Sphere()
        else:
            self.manifold = None

    def _get_similarity_matrix(self, x: torch.Tensor, y: torch.Tensor):
        if self.space == Space.HYPERBOLIC or self.space == Space.SPHERICAL:
            # For non-Euclidean, negative distance as a proxy for similarity
            # unsqueeze(1) and unsqueeze(0) creates the pairwise distance matrix [B, B]
            dist = self.manifold.dist(x.unsqueeze(1), y.unsqueeze(0))
            return -dist 
        else:
            # Standard Cosine Similarity for Euclidean
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            return torch.matmul(x_norm, y_norm.t())
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        bsize = x.shape[0]
        device = x.device
        
        target = torch.arange(bsize, device=device)
        # Mask self-similarity on the diagonal
        eye_mask = torch.eye(bsize, device=device) * 1e9

        sim_xx = self._get_similarity_matrix(x, x) / self.thau - eye_mask
        sim_xy = self._get_similarity_matrix(x, y) / self.thau

        # Standard InfoNCE / NT-Xent structure
        logits = torch.cat([sim_xy, sim_xx], dim=1) 
        
        # Stability trick: subtract max logits
        logits = logits - logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target) # cross entropy is manifold invariant
        
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == target).float().mean().item()

            # Positive pairs vs Negative pairs separation
            # logits01 diagonal= the positive pairs
            pos_sim = torch.diag(sim_xy) 
            
            stats = {
                "logits/acc": acc,
                "sim/pos_mean": pos_sim.mean().item(),  # if value too high and logits/acc low -> COLLAPSE
                "sim/diff": (pos_sim.mean() - sim_xy.mean()).item(),# Contrastive Gap. Low -> model fails to distinguish the augmented view
            }
        
        return loss, stats

class ProjectorSimSiamLoss(nn.Module):
    """
    SimSiam loss adapted to project embeddings onto specific manifolds.
    """
    def __init__(self, space: Space, curvature: float = 1.0, thau: float = 0.1):
        super().__init__()
        self.space = space
        self.thau = thau
        
        if self.space == Space.HYPERBOLIC:
            self.manifold = Lorentz(k=curvature)
        elif self.space == Space.SPHERICAL:
            self.manifold = Sphere()
        else:
            self.manifold = None

    def d_func(self, x: torch.Tensor, y: torch.Tensor):
        if self.manifold:
            # Minimize distance on the manifold
            return self.manifold.dist(x, y) / self.thau
        else:
            # Standard Negative Cosine Similarity
            p_norm = F.normalize(x, dim=-1)
            z_norm = F.normalize(y, dim=-1)
            return -(p_norm * z_norm).sum(dim=-1)
    
    def forward(self, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Symmetrized loss with stop-gradient (detach) on z
        loss = (self.d_func(p1, z2.detach()).mean() + self.d_func(p2, z1.detach()).mean()) * 0.5
        
        stats = {
            "dist/mean": loss.item() * self.thau,
            "p/norm_mean": torch.norm(p1, dim=-1).mean().item()
        }

        return loss, stats

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = self.ce_loss(outputs, targets)
        
        with torch.no_grad():
            preds = outputs.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            stats = {"eval/accuracy": acc, "eval/loss": loss.item()}

        return loss, stats