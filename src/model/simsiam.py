import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.model.tools import freeze_patch_embed, remove_original_heads, get_projection_layer
from src.utils.enums import Space

# TODO: load simsiam encoder in factory.py
# def load_simsiam_vanilla_encoder(
#     ckpt_path: str,
#     encoder_version: str,
#     device="cuda",
# ):
#     """
#     Loads the Simsiam Vanilla Encoder for linear probing or evaluation.
#     Only the backbone weights are kept.
#     """
#     # Recreate the backbone 
#     encoder = timm.create_model(
#         encoder_version,
#         pretrained=False,
#         cache_dir="./cached_models/",
#     )

#     # Strip heads and freeze patch embeddings
#     remove_original_heads(encoder)
#     freeze_patch_embed(encoder)

#     # Load weights
#     # Note: Check if checkpoint contains 'model_state_dict' or 'state_dict'
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     state_dict = checkpoint.get('model_state_dict', checkpoint)
    
#     # Filter state_dict to only include encoder weights if necessary
#     encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
#     if not encoder_state_dict:
#         encoder_state_dict = state_dict

#     encoder.load_state_dict(encoder_state_dict, strict=True)
#     encoder.to(device)
#     encoder.eval()

#     return encoder


class SimSiamVanilla(nn.Module):
    """
    SimSiam architecture based on 'Exploring Simple Siamese Representation Learning'.
    Features a 3-layer Projector and a 2-layer Predictor (Bottleneck).
    """
    def __init__(
        self, 
        encoder_version: str, 
        space: Space = None,
        dim: int = 128,
        pred_dim: int = 64, 
        hidden_dim: int = 2048,
        hyp_c: float = 1.0, 
        clip_radius: float = 2.3, 
        freeze: bool = True,
        pretrained: bool = True, 
        device: torch.device | str = 'cuda',
        encoder_variant: str = None,
    ):
    
        super(SimSiamVanilla, self).__init__()
        
        self.encoder_version = encoder_version
        self.space = space
        self.dim = dim
        self.pred_dim = pred_dim
        self.hidden_dim = hidden_dim
        self.hyp_c = hyp_c
        self.clip_radius = clip_radius
        self.device = device
        self.encoder_variant = encoder_variant


        # Backbone initialization
        backbone = timm.create_model(encoder_version, pretrained=True, cache_dir="./cached_models/")
        remove_original_heads(backbone)
        freeze_patch_embed(backbone)
        
        if hasattr(backbone, 'num_features'):
            prev_dim = backbone.num_features
        elif hasattr(backbone, 'embed_dim'):
            prev_dim = backbone.embed_dim
        else:
            prev_dim = 384

        self.encoder = backbone 

        # Projector arquitechture: Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear -> BN (no affine)
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False) 
        )
        
        # Projector arquitechture (Bottleneck): Linear -> BN -> ReLU -> Linear
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        """
        Forward pass for Siamese training.
        Returns predictions (p) and detached projections (z).
        """
        f1, f2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)

        # IMPORTANT PREVENT COLLAPSE: Detach z to stop gradients
        return p1, p2, z1.detach(), z2.detach()


class SimSiamRiemannProjector(SimSiamVanilla):
    """
    Geometric extension of SimSiam.
    Applies Spherical or Hyperbolic projections to both the predictions and targets.
    """
    def __init__(self, 
                 space: Space,
                 encoder_version, 
                 hyp_c=0.0, 
                 clip_radius=2.3,
                 dim=128, 
                 pred_dim=64, 
                 hidden_dim=2048):
        
        super(SimSiamRiemannProjector, self).__init__(encoder_version, dim, pred_dim, hidden_dim)

        self.riemann_head = get_projection_layer(space, hyp_c=hyp_c, clip_r=clip_radius)
    
    def forward(self, x1, x2):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Geometric projection
        p1_riem = self.riemann_head(p1)
        p2_riem = self.riemann_head(p2)
        
        # target representation in the manifold
        z1_riem = self.riemann_head(z1)
        z2_riem = self.riemann_head(z2)

        return p1_riem, p2_riem, z1_riem.detach(), z2_riem.detach() # stop-gradient on targets