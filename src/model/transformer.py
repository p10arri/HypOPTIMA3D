from __future__ import annotations

import torch
import torch.nn as nn
import timm
from loguru import logger

from src.model.tools import ModelConstructor, freeze_patch_embed, get_projection_layer, remove_original_heads, init_weights
from src.utils.enums import Space

class Transformer(nn.Module):
    def __init__(
        self, 
        encoder_version: str, 
        space: Space = None,
        emb_dim: int = 128, 
        hyp_c: float = 1.0, 
        clip_radius: float = 2.3, 
        freeze: bool = True,
        pretrained: bool = True, 
        encoder_variant: str = None,
    ):
        super(Transformer, self).__init__()

        self.encoder_variant = encoder_variant
        self.space = space
        self.emb_dim = emb_dim
        self.encoder_version = encoder_version
        self.hyp_c = hyp_c
        self.clip_radius = clip_radius

        backbone = timm.create_model(encoder_version, pretrained=pretrained)
        
        # use encoder as a feature extractor
        remove_original_heads(backbone)
        
        if freeze:
            freeze_patch_embed(backbone)

        projection = get_projection_layer(space, hyp_c, clip_radius, emb_dim)
        
        backbone_dim = backbone.num_features
        head = nn.Sequential(
            nn.Linear(backbone_dim, emb_dim), # TODO: random linear layer (not trained) follow HypViT paper. IN THE PAPER THEY DO TRAIN IT!! https://github.com/htdt/hyp_metric/blob/master/model.py
            projection
        )

        init_weights(head[0])
        self.model = ModelConstructor(backbone, head)

    def forward(self, x: torch.Tensor, skip_head: bool = False) -> torch.Tensor:
        return self.model(x, skip_head=skip_head)