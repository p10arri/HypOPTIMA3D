import torch.nn as nn
from src.model.vit3d import ViT3D
from src.model.projector import Projector
from src.utils.enums import Space

class GeometricModel(nn.Module):
    """
    A wrapper that connects the 3D Vision Transformer backbone 
    to the Projector head.
    """
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # x shape: [Batch, Channels, Depth, Height, Width]
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
def get_model(cfg, num_classes, num_frames, in_channels) -> nn.Module:
    # Initialize 3D Backbone
    backbone = ViT3D(
        img_size=cfg.data.image_size,
        in_chans=in_channels,
        num_classes=num_classes,
        num_frames=num_frames,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        transfomer_depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        qkv_bias=cfg.model.qkv_bias,
        drop_rate=cfg.model.drop_rate,
        attn_drop_rate=cfg.model.attn_drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
        dropout=cfg.model.dropout,
        skip_class_head=cfg.model.skip_class_head,
        pretrained=cfg.model.pretrained,
        checkpoint_path=cfg.model.checkpoint_path
    )

    # Initialize Geometric Head
    # Using backbone.out_dim if available, otherwise fallback to embed_dim
    input_dim = backbone.out_dim
    
    head = Projector(
        space= Space(cfg.space.name),
        embed_dim=input_dim,
        num_classes=num_classes,
        curvature=cfg.space.get("curvature", 0.0),
        clip_r=cfg.space.clip_radius
    )

    return GeometricModel(backbone, head)


class SupervisedViT3D(nn.Module):
    """
    A simple wrapper for fine-tuning that ensures the output 
    is a dictionary, matching the Projector's structure.
    """
    def __init__(self, backbone: ViT3D):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)
        return {"logits": logits}
    
def get_vit3d(cfg, num_classes, num_frames, in_channels) -> nn.Module:
    model = ViT3D(
        img_size=cfg.data.image_size,
        in_chans=in_channels,
        num_classes=num_classes,
        num_frames=num_frames,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        transfomer_depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        qkv_bias=cfg.model.qkv_bias,
        drop_rate=cfg.model.drop_rate,
        attn_drop_rate=cfg.model.attn_drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
        dropout=cfg.model.dropout,
        skip_class_head=cfg.model.skip_class_head,
        pretrained=cfg.model.pretrained,
        checkpoint_path=cfg.model.checkpoint_path
    )

    model.reset_classifier(num_classes=num_classes)
    return SupervisedViT3D(model)