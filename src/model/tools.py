import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.enums import Space
from src.model.hyperbolic import ToPoincare

def freeze_patch_embed(model: nn.Module):
    
    '''Freeze the linear projection for patch embeddings. 
    Code extracted from https://github.com/htdt/hyp_metric/blob/master/model.py

    Args:
        model (nn.Module): Model to freeze layer
        num_block (int): The number of transformer blocks (from the start of model.blocks) 
            to freeze. The patch embedding and positional dropout layers are always frozen.
    '''
    def _freeze_module(m):
        """Set requires_grad=False for all parameters of mod (no-op if mod is None)."""
        for param in m.parameters():
            param.requires_grad = False

        # Freeze patch embedding and positional dropout if they exist
        _freeze_module(model.patch_embed)
        _freeze_module(model.pos_drop)

        # Authors dont freeze the whole network. They freeze only the patch_embeddings or nothing
        # num_block = None or 0
        # for i in range(num_block):
        #     _freeze_module(model.blocks[i])

def get_projection_layer(space, hyp_c=0, clip_r=0, emb_dim=None) -> nn.Module:
        """Determines the projection layer based on the geometric space."""
        if space == Space.HYPERBOLIC:
            return ToPoincare(c=hyp_c, clip_r=clip_r)
        
        elif space == Space.SPHERICAL:
            return nn.Sequential(nn.BatchNorm1d(emb_dim), NormLayer())
        elif space == Space.EUCLIDEAN:
            return nn.Identity()
        else: 
            raise ValueError(
                f"Invalid space type: {space} (Type: {type(space)}). "
                f"Expected a member of src.utils.enums.Space."
                )

def remove_original_heads(backbone: nn.Module):
    """Replaces classifier heads with Identity for feature extraction."""
    for attr in ["head", "head_dist", "fc"]:
        if hasattr(backbone, attr):
            setattr(backbone, attr, nn.Identity())

def init_weights(layer: nn.Linear):
    """No bias and orthogonal weight initialization."""
    nn.init.constant_(layer.bias.data, 0)
    nn.init.orthogonal_(layer.weight.data)


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)
    
# class ModelConstructor(nn.Module):
#     def __init__(self, backbone, head):
#         '''  Wrapper class for combining a transformer-based backbone and a head module.
#             Integrates a normalization layer into the forward pass.
#         '''
#         super(ModelConstructor, self).__init__()
#         self.backbone = backbone
#         self.head = head
#         self.norm = NormLayer()

#     def forward(self, x, skip_head=False):
#         x = self.backbone(x)
#         if type(x) == tuple:
#             x = x[0]
            
#         if not skip_head:
#             x = self.head(x)
#         else:
#             x = self.norm(x)
#         return x

class ModelConstructor(nn.Module):
    def __init__(self, backbone, head, space: Space = Space.EUCLIDEAN):
        super(ModelConstructor, self).__init__()
        self.backbone = backbone
        self.head = head
        self.space = space
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.backbone(x)
        # ViT/DINO/timm outputs can be tuples
        if isinstance(x, (tuple, list)):
            x = x[0]
        if skip_head:
            if self.space == Space.EUCLIDEAN:
                return x #  raw featuresv 
            # Only normalize if we are in Spherical/Hyperbolic
            return self.norm(x)         
        return self.head(x)

    # def forward(self, x, skip_head=False):
    #     x = self.backbone(x)
    #     if isinstance(x, (tuple, list)): x = x[0]
        
    #     # If not skipping head, get the linear output
    #     if not skip_head:
    #         x = self.head(x) 
            
    #     # FORCE PROJECTION HERE based on space
    #     if self.space == Space.SPHERICAL:
    #         return torch.nn.functional.normalize(x, p=2, dim=-1)
    #     elif self.space == Space.HYPERBOLIC:
    #         # Assuming ToPoincare is accessible or use a hard clip
    #         norm = x.norm(dim=-1, keepdim=True)
    #         return x * (0.99 / torch.clamp(norm, min=0.99))
            
    #     return x