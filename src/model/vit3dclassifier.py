"""TimeSformer adapted for 3D OCT Volumetric Classification:
    https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py"""


import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import rearrange

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Depth Attention (Inter-slice)
        self.depth_norm = norm_layer(dim)
        self.depth_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)
        self.depth_fc = nn.Linear(dim, dim)

        # Spatial Attention (Intra-slice)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, B, D, W, H):
        # Depth Attention
        xt = x[:, 1:, :] # Body tokens
        xt = rearrange(xt, 'b (h w d) m -> (b h w) d m', b=B, h=H, w=W, d=D)
        res_depth = self.depth_attn(self.depth_norm(xt))
        res_depth = rearrange(res_depth, '(b h w) d m -> b (h w d) m', b=B, h=H, w=W, d=D)
        res_depth = self.depth_fc(res_depth)
        x_body = x[:, 1:, :] + res_depth

        # Spatial Attention
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, D, 1)
        cls_token = rearrange(cls_token, 'b d m -> (b d) m', b=B, d=D).unsqueeze(1)
        
        xs = rearrange(x_body, 'b (h w d) m -> (b d) (h w) m', b=B, h=H, w=W, d=D)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.attn(self.norm1(xs))

        # Averaging CLS token across the depth dimension
        cls_token = rearrange(res_spatial[:, 0, :], '(b d) m -> b d m', b=B, d=D).mean(1, keepdim=True)
        res_spatial = rearrange(res_spatial[:, 1:, :], '(b d) (h w) m -> b (h w d) m', b=B, h=H, w=W, d=D)
        
        # Mlp
        x = torch.cat((init_cls_token, x_body), 1) + torch.cat((cls_token, res_spatial), 1)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = self.proj(x)
        Ho, Wo = x.size(-2), x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, D, Wo, Ho

# -------------------------------------------------------------------------

class ViT3DClassifier(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=9, 
                 embed_dim=768, depth=12, num_heads=12, num_frames=49, drop_rate=0.):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_frames = num_frames # 'D' in forward
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.depth_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, drop=drop_rate)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier Head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.depth_embed, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)

        # Initialization of depth attention weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # Input x: [B, 1, D, H, W]
        B = x.shape[0]
        x, D, W, H = self.patch_embed(x)
        
        # Spatial Embedding
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Depth Embedding
        cls_token = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]
        x = rearrange(x, '(b d) n m -> (b n) d m', b=B, d=D)
        
        # Linear interpolation if D != defined num_frames
        ## resizing the positional embeddings in case they don't match the input at inference
        if D != self.depth_embed.size(1):
            d_embed = self.depth_embed.transpose(1, 2)
            d_embed = F.interpolate(d_embed, size=D, mode='linear', align_corners=False)
            x = x + d_embed.transpose(1, 2)
        else:
            x = x + self.depth_embed
            
        x = rearrange(x, '(b n) d m -> b (n d) m', b=B, d=D)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)

        # Transformer Blocks (Divided Space-Depth Attention)
        for blk in self.blocks:
            x = blk(x, B, D, W, H)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sanity check on: {device}")

    # Initialize model with your OCT specs
    # embed_dim=768 is the standard 'Base' size
    model = ViT3DClassifier(
        img_size=224, 
        patch_size=16, 
        in_chans=1, 
        num_classes=9, 
        num_frames=49
    ).to(device)

    # Create a dummy input similar to train sample
    dummy_input = torch.randn(2, 1, 49, 224, 224).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print("-" * 30)
        print("Success!")
        print(f"Input Shape:  {dummy_input.shape}")  # [2, 1, 49, 224, 224]
        print(f"Output Shape: {output.shape}")       # [2, 9]
        print("-" * 30)
        
    except Exception as e:
        print(f"Forward pass failed! Error: {e}")