"""TimeSformer adapted for 3D OCT Volumetric Classification:
    https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from einops import rearrange
from loguru import logger
import timm
from timm.layers import DropPath
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
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn_drop = nn.Dropout(attn_drop)

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
    """Duvuded Space-Time(Depth) Attention fixed."""
    # TODO: Add a "cilindrical" Space-Time attention (Full Space + a local 9 patch square at Depths intervals)    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Depth Attention (Inter-slice)
        self.depth_norm1 = norm_layer(dim)
        self.depth_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.depth_fc = nn.Linear(dim, dim)

        # Spatial Attention (Intra-slice)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        

    def forward(self, x, B, D, W):
        num_spatial_tokens = (x.size(1) - 1) // D
        H = num_spatial_tokens // W

        # Depth
        xd = x[:, 1:, :] # Body tokens
        xd = rearrange(xd, 'b (h w d) m -> (b h w) d m', b=B, h=H, w=W, d=D)
        res_depth = self.drop_path(self.depth_attn(self.depth_norm1(xd)))
        res_depth = rearrange(res_depth, '(b h w) d m -> b (h w d) m', b=B, h=H, w=W, d=D)
        res_depth = self.depth_fc(res_depth)
        xd = x[:, 1:, :] + res_depth

        # Spatial 
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, D, 1)
        cls_token = rearrange(cls_token, 'b d m -> (b d) m', b=B, d=D).unsqueeze(1)
        xs = xd
        xs = rearrange(xs, 'b (h w d) m -> (b d) (h w) m', b=B, h=H, w=W, d=D)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        # Averaging CLS token across the depth dimension
        cls_token = rearrange(res_spatial[:, 0, :], '(b d) m -> b d m', b=B, d=D)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
        res_spatial = rearrange(res_spatial[:, 1:, :], '(b d) (h w) m -> b (h w d) m', b=B, h=H, w=W, d=D)
        res = res_spatial
        x = xd

        # Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, D, W

# -------------------------------------------------------------------------

class ViT3D(nn.Module):    
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dim=768, transfomer_depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=49,dropout=0., 
                 skip_class_head=False, pretrained=False, checkpoint_path=None, use_grad_checkpoint=True):

        super().__init__()

        self.embed_dim = embed_dim
        self.transfomer_depth = transfomer_depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.depth_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.depth_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = torch.linspace(0,drop_path_rate, self.transfomer_depth).tolist() # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.transfomer_depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier Head
        self.skip_class_head = skip_class_head
        if not skip_class_head:
            self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        self.use_grad_checkpoint = use_grad_checkpoint
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)
        # Initialization of depth attention weights. Spatial only model at start
        for i, blk in enumerate(self.blocks):
            if i > 0:
                nn.init.constant_(blk.depth_fc.weight, 0)
                nn.init.constant_(blk.depth_fc.bias, 0)

        if pretrained:
            self.load_pretrained(checkpoint_path)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def load_pretrained(self, checkpoint_path:str =None):
        """
        If checkpoint_path is None: Fetches 2D ImageNet weights and adapts them for 3D 
        (RGB -> Gray conversion, skipping depth components).
        If checkpoint_path is provided: Loads a full ViT3D checkpoint (useful for 
        inference or resuming training).
        """
        own_state = self.state_dict()
        loaded_keys = []

        if checkpoint_path is None:
            logger.info("No checkpoint provided. Fetching 2D ImageNet weights (timm vit_base_patch16_224)...")
            pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            state_dict = pretrained_model.state_dict()
            is_3d_checkpoint = False
        else:
            logger.info(f"Loading ViT3D weights from local checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            is_3d_checkpoint = True

        for name, param in state_dict.items():
            if name not in own_state:
                logger.debug(f"Skipping {name}: Not present in ViT3D.")
                continue

            # 2D Adaptation Logic
            if not is_3d_checkpoint:
                # Convert RGB Weights to Grayscale for the first convolution
                if name == 'patch_embed.proj.weight' and param.shape[1] == 3:
                    logger.info(f"Adapting {name}: Converting 3-channel RGB to 1-channel Grayscale.")
                    param = param.mean(dim=1, keepdim=True)
                
                # Skip the Head (classes mismatch) and Temporal/Depth components
                if 'head' in name or 'time_embed' in name or 'temporal' in name:
                    logger.debug(f"Skipping 2D component: {name}")
                    continue

            # Check for shape mismatches
            if param.shape != own_state[name].shape:
                logger.warning(f"Shape mismatch for {name}: {param.shape} vs {own_state[name].shape}. Skipping.")
                continue

            # Successful match
            own_state[name].copy_(param)
            loaded_keys.append(name)

        self.load_state_dict(own_state)
        
        source = "timm (2D)" if not is_3d_checkpoint else "local (3D)"
        logger.info(f"Successfully loaded {len(loaded_keys)} parameters from {source} source.")
    
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # Input x: [B, 1, D, H, W]
        B = x.shape[0]
        x, D, W = self.patch_embed(x)

        # expand cls token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Spatial Positional Embedding
        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Depth Embedding
        cls_tokens = x[:B, 0, :].unsqueeze(1)

        x = x[:, 1:] # remove cls to process depth
        x = rearrange(x, '(b d) n m -> (b n) d m', b=B, d=D)

        # Depth Positional Embedding
        # Resize depth embeddings in case they don't match
        if D != self.depth_embed.size(1):
            depth_embed = self.depth_embed.transpose(1, 2)
            new_depth_embed = F.interpolate(depth_embed, size=(D), mode='nearest')
            new_depth_embed = new_depth_embed.transpose(1, 2)
            x = x + new_depth_embed
        else:
            x = x + self.depth_embed

        x = self.depth_drop(x)

        # Reconstruct sequence
        x = rearrange(x, '(b n) d m -> b (n d) m',b=B,d=D)
        x = torch.cat((cls_tokens, x), dim=1)

        # Attention blocks
        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = checkpoint(blk, x, B, D, W, use_reentrant=False)
            else:
                x = blk(x, B, D, W)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if not self.skip_class_head:
            x = self.head(x)
        return x
    
    @property
    def out_dim(self) -> int:
        """Returns the dimension of the tensor produced by forward()."""
        return self.num_classes if not self.skip_class_head else self.embed_dim


if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    # Dummy hyperparameters for the test
    batch_size = 2
    num_frames = 49  #  OCT slices (Depth)
    img_size = 224
    num_classes = 2  #  Healthy vs. Diseased
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running test on device: {device}")

    model = ViT3D(
        img_size=img_size,
        patch_size=16,
        in_chans=1, 
        num_classes=num_classes,
        transfomer_depth=4, 
        num_frames=num_frames,
        pretrained=False # Set to False for local dummy testing
    ).to(device)

    # Create Dummy Data
    # Shape: [B, C, D, H, W] -> [2, 1, 49, 224, 224]
    dummy_input = torch.randn(batch_size, 1, num_frames, img_size, img_size).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #  TRAINING CHECK 
    model.train()
    logger.info("Starting forward pass (Training Mode)...")
    
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    
    logger.info(f"Forward pass successful. Loss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("Backward pass and optimizer step successful.")

    # EVALUATION CHECK 
    model.eval()
    with torch.no_grad():
        logger.info("Starting inference (Eval Mode)...")
        eval_output = model(dummy_input)
        
        # Check shapes
        assert eval_output.shape == (batch_size, num_classes), \
            f"Output shape mismatch: expected {(batch_size, num_classes)}, got {eval_output.shape}"
        
        preds = torch.argmax(eval_output, dim=1)
        logger.info(f"Inference successful. Predictions: {preds.cpu().numpy()}")

    logger.success("ViT3D Sanity Check Passed!")