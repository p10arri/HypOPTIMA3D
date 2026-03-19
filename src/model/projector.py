import torch.nn as nn
from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.sphere import Sphere

from src.model.vit3d import ViT3D
from src.utils.enums import Space

class SphericalHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.manifold = Sphere()
        self.embed_dim = embed_dim
        
        # Initialize centroids directly on the sphere surface
        # random_normal on Sphere returns a ManifoldTensor of the same dimension
        init_centroids = self.manifold.random_uniform(num_classes, embed_dim)
        self.centroids = nn.Parameter(init_centroids.data)

    def forward(self, x):
        # x is Euclidean
        
        # Map to Sphere
        x_sph = self.manifold.projx(x)
        
        # Calculate cosine-like distances on the sphere
        # Negative distance so that closer = higher logit
        logits = -self.manifold.dist(
            x_sph.unsqueeze(1), 
            self.centroids.unsqueeze(0)
        )

        return {
            "logits": logits,
            "embeddings": x_sph,
            "centroids": self.centroids
        }
                          
class HyperbolicHead(nn.Module):
    def __init__(self, embed_dim, num_classes, clip_r=2.3, k=1.0):
        super().__init__()
        self.manifold = Lorentz(k=k)
        self.embed_dim = embed_dim
        self.clip_r = clip_r

        # Centroid directly in manifold
        # NOTE: self.manifold.origin() and random_normal() DOES NOT handle the d+1 dimensionality internally.
        
        # manifold tensor
        init_centroids = self.manifold.random_normal(num_classes, embed_dim + 1)
        # convert to raw tensor
        self.centroids = nn.Parameter(init_centroids.data)

    def forward(self, x):
        # x is Euclidean 
        
        # Smoothly scale the norm using tanh so it never exceeds 1.0
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-5)
        rescale = torch.tanh(x_norm) * self.clip_r / x_norm
        x_scaled = x * rescale
        
        # # TEST: Taha-> float64  stabilizes training, but increases per-epoch time by ~20%.
        # x_scaled = x_scaled.double()

        # Add time dimension
        # move the vector to the curved manifold. Add +1 dim.
        # NOTE: Padding by 0 places the vector into the flat tangent plane (at the bottom of the hyperboloid bowl)
        x_tangent = torch.cat([torch.zeros_like(x_scaled[:, :1]), x_scaled], dim=-1)

        # Map to Lorentz Manifold
        x_hyp = self.manifold.expmap0(x_tangent)        
        # Project back to manifold to ensure <x,x> = -1
        x_hyp = self.manifold.projx(x_hyp)

        # Calculate negative distances (logits) for inference
        logits = -self.manifold.dist(
            x_hyp.unsqueeze(1), 
            self.centroids.unsqueeze(0), 
            dim=-1
        )

        return {
            "logits": logits,
            "embeddings": x_hyp,
            "centroids": self.centroids
        }


class Projector(nn.Module):
    def __init__(self, space: Space, embed_dim: int, num_classes: int, curvature: float = 1.0, clip_r: float = 1.0):
        super().__init__()
        self.space_type = space

        if space == Space.HYPERBOLIC:
            self.head = HyperbolicHead(embed_dim, num_classes, clip_r=clip_r, k=curvature)
        elif space == Space.SPHERICAL:
            self.head = SphericalHead(embed_dim, num_classes)
        else:
            raise ValueError(f"Unsupported space: {space} in ProjectorHead")

    def forward(self, x):
        return self.head(x)

    @property
    def manifold(self):
        """Helper to access the manifold for Riemannian optimization if needed."""
        return getattr(self.head, 'manifold', None)


if __name__ == "__main__":
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    batch_size = 2
    embed_dim = 128  # Must be divisable by number of heads
    num_heads = 4
    num_classes = 9
    clip_r_value = 2.3
    dummy_input = torch.randn(batch_size, 1, 16, 224, 224).to(device) # Smaller frames for speed

    # Iterate through different Geometric Spaces
    for space in [Space.SPHERICAL, Space.HYPERBOLIC]:
        print(f"\n--- Testing Space: {space.name} ---")
        
        # Initialize Backbone
        backbone = ViT3D(
            img_size=224, 
            patch_size=16, 
            embed_dim=embed_dim, 
            num_classes=num_classes,
            num_heads=num_heads,
            num_frames=16,
            skip_class_head=False,
        ).to(device)
        
        # Apply head removal tool
        # remove_original_heads(backbone) 
        # # For this test, we'll just manually replace it:
        # backbone.head = nn.Identity()
        
        # Initialize Projector Head
        head = Projector(
            space=space, 
            embed_dim=backbone.out_dim, 
            num_classes=num_classes, 
            curvature=1.0,
            clip_r=clip_r_value
        ).to(device)
        
        backbone.eval()
        with torch.no_grad():
            features = backbone(dummy_input) # Should be [2, 128]
            output = head(features)
        
        logits = output["logits"]
        embeddings = output["embeddings"]
        centroids = output["centroids"]
        
        # Shape Validations
        print(f"Backbone Feat Shape: {features.shape}") # [2, 128]
        print(f"Logits Shape:        {logits.shape}")   # [2, 9]
        
        if space == Space.HYPERBOLIC:
            # Lorentz requires d+1 dimensions
            expected_dim = embed_dim + 1
            print(f"Embedding Shape: {embeddings.shape} (Expected {expected_dim} for Lorentz)")
        else:
            expected_dim = embed_dim
            print(f"Embedding Shape:     {embeddings.shape}")
            
        # Manifold Constraint Check
        manifold = head.manifold
        with torch.no_grad():
            # Check if centroids are actually on the manifold
            is_on_manifold = manifold.check_point_on_manifold(centroids)
            if is_on_manifold:
                print(f"Centroids successfully placed on {space.name} manifold.")
            else:
                print(f"Manifold Error: Points are not on the {space.name} surface.")
                
            # Verify mathematical constraints
            if space == Space.SPHERICAL:
                # Norm^2 should be 1.0
                norm_sq = (embeddings**2).sum(dim=-1)
                print(f"Mean Norm^2: {norm_sq.mean().item():.4f} (Target: 1.0)")
            
            elif space == Space.HYPERBOLIC:
                # Manual Minkowski check for Lorentz (assuming k=1)
                # <x, x>_L = -x_0^2 + x_1^2 + ... + x_n^2
                m_inner = -embeddings[..., 0]**2 + (embeddings[..., 1:]**2).sum(dim=-1)
                print(f"Mean Minkowski <x,x>: {m_inner.mean().item():.4f} (Target: -1.0)")
                
                # Also check if x_0 is always positive (Lorentz points must be in the upper sheet)
                is_upper_sheet = torch.all(embeddings[..., 0] > 0)
                print(f"Points in upper sheet: {is_upper_sheet}")
    print("\n--- All Projector Tests Completed Successfully ---")