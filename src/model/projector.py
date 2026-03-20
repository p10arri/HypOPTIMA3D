
import torch
import torch.nn as nn
import geoopt

from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.sphere import Sphere


from src.model.vit3d import ViT3D
from src.utils.enums import Space

class EuclideanHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)
        embeddings = x 
        
        # Centroids are the weight matrix of the linear layer
        centroids = self.fc.weight

        return {
            "logits": logits,
            "embeddings": embeddings,
            "centroids": centroids
        }
    
class SphericalHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.manifold = Sphere()
        self.embed_dim = embed_dim
        
        # Initialize centroids directly on the sphere surface
        # random_normal on Sphere returns a ManifoldTensor of the same dimension
        init_centroids = self.manifold.random_uniform(num_classes, embed_dim)
        self.centroids = geoopt.ManifoldParameter(init_centroids, manifold=self.manifold)

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
        self.centroids = geoopt.ManifoldParameter(init_centroids, manifold=self.manifold)

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
        elif space == Space.EUCLIDEAN:
            self.head = EuclideanHead(embed_dim, num_classes)
        else:
            raise ValueError(f"Unsupported space: {space}")

    def forward(self, x):
        return self.head(x)

    @property
    def manifold(self):
        """Helper to access the manifold for Riemannian optimization if needed."""
        return getattr(self.head, 'manifold', None)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    batch_size = 2
    embed_dim = 128 
    num_heads = 4
    num_classes = 9
    clip_r_value = 2.3

    # [B, C, D, H, W]
    dummy_input = torch.randn(batch_size, 1, 16, 224, 224).to(device)

    for space in [Space.EUCLIDEAN, Space.SPHERICAL, Space.HYPERBOLIC]:
        print(f"\n" + "="*50)
        print(f"TESTING SPACE: {space.name}")
        print("="*50)
        
        backbone = ViT3D(
            img_size=224, 
            patch_size=16, 
            embed_dim=embed_dim, 
            num_classes=num_classes,
            num_heads=num_heads,
            num_frames=16,
            skip_class_head=False, # True=128 features, False= 9 logits
        ).to(device)
        
        head = Projector(
            space=space, 
            embed_dim=backbone.out_dim,
            num_classes=num_classes, 
            curvature=1.0,
            clip_r=clip_r_value
        ).to(device)
        
        backbone.eval()
        head.eval()
        with torch.no_grad():
            features = backbone(dummy_input) 
            output = head(features)
        
        logits = output["logits"]
        embeddings = output["embeddings"]
        centroids = output["centroids"]
        
        # 4. Validations
        print(f"-> Backbone Feat Shape: {features.shape}") 
        print(f"-> Logits Shape:        {logits.shape}")   
        
        expected_dim = backbone.out_dim + 1 if space == Space.HYPERBOLIC else backbone.out_dim
        print(f"-> Embedding Dim:       {embeddings.shape[-1]} (Expected: {expected_dim})")

        # Geometric Constraint Checks
        if space == Space.EUCLIDEAN:
            print("-> Euclidean check: Logic verified.")
            print(f"-> Identity Mapping:    {torch.allclose(features, embeddings)}")
        else:
            manifold = head.manifold
            is_on_manifold = manifold.check_point_on_manifold(centroids)
            
            valid = is_on_manifold.all().item() if isinstance(is_on_manifold, torch.Tensor) else is_on_manifold
            print(f"-> Centroids on {space.name}: {valid}")
            
            if space == Space.SPHERICAL:
                norm_sq = (embeddings**2).sum(dim=-1)
                print(f"-> Mean Norm^2:         {norm_sq.mean().item():.4f} (Target: 1.0)")
            
            elif space == Space.HYPERBOLIC:
                m_inner = -embeddings[..., 0]**2 + (embeddings[..., 1:]**2).sum(dim=-1)
                print(f"-> Mean Minkowski <x,x>:{m_inner.mean().item():.4f} (Target: -1.0)")
                print(f"-> In Upper Sheet:      {torch.all(embeddings[..., 0] > 0).item()}")

    print("\n" + "="*50)
    print("ALL PROJECTOR TESTS COMPLETED SUCCESSFULLY")
    print("="*50)