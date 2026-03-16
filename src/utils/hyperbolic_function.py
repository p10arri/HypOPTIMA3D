import torch
import geoopt
import math
import numpy as np

# fix seed
seed = 42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

class HyperbolicSpace:
    def __init__(self, c=0.1, dim=None):
        self.c = c
        self.ball = geoopt.PoincareBall(c=c)
        self.dim = dim

    def exp_map(self, x):
        """
        Exponential map at origin
        x: Euclidean vector (torch.Tensor)
        returns: point on Poincare ball
        """
        return self.ball.expmap0(x)

    def log_map(self, y):
        """
        Log map at origin
        y: point on Poincare ball
        returns: Euclidean vector in tangent space
        """
        return self.ball.logmap0(y)

    def mobius_add(self, x, y):
        """
        Möbius addition on Poincare ball
        """
        return self.ball.mobius_add(x, y)

    def distance(self, x, y):
        """
        Poincare distance
        Dhyp(x,y) = (2 / sqrt(c)) * artanh(sqrt(c) * ||-x ⊕ y||)
        Supports:
            - x, y: [dim] -> scalar
            - x, y: [batch, dim] -> [batch] elementwise
            - x: [batch_x, dim], y: [batch_y, dim] -> [batch_x, batch_y] pairwise
        """
        # GEOOPT ALREADY SATISFIES STABILITY
        # Euclidean case
        # if self.c <=1e-5:
        #     # Euclidean fallback
        #     x_exp = x.unsqueeze(1)  # [batch_x,1,dim]
        #     y_exp = y.unsqueeze(0)  # [1,batch_y,dim]
        #     return 2 * (x_exp - y_exp).norm(dim=-1)  # [batch_x, batch_y]

        if x.shape[-1] != y.shape[-1]:
            raise ValueError(f"Embedding dimension mismatch: {x.shape[-1]} vs {y.shape[-1]}")
    
        # Batched -> produce pairwise distances:
        if x.dim() == 2 and y.dim() == 2:
            x_exp = x.unsqueeze(1)  # [Bx,1,D]
            y_exp = y.unsqueeze(0)  # [1,By,D]
            return self.ball.dist(x_exp, y_exp)

        # If strange shapes geoopt broadcasting
        return self.ball.dist(x,y)
        
    def project(self, x):
        """
        Ensure point stays within valid Poincare ball radius
        """
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-5) / math.sqrt(self.c)
        factor = torch.clamp(max_norm / norm, max=1.0)
        return x * factor

    def project_hyperboloid_to_poincare(self, embs_2d: np.ndarray) -> np.ndarray:
        """
        Maps points from the Hyperboloid (Lorentz) model to the Poincare Ball.
        The Hyperboloid coordinates (x_1, x_2, ..., x_n, t) satisfy:
            sum(x_i^2) - t^2 = -1/c
        """
        c = self.c
        x, y = embs_2d[:, 0], embs_2d[:, 1]
        
        # Calculate the height on the hyperboloid z=sqrt(1/c + x^2 + y^2)
        z = np.sqrt(1/c + x**2 + y**2)
        
        # Projection formula: point = x_spatial / (1/sqrt(c) * z)
        # This maps the hyperboloid onto the unit ball (or ball of radius 1/sqrt(c))
        denom = (1 / np.sqrt(c)) + z
        disk_x = x / denom
        disk_y = y / denom
        
        projected = np.stack([disk_x, disk_y], axis=1)

        # Final safety check: Ensure numerical precision hasn't pushed points outside the boundary
        projected_tensor = torch.from_numpy(projected).float()
        return self.project(projected_tensor).numpy()