
"""
Hyperbolic Geometry Modules for Deep Learning.
Includes Riemannian gradient scaling and Poincare ball projections.
Code extracted and modified from https://github.com/htdt/hyp_metric/blob/master/hyptorch/nn.py#L143"""


import torch
import torch.nn as nn
import geoopt
import geoopt.manifolds.stereographic.math as pmath
import torch
import geoopt
import math

import torch.nn.functional as F 

class RiemannianGradient(torch.autograd.Function):
    """
    Autograd function that forwards x unchanged and converts incoming Euclidean
    gradients to Riemannian gradients on the Poincare ball during backward.
    Backward returns gradient w.r.t. x
    """
    c = 1
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        
        scale = (1 - RiemannianGradient.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale
        
class ToPoincareV1(nn.Module):
    '''Maps points in n dimensional Euclidean space to n dimensional Poincaré ball with base in the origin.
    Also implements clipping from https://arxiv.org/pdf/2506.05826'''

    def __init__(self, c, clip_r=None):
        super(ToPoincare, self).__init__()
        #self.register_parameter("xp", None)

        self.c = c
        self.ball = geoopt.PoincareBall(c=c)

        self.riemannian = RiemannianGradient
        self.riemannian.c = c
        
        
        self.clip_r = clip_r
        self.grad_fix = lambda x: x # TODO: from authors -> no riemanian gradients applied

    def project(self, x):
        """
        Ensure point stays within valid Poincare ball radius.
        """
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-5) / math.sqrt(self.c)
        factor = torch.clamp(max_norm / norm, max=1.0)
        return x * factor

    def forward(self, x):
        # Euclidean norm
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-5)
        
        # Smoothly scale the norm using tanh so it never exceeds 1.0
        # This maps [0, inf) -> [0, 1)
        # We use self.clip_r to control how much of the ball we use
        rescale = torch.tanh(x_norm) * self.clip_r / x_norm
        x_scaled = x * rescale
        
        # TEST: Taha-> float64  stabilizes training, but increases per-epoch time by ~20%.
        x_scaled = x_scaled.double()

        # Map to Poincare Ball
        return self.grad_fix(self.project(self.ball.expmap0(x_scaled)))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class ToPoincare(nn.Module):
    '''Maps points in n dimensional Euclidean space to n dimensional Poincaré ball with base in the origin.
    Also implements clipping from https://arxiv.org/pdf/2506.05826'''

    def __init__(self, c, clip_r=None):
        super(ToPoincare, self).__init__()
        #self.register_parameter("xp", None)

        self.c = c
        self.ball = geoopt.PoincareBall(c=c)

        self.riemannian = RiemannianGradient
        self.riemannian.c = c
        
        
        self.clip_r = clip_r
        self.grad_fix = lambda x: x # TODO: from authors -> no riemanian gradients applied

    def project(self, x):
        """
        Ensure point stays within valid Poincare ball radius.
        """
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-5) / math.sqrt(self.c)
        factor = torch.clamp(max_norm / norm, max=1.0)
        return x * factor

    def forward(self, x):
        if self.clip_r is not None:
            # x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-5)
            # # Clip using rescale-and-threshold rule 
            # d = x.size(-1)
            # sqrt_d = x_norm.new_tensor(d).sqrt()
            # max_norm = self.clip_r * sqrt_d  # effective threshold
            # fac = torch.minimum(torch.ones_like(x_norm), max_norm / x_norm)
            # x = x * fac


        # Author's clipping
        # if self.clip_r is not None:
        #     x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
        #     fac =  torch.minimum(
        #         torch.ones_like(x_norm), 
        #         self.clip_r / x_norm
        #     )
        #     x = x * fac
        
            # Taha's prenormalization and clipping:
            # Normalize and clip embeddings
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), 1.0/x_norm)*x # Clipped
        
        return self.grad_fix(self.project(self.ball.expmap0(x)))
        # normalize after projection TEST PABLO
        # x_norm = F.normalize(self.project(self.ball.expmap0(x)), p=2, dim=-1)
        # return self.grad_fix(x_norm)

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)
    