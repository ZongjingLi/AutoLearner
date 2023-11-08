import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryDiffuser(nn.Module):
    """
    An action diffusion model that is used to perform diffusion and sample valid trajectory
    for a specific action.
    """
    def __init__(self, action_dim, latent_dim, beta = 0.1):
        super().__init__()