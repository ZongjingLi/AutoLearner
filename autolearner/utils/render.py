import torch
import torch.nn as nn
import numpy as np

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode = "center"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    i, j = torch.meshgrid(\
        torch.linspace(0,W-1, W, device = device),
        torch.linspace(0,H-1, H, device = device))
    i = i.t().float()
    j = j.t().float()

    if mode == "lefttop":
        pass
    elif mode == "center":
        i, j = i + 0.5, j + 0.5
    else:
        raise NotImplementedError
    if flip_x: i = i.flip((1,))
    if flip_y: j = j.flip((0,))

    if inverse_y:
        dirs = torch.stack([
            (i - K[0][2])/K[0][0],
            (j - K[1][2])/K[1][1],
            torch.ones_like(i)
        ], dim = -1)
    else:
        dirs = torch.stack([
            (i - K[0][2])/K[0][0],
            -(j - K[1][2])/K[1][1],
            -torch.ones_like(i)
        ], dim = -1)
    print("dirs", dirs.shape)
    # Rotate ray direction from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], dim = -1)
    print("rays_d",rays_d.shape)
    # Translate the camerta's frame to the world frame.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    print("rays_o",rays_o.shape)
    return rays_o, rays_d
