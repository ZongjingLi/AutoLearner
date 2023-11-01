import torch
import torch.nn as nn

import karanir
from karanir.dklearn import *
from karanir.dklearn.nn.cnn import ConvolutionUnits

from .propagation import *
from .competition import *
from .projection  import *

from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision


def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[64,64], antialias=False)
    img2_batch = F.resize(img2_batch, size=[64,64], antialias=False)
    return transforms(img1_batch, img2_batch)

def compute_optical_flow(img1_batch, img2_batch):
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    return list_of_flows

class SceneNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.imsize = config.imsize
        self.perception_size = config.perception_size

        # [Convolutional Backbone]
        self.convs_backbone = ConvolutionUnits(config.channels,config.conv_dim, config.latent_dim)

        # [Edge Link Prediction Module]
        kq_dim = config.conv_dim
        latent_dim = config.conv_dim
        norm_fn = "batch"
        kernel_size = 3
        downsample = False
        self.k_convs = nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size = kernel_size, bias = False, stride = 1, residual = True, downsample = downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size = 1, bias = True, padding = "same"),
            )

        self.q_convs = nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size = kernel_size, bias = False, stride = 1, residual = True, downsample = downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size = 1, bias = False, padding = "same"),
            )
        self.edges = build_perception(self.imsize, self.perception_size, device)

        # [Propagation Module]
        self.graph_propagation = GraphPropagation(num_iters = 32)

        # [Competition Module]
        self.competition = Competition(num_masks = config.num_segments)

        self.verbose = 0
        self.device = device

    def forward(self, x, masks = None):
        """
        perform object level segmentation using optical flow estimation.
        """
        B, W, H, C = x.shape

        # [Grid Convs]
        x = x.permute(0,3,1,2)
        conv_features = self.convs_backbone(x)

        # [Decode Ks:Qs]
        decode_ks = self.k_convs(conv_features).flatten(2,3).permute(0,2,1)
        decode_qs = self.q_convs(conv_features).flatten(2,3).permute(0,2,1)

        # [Edge Connection]
        edges = self.edges.to(self.device)

        scale = 1.0
        weights = torch.cosine_similarity(decode_qs[:,edges[0][:],:],decode_ks[:,edges[1][:],:], dim = -1)
        weight = torch.softmax(weights * scale, dim = -1)

        # [Graph Propagation]
        N = B * decode_ks.shape[1] # number of edges
        Q = 128 # graph prop inis dim 
        random_init_state = torch.randn([B,N,Q])
        adjs = SparseTensor(
            row = edges[0],
            col = edges[1],
            value = weights.flatten(),
            sparse_sizes = (B*N, B*N)
            )

        prop_features = self.graph_propagation(random_init_state, adjs)
        
        # [Competition]
        segment_map = prop_features[-1].reshape([B,W,H,Q])
        masks, agents, alive, pheno, unharv = self.competition(segment_map)
        masks_extracted = torch.cat([masks,], dim = -1).permute(0,3,1,2)

        # [Segment Loss]
        loss = 0.

        return {"masks":masks_extracted, "loss": loss}

    def store_parameters(self, path):
        return
    
    def load_parameters(self, path, map_location):
        self.load_state_dict(torch.load(path, map_location= map_location))