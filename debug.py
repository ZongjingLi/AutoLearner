
from models import *
from config import *

from datasets import *

import math

dataset = ToyData("train")

idx = [4, 7, 9, 11]
ims = torch.cat([dataset[i]["image"].unsqueeze(0) for i in idx])

print(ims.shape)

percept = PSGNet(config.imsize,3)


outputs = percept(ims)

def normalize_outputs(outputs):
    recons = outputs["recons"]
    B = recons[0].shape[0]
    W = int(math.sqrt(recons[0].shape[1]))

    clusters = outputs["clusters"]

    for item in clusters:print(item[0].shape, item[1].shape)

    level_reconstructions = [item.reshape([B,W,W,3]) for item in recons]
    level_scene_graphs  = []

    return {"recons":level_reconstructions}

outputs = normalize_outputs(outputs)
recons = outputs["recons"]

for i,item in enumerate(recons):print("level:{}".format(i),item.shape)

import torch
import torch.nn as nn

B = 1
M = 20
N = 8
C = 64

class AbstractNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.num_heads =8
        self.feature_decoder = nn.Transformer(nhead=16, num_encoder_layers=12,d_model = config.global_feature_dim,batch_first = True)
        self.spatial_decoder = nn.Transformer(nhead=16, num_encoder_layers=12,d_model = config.global_feature_dim,batch_first = True)
        self.source_heads = nn.Parameter(torch.randn([self.num_heads,config.global_feature_dim]))

        self.coordinate_decoder = nn.Linear(config.global_feature_dim, 2)
    
    def forward(self, feature, spatial):
        B, M, C = feature.shape
        N = self.num_heads
        # [Feature Propagation]
        component_features = feature
        component_spaitals = spatial

        # [Decode Proposals]
        global_feature = torch.randn()
        source_heads = self.souce_heads
        feature_proposals = self.feature_decoder(source_heads,global_feature)
        spaital_proposals = self.spatial_decoder(source_heads,global_feature)

        # [Component Matching]
        # component_features : [B,M,C]
        # feature_proposals  : [B,N,C]

        match = torch.softmax(torch.einsum("bnc,bmc -> bnm",component_features, proposal_features)/math.sqrt(C), dim = -1)
        existence = torch.max(match, dim = 1).values  # [B, N, 1]

        # [Construct Representation]
        output_graph = 0

        return output_graph