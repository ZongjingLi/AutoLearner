
import torch
import torch.nn as nn

import numpy as np
from torch_scatter import scatter_sum

from psgnet import *
from reason import *

def spatial_expansion(level):
    coords  =  level.spatial_coords
    centers =  level.centroids
    features = level.features
    clusters = level.clusters

    # expand center to grid level
    centers = centers[clusters]
    spatial_features = []
    for n in range(7):
        spatial_features.append(scatter_mean( ((coords - centers) ** (n+1))[:,0],clusters).unsqueeze(-1))
        spatial_features.append(scatter_mean( ((coords - centers) ** (n+1))[:,1],clusters).unsqueeze(-1))
    spatial_features.append(scatter_sum(torch.ones_like(clusters),clusters).unsqueeze(-1)/400)

    moments = torch.cat(spatial_features,dim = -1)

    return moments,features

class Avatar(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config;

    def forward(self,x):
        return x

class Mifafa(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        # [Create Visual Model]
        node_feat_size = 32
        if isinstance(config.visual_path,str):self.psgnet = torch.load(config.visual_path)
        else:self.psgnet = PSGNet(config.imsize)

        # [Entity Projector]
        self.entity_projector = FCBlock(128,3,node_feat_size + config.spatial_dim,config.concept_dim) #[NFS + SFS,C]

        # [Scene Joint Reasoning]
        circle = ConceptBox("Circle","shape",dim = 100)
        cube  = ConceptBox("Cube" ,"shape",dim = 100)
        diamond  = ConceptBox("Diamond" ,"shape",dim = 100)
        concepts = {"static_concepts":torch.nn.ModuleList([circle,cube,diamond]),"dynamic_concepts":[],"relations":["relations"]}
        self.quasi_executor = QuasiExecutor(concepts)

        # [Perception Level]
        self.levels = None
        self.context = None
    
    def scene_perception(self,im):
        outputs = self.psgnet(im) # perform perception on a single image each time
        recons, clusters, all_losses, levels = outputs["recons"],outputs["clusters"],outputs["losses"], outputs["levels"]
        self.levels = levels
        moments,features = spatial_expansion(levels[-1])
        entity = cast_to_entities(self.entity_projector(torch.cat([moments,features],dim = -1)))

        self.context = {"features":entity,"scores":torch.zeros([len(entity)]),"coords":levels[-1].centroids}
        return outputs

    def joint_reason(self,questions,cast = True):
        if isinstance(questions[0],FuncNode):
            # questions are already functional nodes
            programs = questions
        else:programs = [toFuncNode(q) for q in questions]
        context = self.context
        if cast:answers = [regular_result(self.quasi_executor(p,context)) for p in programs]
        else:answers = [self.quasi_executor(p,context) for p in programs]
        return answers 
    
    def ground_concepts(self,questions,answers):
        return 0