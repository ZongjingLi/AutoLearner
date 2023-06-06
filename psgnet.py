from types import SimpleNamespace

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.nn    import max_pool_x
from torch_geometric.utils import add_self_loops
from torch_scatter         import scatter_mean
from torch_sparse          import coalesce


from torch_sparse  import SparseTensor
from torch_scatter import scatter_max


from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch
from torch_scatter import scatter_mean,scatter_max

from primary import * 
from utils import *


import networkx as nx

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# modified from LIIF which was modified from github.com/thstkdgus35/EDSR-PyTorch

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = args.RDNconfig
        """
        {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]
        """

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


class AffinityConditionedAggregation(torch.nn.Module, ABC):

    # Takes in tensor of node pairs and returns an affinity tensor and a 
    # threshold tensor to filter the affinites with. Also returns any loss
    # items to pass back to the training layer as a dict.
    # x is the list of graph nodes and row, col are the tensors of the adj. list
    @abstractmethod
    def affinities_and_thresholds(self, x, row, col):
        pass

    # Filters edge index based on base method's affinity thresholding and 
    # coarsens graph to produce next level's nodes
    def forward(self, x, edge_index, batch, device=device):


        row, col = edge_index

        # Collect affinities/thresholds to filter edges 

        affinities, threshold, losses = self.affinities_and_thresholds(x,row,col)

        if self.Type == "P1":
            filtered_edge_index = edge_index[:, affinities <= threshold]
        if self.Type == "P2":
            filtered_edge_index = edge_index[:, affinities >= threshold]
            

        # Coarsen graph with filtered adj. list to produce next level's nodes
        x = x.to(device)

        if x.size(0) != 0:
            try:
                node_labels    = LP_clustering(x.size(0), filtered_edge_index, 70).to(device)
            except:
                node_labels = torch.arange(x.size(0))
        else:
            node_labels = torch.arange(x.size(0))
        
        #node_labels    = LP_clustering(x.size(0), filtered_edge_index, 40).to(device)
        cluster_labels = node_labels.unique(return_inverse=True,sorted=False)[1].to(device)

        coarsened_x, coarsened_batch = max_pool_x(cluster_labels, x, batch)

        # [Very Sucptible Step, Why use this way to coarse edges]
        coarsened_edge_index = coalesce(cluster_labels[filtered_edge_index],
                              None, coarsened_x.size(0), coarsened_x.size(0))[0]

        return (coarsened_x, coarsened_edge_index, coarsened_batch,
                                                         cluster_labels, losses)

class P2AffinityAggregation(AffinityConditionedAggregation):

    def __init__(self, node_feat_size, v2= 3.5 ):
        super().__init__()
        self.Type = "P2"
        self.v2 = v2
        self.node_pair_vae = VAE( in_features=node_feat_size ,beta = 30)

    # Note question to ask: why reconstructing difference of nodes versus
    # concatenating them, as many different node pairs can have similar
    # differences? Is it just to make it symmetric? Try both.l;p[']\


    
    def affinities_and_thresholds(self, x, row, col):

        # Affinities as function of vae reconstruction of node pairs
        #_, recon_loss, kl_loss = self.node_pair_vae( torch.abs(x[row]-x[col]) )
        _, recon_loss, kl_loss = self.node_pair_vae( x[row] - x[col] )
        edge_affinities = 1/(1 + self.v2*recon_loss)

        losses = {"recon_loss":recon_loss.mean(), "kl_loss":kl_loss.mean()}

        return edge_affinities, .5, losses

class P1AffinityAggregation(AffinityConditionedAggregation):
    def __init__(self):
        super().__init__()
        self.Type = "P1"

    # P1 is a zero-parameter affinity clustering algorithm which operates on
    # the similarity of features
    def affinities_and_thresholds(self, nodes, row, col):
        # Norm of difference for every node pair on grid
        edge_affinities = torch.linalg.norm(nodes[row] - nodes[col],dim = 1) 

        # Inverse mean affinities for each node to threshold each edge with
        inv_mean_affinity = scatter_mean(edge_affinities, row.to(nodes.device))
        affinity_thresh   = torch.min(inv_mean_affinity[row],
                                      inv_mean_affinity[col])
        return edge_affinities.to(device), affinity_thresh.to(device), {}

def LP_clustering(num_nodes,edge_index,num_iter=50,device=device):

    # Sparse adjacency matrix
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],value = torch.ones_like(edge_index[1]).float(),
                         sparse_sizes=(num_nodes, num_nodes)).t().to(device)

    # Each node starts with its index as its label
    x = SparseTensor.eye(num_nodes).to(device)

    for _ in range(num_iter):


        # Add each node's neighbors' labels to its list of neighbor nodes
        #row,col,v = adj_t.coo()
        #out = torch_sparse.spmm(adj_t,x)
        out = adj_t @ x
        # Argmax of each row to assign new label to each node
        row, col, value = out.coo()

        argmax = scatter_max(value, row, dim_size=num_nodes)[1]

        new_labels = col[argmax]
        x = SparseTensor(row=torch.arange(num_nodes).to(device), col=new_labels,
                            sparse_sizes=(num_nodes, num_nodes),
                            value=torch.ones(num_nodes).to(device))

    return new_labels

def QTR(grid,center,a,ah,aw,ahh,aww,ahw):
    ch,cw = center
    mx,my = grid[:,:,0],grid[:,:,1]
    var = a + \
        ah*(mx - ch) + aw*(my - cw) +\
        ahh*(mx - ch)**2 + aww*(my - cw)**2 + \
            ahw * (mx - ch) * (my - cw)
    return var

def RenderQTR(G,features):

    # Render 20 dim Features using 2 centroids and attribute
    centers   = features[:,:2] # Size:[N,2]
    paras     = features[:,2:] # Size:[N,18]
    mx,my     = G[:,0],G[:,1]
    output_channels = []
    for c in range(3):
        ch,cw = centers[:,0],centers[:,1]
        bp = paras[:,c * 6: (c + 1) * 6]

        a,ah,aw,ahh,aww,ahw = bp[:,0],bp[:,1],bp[:,2],bp[:,3],bp[:,4],bp[:,5]
        qtr_var = a + \
        ah*(mx - ch) + aw*(my - cw) +\
        ahh*(mx - ch)**2 + aww*(my - cw)**2 + \
            ahw * (mx - ch) * (my - cw)
        output_channels.append(qtr_var.unsqueeze(0))

    return torch.cat(output_channels,0).permute([1,0])

def QSR(grid,px,py,pa,pr):
    
    mx,my = grid[:,:,0],grid[:,:,1]
    return torch.sigmoid(\
        pa * (my * (torch.cos(pr)) - mx * (torch.sin(pr)) - px) ** 2 - \
             (mx * (torch.cos(pr)) + my * (torch.sin(pr)) - py)
    )

class Level:
    def __init__(self,centroids,features,edges,clusters,batch,spe):
        # input level centroids, edges connected and the batch info
        self.centroids = centroids
        
        self.features = features
        self.edges = edges
        self.clusters = clusters
        self.batch = batch
        self.spatial_coords = spe

        self.centx = scatter_mean(self.spatial_coords[:,0],clusters,dim = -1).unsqueeze(0)
        self.centy = scatter_mean(self.spatial_coords[:,1],clusters,dim = -1).unsqueeze(0)
        self.centroids = torch.cat([self.centx,self.centy],0).permute([1,0])#.unsqueeze()
    

def render_level(level,name = "Namo",scale = 64):

    plt.scatter(level.centroids[:,0] * scale,(1 - level.centroids[:,1]) * scale,c = "cyan")
    row,col = level.edges

    rc,cc = level.centroids[row] * scale,level.centroids[col] * scale

    for i in range(len(rc)):
        point1 = rc[i];point2 = cc[i]
        x_values = [point1[0], point2[0]]
        y_values = [scale - point1[1], scale - point2[1]]
        plt.plot(x_values,y_values,color = "red",alpha = 0.3)


def optical_flow_motion_mask(video):
    masks = 0
    return masks

class PSGNet(torch.nn.Module):
    def __init__(self,imsize):

        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.imsize = imsize

        node_feat_size   = 32
        num_graph_layers = 2

        
        self.spatial_edges,self.spatial_coords = grid(imsize,imsize,device=device)
        
        # [Global Coords]
        self.global_coords = self.spatial_coords.float()

        self.spatial_edges = build_perception(imsize,2,device = device)
        self.spatial_coords = self.spatial_coords.to(device).float() / imsize
        # Conv. feature extractor to map pixels to feature vectors
        self.rdn = RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))

        #self.rdn = nn.Conv2d(3,node_feat_size,3,1,1)

        # Affinity modules: for now just one of P1 and P2 
        self.affinity_aggregations = torch.nn.ModuleList([
            P1AffinityAggregation(),
            P2AffinityAggregation(node_feat_size),
            P2AffinityAggregation(node_feat_size),
            #P2AffinityAggregation(node_feat_size),

        ])

        # Node transforms: function applied on aggregated node vectors

        self.node_transforms = torch.nn.ModuleList([
            FCBlock(hidden_ch=100,
                    num_hidden_layers=3,
                    in_features =node_feat_size + 4,
                    out_features=node_feat_size,
                    outermost_linear=True) for _ in range(len(self.affinity_aggregations))
        ])

        # Graph convolutional layers to apply after each graph coarsening
        gcv = GraphConv(node_feat_size, node_feat_size)  
        self.graph_convs = torch.nn.ModuleList([
            GraphConv(node_feat_size , node_feat_size ,aggr = "mean")   for _ in range(len(self.affinity_aggregations))
        ])

        # Maps cluster vector to constant pixel color
        self.node_to_rgb  = FCBlock(hidden_ch=100,
                                    num_hidden_layers=3,
                                    in_features =20,
                                    out_features=3,
                                    outermost_linear=True)

        self.node_to_qtr_p1  = FCBlock(100,3,node_feat_size,6 * 3,outermost_linear = True)
        self.node_to_qtr_p2  = FCBlock(100,3,node_feat_size,6 * 3,outermost_linear = True)
        self.gauge = nn.Linear(node_feat_size,node_feat_size)


    def dforward(self,imgs,effective_mask = None):
        return 0

    def forward(self,img,effective_mask = None):
        batch_size = img.shape[0]

        mask_shape = [batch_size,self.imsize,self.imsize,1]
        
        # [ Effective Mask Not Considered ]
        if effective_mask is None: effective_mask = torch.ones(mask_shape)


        # [Create the Local Coords]

        # Collect image features with rdn

        im_feats = self.rdn(img.permute(0,3,1,2))
        #im_feats = img.permute(0,3,1,2) 

        #coords_added_im_feats = torch.cat([
        #          self.spatial_coords.unsqueeze(0).repeat(im_feats.size(0),1,1),
        #          im_feats.flatten(2,3).permute(0,2,1)
        #                                  ],dim=2)
        coords_added_im_feats = im_feats.flatten(2,3).permute(0,2,1)

        ### Run image feature graph through affinity modules

        graph_in = Batch.from_data_list([Data(x,self.spatial_edges)
                                                for x in coords_added_im_feats])

        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch

        clusters, all_losses = [], [] # clusters just used for visualizations
        intermediates = [] # intermediate values
        levels = []

        ## Perform Affinity Calculation and Graph Clustering

        
        for pool, conv, transf in zip(self.affinity_aggregations,
                                      self.graph_convs, self.node_transforms):
            batch_uncoarsened = batch

            x, edge_index, batch, cluster, losses = pool(x, edge_index, batch)

            clusters.append( (cluster, batch_uncoarsened) )
            for i,(cluster_r,_) in enumerate(clusters):
                for cluster_j,_ in reversed(clusters[:i]):cluster_r = cluster_r[cluster_j]
                device = self.device

                centroids = scatter_mean(self.spatial_coords.repeat(batch_size,1).to(device),cluster_r.to(device),dim = 0)
                moments   = scatter_mean(self.spatial_coords.repeat(batch_size,1).to(device) ** 2,cluster_r.to(device),dim = 0 )
            
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
    

            # augument each node with explicit calculation of moment and centroids
            x = torch.cat([x,centroids,moments],dim = -1)
            x = transf(x)
            x = conv(x, edge_index)

            all_losses.append(losses)
            intermediates.append(x)
            levels.append(Level(centroids = centroids,features =x, edges = edge_index,clusters = cluster_r, batch = batch,spe = self.spatial_coords))

        joint_spatial_features = []

        for i,(cluster_r,_) in enumerate(clusters):
            for cluster_j,_ in reversed(clusters[:i]):cluster_r = cluster_r[cluster_j]
            centroids = scatter_mean(self.spatial_coords.repeat(batch_size,1),cluster_r,dim = 0)

            joint_features = torch.cat([self.node_to_qtr_p2(intermediates[i]),centroids],-1)
            joint_spatial_features.append(joint_features)

        recons = [] # perform reconstruction over each layer composed

        for i,jsf in enumerate(joint_spatial_features):
            for cluster,_ in reversed(clusters[:i+1]):jsf = jsf[cluster]

            
            paint_by_numbers = to_dense_batch( 1.0 *  (\
                RenderQTR(self.spatial_coords.repeat(batch_size,1),jsf))\
                    ,graph_in.batch)[0]
            recons.append(paint_by_numbers)
            
        return {"recons":recons,"clusters":clusters,"losses":all_losses,"features":intermediates,"levels":levels}
       
