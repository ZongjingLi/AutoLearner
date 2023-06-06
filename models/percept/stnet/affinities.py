import torch

from torch_geometric.nn    import max_pool_x
from torch_sparse          import coalesce
from torch_sparse  import SparseTensor
from torch_scatter import scatter_max, scatter_mean

from abc import ABC, abstractmethod

from .primary import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ControlBasedAggregation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Type = "P1"
    def affinities_and_thresholds(self, nodes, row, col):
        # Norm of difference for every node pair on grid
        edge_affinities = torch.linalg.norm(nodes[row] - nodes[col],dim = 1) 

        # Inverse mean affinities for each node to threshold each edge with
        inv_mean_affinity = scatter_mean(edge_affinities, row.to(nodes.device))
        affinity_thresh   = torch.min(inv_mean_affinity[row],
                                      inv_mean_affinity[col])
        return edge_affinities.to(device), affinity_thresh.to(device), {}


    def forward(self, x, edge_index, batch, device=device):


        row, col = edge_index

        # Collect affinities/thresholds to filter edges 

        affinities, threshold, losses = self.affinities_and_thresholds(x,row,col)

        if self.Type == "P1":
            filtered_edge_index = edge_index[:, affinities <= threshold]
        if self.Type == "P2":
            filtered_edge_index = edge_index[:, affinities <= threshold]
            

        # Coarsen graph with filtered adj. list to produce next level's nodes
        x = x.to(device)

        if x.size(0) != 0:
            try:
                node_labels    = LP_clustering(x.size(0), filtered_edge_index, 30).to(device)
            except:
                node_labels = torch.arange(x.size(0))
        else:
            node_labels = torch.arange(x.size(0))
        
        #node_labels    = LP_clustering(x.size(0), filtered_edge_index, 40).to(device)
        cluster_labels = node_labels.unique(return_inverse=True,sorted=False)[1].to(device)

        coarsened_x, coarsened_batch = max_pool_x(cluster_labels, x, batch)

        # [Very Suceptible Step, Why use this way to coarse edges]
        coarsened_edge_index = coalesce(cluster_labels[filtered_edge_index],
                              None, coarsened_x.size(0), coarsened_x.size(0))[0]

        return (coarsened_x, coarsened_edge_index, coarsened_batch,
                                                         cluster_labels, losses)


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
            filtered_edge_index = edge_index[:, affinities <= threshold]
            

        # Coarsen graph with filtered adj. list to produce next level's nodes
        x = x.to(device)

        if x.size(0) != 0:
            try:
                node_labels    = LP_clustering(x.size(0), filtered_edge_index, 30).to(device)
            except:
                node_labels = torch.arange(x.size(0))
        else:
            node_labels = torch.arange(x.size(0))
        
        #node_labels    = LP_clustering(x.size(0), filtered_edge_index, 40).to(device)
        cluster_labels = node_labels.unique(return_inverse=True,sorted=False)[1].to(device)

        coarsened_x, coarsened_batch = max_pool_x(cluster_labels, x, batch)

        # [Very Suceptible Step, Why use this way to coarse edges]
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
        #edge_affinities = 1/(1 + self.v2*recon_loss)
        
        edge_affinities = self.v2*recon_loss +  torch.linalg.norm(x[row] - x[col],dim = 1) * 0.1

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