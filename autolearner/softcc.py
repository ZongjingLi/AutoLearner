from config import *
from model import *
from datasets import *

class GraphFuser(nn.Module):
    def __init__(self, features, edges, scores = None):
        super().__init__()
        self.features = features
        self.scores = scores if scores is not None else torch.ones(features.shape[0])
        self.edges = edges

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.max_label = torch.tensor(0.0).to(self.device)
    
    def cut_softcc(self):
        self.max_label = torch.tensor(0.0).to(self.device)
        labels = torch.zeros_like(self.scores)
        visited = [False] * self.scores.shape[0]
        def dfs(u):
            visited[u] = True
            edge_weights_u = self.edges[u]
            def key_func(i): return edge_weights_u[i]
            sorted_nodes = sorted(list(range(len(edge_weights_u))), key=key_func)
            for v in sorted_nodes:
                if u != v and not visited[v]:
                    labels[v] = labels[u]
                    labels[v] += self.scores[u] * self.scores[v] * (1 - edge_weights_u[v])
                    self.max_label = max(self.max_label, labels[v])
                    dfs(v)
        for i in range(len(visited)):
            if not visited[i]:
                labels[i] = self.max_label
                dfs(i)
        return labels, self.max_label

if __name__ == "__main__":
    test_features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.1]
    ])


    idx = [[0, 2], [1, 0], [2, 0], [0, 1]]
    v =  [1,      1,      1,        1]
    n = len(test_features)
    test_edges = torch.sparse_coo_tensor(list(zip(*idx)), v, (n,n))
    test_edges = test_edges.to_dense()

    #test_edges = test_edges.to_sparse()
    print(test_edges)

    fuser = GraphFuser(test_features, test_edges)

    labels, max_label = fuser.cut_softcc()
    print(labels)
    print(max_label)
