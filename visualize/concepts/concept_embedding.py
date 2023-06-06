from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from visualize.visualizer import Visualizer

class ConceptEmbeddingVisualizer(Visualizer):
    per_batch = False

    def visualize(self, results, model, concept_split_specs, iteration, **kwargs):
        embeddings = results["embedding"]
        labels = results["label"]
        # dimension_stddev
        if model.rep == "box":
            dim = embeddings.shape[-1] // 2
            anchors = embeddings[:, :dim]
        else:
            anchors = embeddings
        tag = "train"
        std = torch.std(anchors, 0)
        self.summary_writer.add_histogram(f"weight_stddev_by_dimension/{tag}/ordered", std,
            iteration)
        
        fig, ax = plt.subplots()
        fig.set_size_inches(8,8)

        ax.scatter(list(range(len(std))), std.sort().values.tolist())
        ax.set_ylabel("Stddev")

        self.summary_writer.add_figure(f"weight_stddev_by_dimension/{tag}/sorted", fig, iteration)

        # wright by split
        split2weight = defaultdict(list)

        for e,l in zip(embeddings, labels):
            if model.rep == "box":
                dim = e.shape[-1] // 2
                size = e[dim:].mean()
            else:
                size = e.norm()
            split2weight[concept_split_specs[l]].append(size)
        split2weight = {k: torch.stack(v) for k, v in split2weight.items()}

        # weight by dimension
        if model.rep == "box":
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            dim = embeddings.shape[-1] // 2
            offset = torch.arange(dim)
            xs = (embeddings[:, offset] - embeddings[:, dim + offset]).flatten().tolist()
            ys = (embeddings[:, offset] + embeddings[:, dim + offset]).flatten().tolist()
            ax.hist2d(xs, ys, bins=40, range=[[-.5, .5], [-.5, .5]])
            ax.plot([0, 1], [0, 1], transform=ax.transAxes)
            ax.set_xlabel("Box begin")
            ax.set_ylabel("Box end")
            self.summary_writer.add_figure(f"weight_by_dimension/{tag}/global", fig, iteration)
            self.dataset
            for offset in range(5):
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 8)
                xs = (embeddings[:, offset] - embeddings[:, dim + offset]).flatten().tolist()
                ys = (embeddings[:, offset] + embeddings[:, dim + offset]).flatten().tolist()
                ax.hist2d(xs, ys, bins=40, range=[[-.5, .5], [-.5, .5]])
                ax.plot([0, 1], [0, 1], transform=ax.transAxes)
                ax.set_xlabel("Box begin")
                ax.set_ylabel("Box end")
                self.summary_writer.add_figure(f"weight_by_dimension/{tag}/local", fig,
                    iteration + offset)

        else:
            self.summary_writer.add_histogram(f"weight_by_dimension/{tag}/global", embeddings,
                iteration)
            for offset in range(5):
                self.summary_writer.add_histogram(f"weight_by_dimension/{tag}/local",
                    embeddings[:, offset], iteration + offset)