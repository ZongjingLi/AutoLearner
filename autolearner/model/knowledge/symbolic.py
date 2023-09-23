import re

import torch
import torch.nn.functional as F

from .abstract_program import AbstractProgram
from utils import copy_dict,apply,EPS

device = "cuda:0" if torch.cuda.is_available() else "cpu"
inf = torch.tensor(int(1e8)).to(device)
EPS = 1e-8

class SymbolicProgram(AbstractProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.kwargs = {}
        self.registered = None, []

    def evaluate(self, box_registry, **kwargs):

        p = super(SymbolicProgram,self)._transform(box_registry, **kwargs)
        p.kwargs = copy_dict(kwargs)
        p.registered = self.registered
        return p
    
    @property
    def object_collections(self):
        return self.kwargs["features"]

    @object_collections.setter
    def object_collections(self,other):
        self.kwargs["features"] = other
        for k in dir(self):
            if re.search("child", k):
                getattr(self, k).object_collection = other

    @property
    def relation_collections(self):
        return self.kwargs["relations"]

    @relation_collections.setter
    def relation_collections(self, other):
        self.kwargs["relations"] = other
        for k in dir(self):
            if re.search("child", k):
                getattr(self, k).relation_collections = other
    
    def apply(self, f):
        p = type(self)(*(apply(arg, f) for arg in self.arguments))
        p.kwargs = apply(self.kwargs, f)
        p.registered = self.registered
        return p

    def __call__(self, executor):
        raise NotImplementedError

    @property
    def right_most(self):
        if isinstance(self.arguments[-1], SymbolicProgram):
            return self.arguments[-1].right_most
        else:
            return self.arguments[-1]


    def __len__(self):
        length = 0
        for a in self.arguments:
            if isinstance(a, SymbolicProgram):
                length = max(length, len(a))
        return length + 1

    def register_token(self, concept_id):
        for i,arg in enumerate(self.arguments):
            # noinspecion PyUnresolvedReferences
            if isinstance(arg, SymbolicProgram):
                arg.register_token(concept_id)
            else:
                arg = torch.tensor(arg)
                if (arg == concept_id).any():
                    self.registered = i,(arg == concept_id).nonzero(as_tuple=False).tolist()
        return self

    def evaluate_token(self, queried_embedding):
        arguments = []
        for i,arg in enumerate(self.arguments):
            if torch.is_tensor(arg) and i == self.registered[0]:
                for t in self.registered[1]:
                    arg[tuple(t)] = queried_embedding
            elif isinstance(arg, SymbolicProgram):
                arg = arg.evaluate_token(queried_embedding)
            arguments.append(arg)
        
        program =  type(self)(*arguments)
        program.kwargs = copy_dict(self.kwargs)
        program.registered = self.registered
        return program


class Scene(SymbolicProgram):
    BIG_NUMBER = 100
    
    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self,executor):
        EPS = 1e-8
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        features = executor.kwargs["features"]
        #logit = torch.ones(features.shape[0] ,device = features.device) * self.BIG_NUMBER
        scores = executor.kwargs["end"]

        scene_tree= executor.kwargs["features"]
        effective_level = executor.effective_level
        logits = []
        for i,score in enumerate(scores):
            #print(i+1,len(score),effective_level)
            score = score.clamp(EPS, 1 - EPS)
            if (i+1<=effective_level):logits.append(torch.log(score / (1 - score)))
            else:logits.append(torch.log(EPS * torch.ones_like(score).to(device)/(1 - EPS * torch.ones_like(score))))
        return {"end":logits}

class Unique(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args
    
    def __call__(self, executor):
        child = self.child(executor)
        logit = child["end"]
        if executor.training:
            prob = F.softmax(logit, dim = -1)
        else:
            prob = F.one_hot(logit.max(-1).indices, logit.shape[-1])
        return {**child,"end":prob}

class Filter(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.child, self.concept = args
        self.flag = True

    def __call__(self, executor):
        child = self.child(executor)
        tree_masks = [ ]

        for i in range(len(child["end"])):
            level_mask = []
            #print(executor.kwargs["features"][i].shape)
            #for feature in executor.kwargs["features"][i]:
            features = executor.kwargs["features"][i]
            if not self.flag:
                mask_value = torch.min(child["end"][i],executor.entailment(features,
                executor.get_concept_embedding(self.concept)))
            else:            

                mask_value = self.get_normalized_prob(features,self.concept,executor)

                mask_value = mask_value.clamp(EPS, 1-EPS)

                mask_value = torch.log(mask_value/ (1 - mask_value))
                mask_value = torch.min(child["end"][i], mask_value)

            level_mask.append(mask_value)
            #print(torch.cat(level_mask, dim = -1))
            tree_masks.append(torch.cat(level_mask, dim = -1))
       
   
        return {**child, "end": tree_masks, 
            "feature": executor.kwargs["features"],} #"query_object": query_object}

    def get_normalized_prob(self, feat, concept, executor):
        pdf = []

        for predicate in executor.concept_vocab:
            pdf.append(torch.sigmoid(executor.entailment(feat,
                executor.get_concept_embedding(predicate) )).unsqueeze(0) )
  
        pdf = torch.cat(pdf, dim = 0)
        idx = executor.concept_vocab.index(concept)

        return pdf[idx]#/ pdf.sum(dim = 0)




class Relate(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, self.direction_collections = args

    def __call__(self, executor):
        child = self.child(executor)
        mask = executor.entailment(self.relation_collections.unsqueeze(0),
            self.direction_collections.unsqueeze(1).unsqueeze(1))
        new_prob = (child["end"].unsqueeze(1) * torch.sigmoid(mask)).sum(-1).clamp(EPS, 1 - EPS)
        new_logit = torch.log(new_prob) - torch.log(1 - new_prob)
        return {**child, "end": new_logit}

class Intersect(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        tree_logits = [
            torch.min(left_child["end"][i], right_child["end"][i]) for i in range(len(right_child["end"]))
        ]
        return {**left_child, **right_child, "end": tree_logits}


class Union(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        tree_logits = [
            torch.max(left_child["end"][i], right_child["end"][i]) for i in range(len(right_child["end"]))
        ]
        return {**left_child, **right_child, "end": tree_logits}


class Count(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor)
        if executor.training:
            count = 0
            for level in child["end"]:count += torch.sigmoid(level).sum(-1)
        else:
            count = (child["end"] >= 0).sum(-1)

        return {**child, "end": count}


class CountGreater(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        if executor.training:
            left_count = torch.sigmoid(left_child["end"]).sum(-1)
            right_count = torch.sigmoid(right_child["end"]).sum(-1)
            logit = 4 * (left_count - right_count - .5)
        else:
            left_count = (left_child["end"] >= 0).sum(-1)
            right_count = (right_child["end"] >= 0).sum(-1)
            logit = -10 + 20 * (left_count > right_count).float()

        return {**left_child, **right_child, "end": logit}


class CountLess(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        if executor.training:
            left_count = torch.sigmoid(left_child["end"]).sum(-1)
            right_count = torch.sigmoid(right_child["end"]).sum(-1)
            logit = 4 * (-left_count + right_count - .5)
        else:
            left_count = (left_child["end"] >= 0).sum(-1)
            right_count = (right_child["end"] >= 0).sum(-1)
            logit = -10 + 20 * (left_count < right_count).float()

        return {**left_child, **right_child, "end": logit}


class CountEqual(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        if executor.training:
            left_count = torch.sigmoid(left_child["end"]).sum(-1)
            right_count = torch.sigmoid(right_child["end"]).sum(-1)
            logit = 8 * (.5 - (left_count - right_count).abs())
        else:
            left_count = (left_child["end"] >= 0).sum(-1)
            right_count = (right_child["end"] >= 0).sum(-1)
            logit = -10 + 20 * (left_count == right_count).float()

        return {**left_child, **right_child, "end": logit}


class Query(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, self.concept_collections = args

    def __call__(self, executor):
        child = self.child(executor)
        mask = executor.entailment(self.object_collections.unsqueeze(0).unsqueeze(2),
            self.concept_collections.unsqueeze(1))
        mask = torch.sigmoid(mask) / (torch.sigmoid(mask).sum(-1, keepdim=True) +EPS)
        new_prob = (child["end"].unsqueeze(2) * mask).sum(1).clamp(EPS, 1 - EPS)
        out = {**child, "end": new_prob, "queried_embedding": self.concept_collections[0],
            "feature": self.object_collections}
        if self.registered[0] is not None:
            out["query_object"] = mask[0, :, self.registered[1][0][1]].max(-1).indices
        return out


class QueryAttributeEqual(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.left_child, self.right_child, self.attribute_collections = args

    def __call__(self, executor):
        left_child = self.left_child(executor)
        right_child = self.right_child(executor)
        mask = executor.entailment(self.relation_collections.unsqueeze(0),
            self.attribute_collections.unsqueeze(1).unsqueeze(1))
        new_prob = (left_child["end"].unsqueeze(1) * torch.sigmoid(mask) * right_child["end"].unsqueeze(2)).sum(
            (1, 2)).clamp(EPS, 1 - EPS)
        new_logit = torch.log(new_prob) - torch.log(1 - new_prob)
        return {**left_child, **right_child, "end": new_logit}


class Exist(SymbolicProgram):

    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor)
        max_logit = -inf
        for level in child["end"]:
            new_logit, query_object = level.max(-1)

            new_logit = new_logit.clamp(-inf,inf)
            max_logit = torch.max(max_logit, new_logit)
        return {**child, "end": max_logit}

class Parents(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor); 
        tree_logits = [score for score in child["end"]]
        connections = executor.kwargs["connections"]
        assert len(connections) + 1 == len(child["end"]),\
            print("Invalid Scene Connection Detected: scene:{} connection:{}".format(len(child["end"]),len(connections)))
        for i in range(0,len(tree_logits)-1):
            current_scores = torch.sigmoid(tree_logits[i])
            path_prob = torch.einsum("mn,n->m",connections[i], current_scores)

            tree_logits[i + 1] = path_prob
            tree_logits[i + 1] = torch.log(tree_logits[i + 1] / (1 - tree_logits[i + 1]))
        tree_logits[0] = torch.log(torch.ones([len(tree_logits[0])],device = device) * EPS)

        return {**child, "end": tree_logits}

class Subtree(SymbolicProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.child, = args

    def __call__(self, executor):
        child = self.child(executor)
        tree_logits = child["end"]
        connections = executor.kwargs["connections"]
        tree_logits = [score for score in child["end"]]
        #print(tree_logits)
        assert len(connections) + 1 == len(child["end"]),\
            print("Invalid Scene Connection Detected: scene:{} connection:{}".format(len(child["end"]),len(connections)))

        for i in range(len(tree_logits)-1,0,-1):
            current_scores = torch.sigmoid(tree_logits[i])

            path_prob = torch.einsum("mn,m->n",connections[i - 1], current_scores)
            path_prob = []
            for j in range(connections[i-1].shape[1]):

                pb = torch.max(torch.min(connections[i - 1].permute(1,0)[j], current_scores))
                path_prob.append(pb.unsqueeze(0))

            path_prob = torch.cat(path_prob, dim = -1)
   
            
            tree_logits[i - 1] = path_prob.clamp(EPS,1-EPS)
            tree_logits[i - 1] = torch.log(tree_logits[i-1] / (1 - tree_logits[i-1]))
        
        tree_logits[-1] = torch.log(torch.ones([len(tree_logits[-1])],device = device) * EPS)
        return {**child, "end": tree_logits}