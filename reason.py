# [Box Concept Structure for Reasoning]

import torch
import torch.nn as nn
import networkx as nx
import numpy    as np
from   utils import * 

class ConceptBox(nn.Module):
    def __init__(self,name,ctype,features = None,edges = None,dim = 100):
        super().__init__()
        self.ctype = ctype
        self.name  = name;self.d = 0.25
        self.features = nn.Parameter(features) if features is not None else nn.Parameter(torch.randn([1,dim]))
        self.edges    = nn.Parameter(edges)    if edges    is not None else nn.Parameter(torch.randn([1,dim]))
    
    def Center(self):return 1 * self.d * torch.tanh(self.features)

    def Edge(self):return 1 *  self.d * torch.sigmoid(self.edges)

    def __repr__(self):return "concept:{}".format(self.name)

class EntityBox(nn.Module):
    def __init__(self,features,name = "entity"):
        super().__init__()
        self.features = nn.Parameter(features)
        self.name = name;self.d = 0.25
        self.edges = torch.ones_like(self.features)

    def Center(self):return self.d * torch.tanh(self.features)

    def Edge(self):  return  1e-6 * self.edges


def BoxMin(box):return box.Center() - box.Edge()

def BoxMax(box):return box.Center() + box.Edge()

def soft_operator_max(x):t = 0.3;return t * torch.nn.functional.softplus(x/t)

def logVolume(box):return torch.sum(torch.log(soft_operator_max(BoxMax(box)-BoxMin(box))))

def M(box1,box2):return torch.min(BoxMax(box1),BoxMax(box2))

def m(box1,box2):return torch.max(BoxMin(box1),BoxMin(box2))

def logJointVolume(box1,box2,soft = True):
    if soft:return torch.sum(torch.log(soft_operator_max(M(box1,box2)-m(box1,box2))))
    values = torch.sum(torch.log(torch.max(M(box1,box2)-m(box1,box2),0).values))
    if str(values.detach().numpy()) == "nan": return float("inf") 
    else: return values

# it should entail the log probability of Pr[e1|e2] = Pr[e1 e2]/Pr[e2]
def logJoint(e1,e2,soft = True):return logJointVolume(e1,e2,soft) - logVolume(e2)

def calculate_categorical_log_pdf(entity,concepts):
    logprobs = [logJoint(concept,entity)for concept in concepts]
    pdf = torch.exp(torch.stack(logprobs))
    return torch.log(pdf/torch.sum(pdf))

def calculate_filter_log_pdf(entities,concept):
    output_logprobs = [logJoint(concept,entity) for entity in entities]
    output_logprobs = torch.stack(output_logprobs)
    return output_logprobs


def cast_to_entities(features):return [EntityBox(features[i:i+1]) for i in range(features.shape[0])]


def regular_result(results):
    outputs = results["outputs"]
    if isinstance(outputs,list):
        truth_index = np.argmax(results["scores"].detach().numpy())
        return outputs[truth_index]
    else:return int(outputs + 0.5)


class QuasiExecutor(nn.Module):
    def __init__(self,concepts):
        super().__init__()
        self.static_concepts = concepts["static_concepts"]
        self.dynamic_concepts = concepts["dynamic_concepts"]
        self.relations = concepts["relations"]
        self.visualize_execution = True
        self.execution_tree      = nx.DiGraph()
        self.tau = 0.2

    def ground_results(self,results,ground_truth,mode = "logp"):
        if mode == "logp":
            # ground discrete measures
            if not ground_truth.isdigit():
                values = results["outputs"]

                scores = results["scores"]
                truth_index = values.index(ground_truth)
                return scores[truth_index]
            # ground continuous measures
            else:
                ground_truth = int(ground_truth)
                return  torch.log(torch.sigmoid((0.5 -torch.abs(ground_truth - results["outputs"]))/(0.5 * 0.25)) )

        if mode == "entropy":
            # ground discrete measures
            if not ground_truth.isdigit():
                values = results["outputs"]

                scores = results["scores"]
                truth_index = values.index(ground_truth)
                loss = 0
                for i in range(len(results["outputs"])):
                    if i == truth_index:loss += scores[i]
                    else:loss +=  - scores[i].exp()
                return loss
            # ground continuous measures
            else:
                ground_truth = int(ground_truth)
                return  torch.log(torch.sigmoid((0.5 -torch.abs(ground_truth - results["outputs"]))/(0.5 * 0.25)) )

    def get_concept_by_name(self,name):
        for k in self.static_concepts:
            if k.name == name:return k
        assert False

    def sample_concept(self,concept = "AND(red,circle)"):
        if isinstance(concept,str):concept = toFuncNode(concept)
        def parse_concept(node):
            if node.token == "AND":return make_joint_concept(parse_concept(node.children[0]),parse_concept(node.children[1]))
            elif(0):pass
            else:return self.get_concept__by_name(node.token)
        return parse_concept(concept)

    def forward(self,program,context):
        if isinstance(program,str):program = toFuncNode(program)
        """
        the general scheme of the context is that there is always such terms in the
        result diction that have key:
            objects: the object features of perception
            scores:  the probability of each feature represents an actual object
        """
        def execute_node(node):
            if node.token == "scene":return context
            if node.token == "filter":
                input_set = execute_node(node.children[0])
                concept_name = node.children[1].token
                filter_concept = self.get_concept_by_name(concept_name)
                filter_pdf = calculate_filter_log_pdf(input_set["features"],filter_concept)
                filter_pdf = torch.min(filter_pdf,input_set["scores"])
                #print(filter_pdf.exp().detach().numpy(),node.children[1].token)
                return {"features":input_set["features"],"scores":filter_pdf}
            if node.token == "exist":
                input_set = execute_node(node.children[0])
                exist_prob = torch.max(input_set["scores"]).exp()

                output_distribution = torch.log(torch.stack([exist_prob,1-exist_prob]))
                return {"outputs":["True","False"],"scores":output_distribution}
            if node.token == "count":
                input_set = execute_node(node.children[0])
                return {"outputs":torch.sum(input_set["scores"].exp())}
            if node.token == "relate":
                return context
            if node.token == "unique":
                return context
            if node.token == "query":
                return context
            return 0
        results = execute_node(program)
        return results

def make_grid(concept):
    return concept

def realize_concept(concept):
    # if the concept is a primitive box
    # if the concept is union of concepts
    # if the concept is intersection of boxes
    return 0

def make_joint_concept(c1,c2):
    # if possible, make the intersection of two concepts
    joint_edge = M(c1,c2)-m(c1,c2)
    lower_bound = m(c1,c2);upper_bound = lower_bound + joint_edge
    center = ( lower_bound + upper_bound )/2.0 # the center of the box
    edge   = upper_bound - center # the edge of the box
    return ConceptBox("{} and {}".format(c1.name,c2.name),"complex",center,edge)

def sample(concept_box):
    # choose a random point in the concept box
    upper_bound = BoxMax(concept_box)
    lower_bound = BoxMin(concept_box)
    return torch.rand(upper_bound.shape) * (upper_bound - lower_bound) + lower_bound

