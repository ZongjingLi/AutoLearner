import pickle
import json 
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        edges[0].append(loc)
                        edges[1].append( (i+dx) * size + (j + dy))
    return torch.tensor(edges).to(device)

from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_large

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

from torchvision.models.optical_flow import Raft_Large_Weights




def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960])
    img2_batch = F.resize(img2_batch, size=[520, 960])
    return transforms(img1_batch, img2_batch)


class Stack(object):
    def __init__(self,top = 0):
        self.top_pointer = top
        self.values = []
    def __str__(self):
        return "Stack:\n (pointer = {},value = {})".format(self.top_pointer,
                                                            self.top_pointer)
    def __eq__(self,o): return isinstance(o,Stack) and self.top_pointer == o.top_pointer

    def __ne__(self,o): return not self == o

class Queue(object):
    def __init__(self,front = 1,rear = 0,values = []):
        self.front = front
        self.rear = rear
        self.values = [-1] * 99999
    def __str__(self): return "Queue:\n (front = %d,rear = %d)" %(self.front,self.rear)

    def __eq__(self,o): return  isinstance(o,Queue) and self.front == o.front and self.rear == o.rear and self.values == o.values

    def __ne__(self,o): return not self == o

    def put(self,x):
        self.rear += 1
        self.values[self.rear] = x
    
    def pop(self):
        value = self.values[self.front]
        self.front += 1
        return value

    def top(self):return self.values[self.front]

    def empty(self):return self.front == self.rear + 1



class FuncNode:
    def __init__(self,token):
        self.token = str(token)
        self.content = token
        self.children = []
        self.father = []
        self.prior_length = None
        self.function = True
        self.type = "Function"
        self.isRoot = False
    def has_children(self): return len(self.children) != 0
    
    def has_args(self): return len(self.children) != 0
    
    def __str__(self):
        return_str = ""
        return_str += self.token + ""
        if (self.function):
            return_str += "("
        for i in range(len(self.children)):
            arg = self.children[i]
            # perform the unlimited sequence processing
            return_str += arg.__str__()
            max_length = len(self.children)
            if (self.prior_length != None):
                max_length = len(self.children)
            if (i < max_length -1):
                return_str += ","

        if (self.function):
            return_str += ")"
        return return_str
    
    def add_child(self,version_space):
        self.children.append(version_space)
        
    def add_token(self,token):
        vs = FuncNode(token)
        self.children.append(vs)

    def clear(self):self.children = []
    
    def length(self):
        score = 0
        if (len(self.children) > 0):
            for child in self.children:
                score += child.length()
        return 1 + score


def find_bp(inputs):
    loc = -1
    count = 0
    for i in range(len(inputs)):
        e = inputs[i]
        if (e == "("):
            count += 1
        if (e == ")"):
            count -= 1
        if (count == 0 and e == ","):
            return i
    return loc

def break_braket(program):
    try:loc = program.index("(")
    except:loc = -1
    if (loc == -1):return program,-1
    token = program[:loc]
    paras = program[loc+1:-1]
    return token,paras        

def break_paras(paras):
    try:loc = find_bp(paras)
    except:loc = -1
    if (loc == -1):return paras,True
    token = paras[:loc]
    rest = paras[loc+1:]
    return token,rest

def to_parameters(paras):
    flag = True
    parameters = []
    while (flag):
        token,rest = break_paras(paras)
        if (rest == True):flag = False
        parameters.append(toFuncNode(token));paras = rest
    return parameters

def toFuncNode(program):
    token,paras = break_braket(program)
    curr_head = FuncNode(token)
    if (paras == -1):curr_head.function = False;curr_head.children = []
    else:children_paras = to_parameters(paras);curr_head.children = children_paras
    return curr_head

def dnp(tensor,gpu = False):
    # detach the tensor from the calculation tree
    if gpu: return tensor.cpu().detach().numpy()
    return tensor.detach().numpy()

def progress_bar(count, total, status=''):

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
# load json data
def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
        return data

def save_json(data,path):
    '''input the diction data and save it'''
    beta_file = json.dumps(data)
    file = open(path,'w')
    file.write(beta_file)
    return True

# load pickle data
def save_pickle(data,name):
    with open(str(name), 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_name):
    pkl_file = open(file_name, 'rb')
    data = pickle.load(pkl_file)
    return data

"""
img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")
predicted_flows = list_of_flows[-1]

flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
#plot(grid)
"""

