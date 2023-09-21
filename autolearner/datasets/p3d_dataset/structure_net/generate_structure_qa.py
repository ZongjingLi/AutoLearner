import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import networkx as nx
import os
import random
from tqdm import tqdm

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    '#2e93d1',
    '#F97934',
    '#C3F2F9',
    '#FFFFF1'
]

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

colors = [torch.tensor(hex2rgb(color))/255.0 for color in colors]
color_names = ["red" for _ in range(10)]

import json

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

def visualize_pointcloud(input_pcs,name="pc"):
    rang = 1.0; N = len(input_pcs)
    num_rows = 3
    fig = plt.figure("visualize",figsize=plt.figaspect(1/N), frameon = True)
    for i in range(N):
        ax = fig.add_subplot(1, N , 1 + i, projection='3d')
        ax.set_zlim(-rang,rang);ax.set_xlim(-rang,rang);ax.set_ylim(-rang,rang)
        # make the panes transparent
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo["grid"]['color'] =  (1,1,1,0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_axis_off()
        #ax.view_init(elev = -80, azim = -90)
        coords = input_pcs[i][0]
        colors = input_pcs[i][1]
        ax.scatter(coords[:,0],coords[:,1],coords[:,2], c = colors)
    plt.savefig("outputs/{}.png".format(name))


def find_outliers(pc):
    # Calculate mean distance from each point to every other point
    pc_expand = pc[None, :, :].repeat(pc.shape[0], axis=0)
    pc_expand = pc_expand - pc[:, None, :].repeat(pc.shape[0], axis=1)
    mean_dists = np.mean(np.linalg.norm(pc_expand, axis=2), axis=1)
    # Assume mean distances are normally distributed
    mean_dists = (mean_dists - np.mean(mean_dists)) / np.std(mean_dists)
    return mean_dists > 3.0

def unit_normalize_pc(pc, filter_outliers=True,):
    if filter_outliers:
        # Calculate outliers
        mask = find_outliers(pc)
        # Filter out the really extreme points
        first_ind = np.argwhere(~mask)[0][0]
        pc[mask] = pc[first_ind]
    # Normalized point cloud to unit sphere
    # Center the point cloud at origin

    mean_coord = pc.mean(0)
    pc = pc - mean_coord
    # Calculate the distance between the origin and the furthest point
    dists = np.linalg.norm(pc, axis=1)
    max_dist = np.max(dists)
    # Divide points by max_dist and fit inside unit sphere
    return pc / max_dist

def calculate_dists_between_pointsets(points, new_points):
    # points is a matrix of B x N x C
    # new_points is a matrix of B x M x C
    # Returns matrix B x M x N of distances from points
    dists = -2 * torch.bmm(new_points, points.mT)
    dists += torch.sum(torch.pow(points, 2), 2)[:, None, :]
    dists += torch.sum(torch.pow(new_points, 2), 2)[:, :, None]
    return dists


def fpsampling(pointset, m, return_cent_inds=False, random_start=True):
    # Iterative farthest point sampling
    # The next point added is the point that is the farthest away from the nearest centroid
    # pointset is a matrix of B x N x 3
    # Returns matrix of B x M x 3 centroid points
    # Returns matrix B x M x N of distances from points
    num_batch, num_points, _ = pointset.size()
    batch_idx = torch.arange(num_batch, dtype=torch.long).to(pointset.device)
    # Initialize centroid points
    centroids = torch.zeros(num_batch, m, dtype=torch.long).to(pointset.device)
    actual_cents = torch.zeros(num_batch, m, 3).to(pointset.device)
    # Initialize distances
    distances = torch.ones(num_batch, num_points).to(pointset.device) * 1e10
    if random_start:
        farthest = torch.randint(0, num_points, (num_batch,),
                                 dtype=torch.long).to(pointset.device)
    else:
        farthest = torch.zeros(num_batch, dtype=torch.long).to(pointset.device)
    # Iterative point sampling
    for i in range(m):
        # Select next centroids
        centroids[:, i] = farthest
        actual_cents[:, i, :] = pointset[batch_idx, farthest, :]
        # Calculate distances and next point
        dists = torch.square(pointset - actual_cents[:, i:i+1, :]).sum(dim=2)
        distances = torch.min(distances, dists)
        farthest = torch.max(distances, 1)[1]
    if return_cent_inds:
        return centroids, actual_cents
    return actual_cents

def create_box(box_params):
    center = box_params[0: 3]
    lengths = box_params[3: 6]
    dir_1 = box_params[6: 9]
    dir_2 = box_params[9:]
    #
    s = 1.0
    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)
    #
    d1 = s * 0.5*lengths[0]*dir_1
    d2 = s * 0.5*lengths[1]*dir_2
    d3 = s * 0.5*lengths[2]*dir_3
    #
    verts = np.zeros([8, 3])
    verts[0, :] = center - d1 - d2 - d3
    verts[1, :] = center - d1 + d2 - d3
    verts[2, :] = center + d1 - d2 - d3
    verts[3, :] = center + d1 + d2 - d3
    verts[4, :] = center - d1 - d2 + d3
    verts[5, :] = center - d1 + d2 + d3
    verts[6, :] = center + d1 - d2 + d3
    verts[7, :] = center + d1 + d2 + d3
    #
    faces = np.zeros([6, 4], dtype='int64')
    faces[0, :] = [3, 2, 0, 1]
    faces[1, :] = [4, 6, 7, 5]
    faces[2, :] = [0, 2, 6, 4]
    faces[3, :] = [3, 1, 5, 7]
    faces[4, :] = [2, 3, 7, 6]
    faces[5, :] = [1, 0, 4, 5]
    #
    return verts, faces

root = "/Users/melkor/Documents/datasets"

def get_scene_tree(h):
    return 
    
def get_leafs(h):
    nodes = []
    if "children" in h:
        for child in h["children"]:
            nodes.extend(get_leafs(child))
        return nodes
    else:
        return [h["id"]]

def get_labels(h):
    nodes = []
    if "children" in h:
        for child in h["children"]:
            nodes.extend(get_labels(child))
        return nodes
    else:
        return [h["label"]]

def build_scene_tree(h):
    nodes = []

    if "children" in h:
        for child in h["children"]:
            nodes.extend(get_labels(child))
        return nodes
    else:
        return [h["label"]]

def dfs_point_cloud(pc_data, nodes = None):
    point_cloud = []
    rgbs = []
    num_pt = 1000
    fig = plt.figure("namo")
    ax = Axes3D(fig)
    if nodes is None: nodes = range(pc_data.shape[0])
    for i in nodes:
        k = np.random.randint(0,10 - 1)
        color_name = color_names[k]
        color = torch.tensor(colors[k])
        pts = torch.tensor(pc_data[i])
        ax.scatter(pts[:,0],pts[:,1],pts[:,2], color = [np.array(color) for _ in range(pts.shape[0])])    
        point_cloud.append(pts)
        rgbs.append(color.unsqueeze(0).repeat(num_pt,1))
    point_cloud = torch.cat(point_cloud, dim = 0)
    rgbs = torch.cat(rgbs, dim = 0)
    return point_cloud, rgbs

def generate_full(cat = "chair", idx = 172, pts_num = 2048):
    pc_path = root + "/partnethiergeo/{}_geo/{}.npz".format(cat, idx)
    pc_data = np.load(pc_path)

    # [Point Cloud]
    pc = torch.tensor(pc_data["parts"][0])

    hier_path = root + "/partnethiergeo/{}_hier/{}.json".format(cat, idx)
    hier_data = load_json(hier_path)

    pc, rgbs = dfs_point_cloud(pc_data["parts"], get_leafs(hier_data))
    valid_indices = random.sample(list(range(pc.shape[0])), pts_num)
    pc = pc[valid_indices,:]
    rgbs = rgbs[valid_indices,:]
    
    scene_tree = get_scene_tree(hier_data)
    
    return {"point_cloud":pc, "rgbs":rgbs,"scene_tree":scene_tree,"questions":questions,"answers":answers}

def build_labels(h,voc):
    if h["label"] not in voc: voc.append(h["label"])
    if "children" in h:
        for child in h["children"]:
            build_labels(child, voc)

def generate_structure(cat = "chair", idx = 176, full_grounding = True):
    pc_path = root + "/partnethiergeo/{}_geo/{}.npz".format(cat, idx)
    pc_data = np.load(pc_path)

    # [Point Cloud]
    pc = torch.tensor(pc_data["parts"][0])

    # [Build Scene Tree]
    hier_path = root + "/partnethiergeo/{}_hier/{}.json".format(cat, idx)
    hier_data = load_json(hier_path)

    scene_tree = nx.DiGraph()
    uniform_tree = nx.DiGraph()
    def build(h,root):
        name = h["label"]+ "_" + np.random.choice([*"abcedfghijklmnopqrstuvwxyz"])
        uniform_name = h["label"]
        uniform_tree.add_node(uniform_name)
        uniform_tree.add_edge(root[:-2],uniform_name)
        if "children" in h:
            for child in h["children"]:
                build(child, name)
    def build_st(h,root):
        name = h["label"]
        scene_tree.add_node(name)
        scene_tree.add_edge(root, name)
        if "children" in h:
            for child in h["children"]:
                build_st(child, name)
    build(hier_data, "root")
    build_st(hier_data,"root")

    # [Build Question Ansering Pairs] build category labels in the scene
    scene_labels = [];  build_labels(hier_data,scene_labels)
    all_labels = ["pot","body","container","containing_things","liquid_or_soil",
                "plant","other","lid","base","foot_base","foot"]
    if not full_grounding:
        qa_pairs = []
        # Existence Questions
        # Template: Is there any {part} in the scene
        # program: exist(filter(scene(),part))
        num_existence = len(scene_labels)
        for n in range(num_existence):
            part = np.random.choice(scene_labels)
            question = "Is there any {} in the scene?".format(part)
            answer = "yes"
            program = "exist(filter(scene(),{}))".format(part)
            qa_pairs.append({"type":"existence","question":question,"program":program,"answer":answer})

        for n in range(num_existence * 1):
            part = np.random.choice(all_labels)
            question = "Is there any {} in the scene??".format(part)
            answer = "yes" if part in scene_labels else "no"
            program = "exist(filter(scene(),{}))".format(part)
            qa_pairs.append({"type":"existence","question":question,"program":program,"answer":answer})

        # Hierarchy Questions
        # Template: Does {part_1} contains {part_2}
        # program: exist(filter(subtree(filter(scene(),part_1)),part_2))
        num_hierarchy = len(scene_labels) * 2
        for n in range(num_hierarchy):
            if len(scene_labels) > 1:
                part_1 = np.random.choice(scene_labels)
                remain_labels = []
                for label in scene_labels:
                    if label != part_1:remain_labels.append(label)
        
                part_2 = np.random.choice(remain_labels)
                question = "Does {} contains {}?".format(part_1, part_2)
                answer = "yes" if nx.has_path(uniform_tree,part_1,part_2) else "no"
                program = "exist(filter(subtree(filter(scene(),{})),{}))".format(part_1,part_2)
                qa_pairs.append({"type":"hierarchy","question":question,"program":program,"answer":answer})

        # Template: Is there any {part_1} in the {part_2}

        # Counting Questions
        # Template: How many {part} are there in the scene
        # program:  count(filter(scene(), part))
        # answers:  number of specific part in the scene
        num_counting = 5
        for n in range(num_counting):
            part = np.random.choice(scene_labels)
            question = "How many {} are there in the scene".format(part)
            answer = 0
            for label in scene_labels:
                if part == label:answer += 1
            program = "count(filter(scene(),{}))".format(part)
            qa_pairs.append({"type":"counting","question":question,"program":program,"answer":answer})
    else:
        qa_pairs,depth = gen_full_grounding(scene_tree,"full")
    

    questions_answers = {"all":qa_pairs,"depth":depth}

    return {"point_cloud":pc, "rgbs":None,"scene_tree":scene_tree,\
        "questions_answers":questions_answers}

def gen_full_grounding(test_tree, mode = "full"):
    all_labels = ["pot","body","container","containing_things","liquid_or_soil",
                "plant","other","lid","base","foot_base","foot"]
    test_dataset = {}
    nodes = [];depths = []
    sons = {}
    def dfs(node, depth = 0):
        nodes.append(node)
        depths.append(depth)
        depth += 1
        for edge in test_tree.edges:
            if edge[0] == node:
                if node not in sons:sons[node] = [edge[1]]     
                else:sons[node].append(edge[1]) 
                dfs(edge[1], depth = depth)
    dfs("root",0)
    D = max(depths)-1
    for d in range(D+1,0,-1):
        if mode == "full":
            test_data = []
            available_nodes = []
            top_nodes = []
            for i,node in enumerate(nodes):
                if depths[i] >= d:available_nodes.append(node)
                if depths[i] == d: top_nodes.append(node)

            # existence data
            for exist_node in available_nodes:
                test_data.append(
                    {"program:":"exist(filter(scene(),{}))".format(exist_node),
                    "answer":"yes","type":"existence","type":"hierarchy",
                    "question":"Is there any {} in the scene?".format(exist_node)})
            # hierarchy data
            for top_node in top_nodes:
                if top_node in sons:
                    for son in sons[top_node]:
                        test_data.append(
                        {"program":"exist(filter(subtree(filter(scene(),{})),{}))".format(top_node, son),
                         "answer":"yes","type":"hierarchy",
                         "question":"exist(filter(subtree(filter(scene(),{})),{}))".format(top_node,son)})
            # count
            for node in available_nodes:
                answer = "yes" if node in available_nodes else "no"
                test_data.append(
                    {"program:":"count(filter(scene(),{}))".format(node),
                    "answer":"1","type":"counting",
                    "question":"count(filter(scene(),{}))".format(node)})

            # negatory
            for node in all_labels:
                answer = "yes" if node in available_nodes else "no"
                test_data.append(
                    {"program:":"exist(filter(scene(),{}))".format(node),
                    "answer":answer,"type":"existence",
                    "question":"count(filter(scene(),{}))".format(node)})

            test_data.append(
                {"program":"count(scene())",
                "answer":str(len(available_nodes)),
                "question":"How many parts are there?","type":"count"}
            )

            test_dataset[str(D - d + 2)] = test_data
        if mode == "sample":
            test_dataset[str(D - d + 2)] = []

    return test_dataset, D+1

#outputs = generate_color(idx = 176)

def tree_contain(p1,p2,tree):pass

import argparse

genparser = argparse.ArgumentParser()
genparser.add_argument("--mode",                default = "Nope")
genparser.add_argument("--category",            default = "chair")
genparser.add_argument("--num_points",          default = 1000)
genargs = genparser.parse_args()

#assert genargs.mode in ["geo","full"],print(genargs.mode)
qadataset_dir = root + "/partnet_{}_qa/{}".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir): os.makedirs(qadataset_dir)

qadataset_dir_train = root + "/partnet_{}_qa/{}/train".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_train): os.makedirs(qadataset_dir_train)
qadataset_dir_train = root + "/partnet_{}_qa/{}/train/point_cloud".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_train): os.makedirs(qadataset_dir_train)
qadataset_dir_train = root + "/partnet_{}_qa/{}/train/qa".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_train): os.makedirs(qadataset_dir_train)
qadataset_dir_train = root + "/partnet_{}_qa/{}/train/annotations".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_train): os.makedirs(qadataset_dir_train)

qadataset_dir_test = root + "/partnet_{}_qa/{}/test".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_test): os.makedirs(qadataset_dir_test)
qadataset_dir_test = root + "/partnet_{}_qa/{}/test/point_cloud".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_test): os.makedirs(qadataset_dir_test)
qadataset_dir_test = root + "/partnet_{}_qa/{}/test/qa".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_test): os.makedirs(qadataset_dir_test)
qadataset_dir_test = root + "/partnet_{}_qa/{}/test/annotations".format(genargs.mode,genargs.category)
if not os.path.exists(qadataset_dir_test): os.makedirs(qadataset_dir_test)

print("Generating: Category:{} Mode: {}".format(genargs.category, genargs.mode))
if genargs.mode == "geo":
    train_split_path = root + "/partnethiergeo/{}_hier/train.txt".format(genargs.category)
    with open(train_split_path,"r") as train_split:
        for index in tqdm(train_split):
            index = int(index.strip())
            outputs = generate_structure(cat = genargs.category, idx = index)
            point_cloud =   outputs["point_cloud"]
            questions_answers   =   outputs["questions_answers"]
            scene_tree  =   outputs["scene_tree"]
            save_json(questions_answers,root + "/partnet_{}_qa/{}/train/qa/{}.json".\
                format(genargs.mode,genargs.category,index))

            np.save(root + "/partnet_{}_qa/{}/train/point_cloud/{}.npy".format(genargs.mode,genargs.category,index)\
                ,np.array(point_cloud))
            nx.write_gpickle(scene_tree,
            root + "/partnet_{}_qa/{}/train/annotations/{}.pickle".format(genargs.mode,genargs.category,index))
    
    test_split_path = root + "/partnethiergeo/{}_hier/test.txt".format(genargs.category)
    with open(test_split_path,"r") as test_split:
        for index in tqdm(test_split):
            index = int(index.strip())
            outputs = generate_structure(cat = genargs.category, idx = index)
            point_cloud =   outputs["point_cloud"]
            
            questions_answers   =   outputs["questions_answers"]
            save_json(questions_answers,root + "/partnet_{}_qa/{}/test/qa/{}.json".\
                format(genargs.mode,genargs.category,index))
            np.save(root + "/partnet_{}_qa/{}/test/point_cloud/{}.npy".format(genargs.mode,genargs.category,index)\
                ,np.array(point_cloud))
            nx.write_gpickle(scene_tree,
            root + "/partnet_{}_qa/{}/test/annotations/{}.pickle".format(genargs.mode,genargs.category,index))
    

if genargs.mode == "full":
    pass

print("Generation Completed: Category:{} Mode: {}".format(genargs.category, genargs.mode))

"""
"""

#outputs = generate_structure(cat = "vase", idx = 4167)
#outputs = generate_structure(cat = "table", idx = 18142)
#st = outputs["scene_tree"]
#nx.draw_networkx(st)
#plt.show()

#if outputs["rgbs"] is None: outputs["rgbs"] = .5 * torch.ones([outputs["point_cloud"].shape[0],3] )

#visualize_pointcloud([
#    (outputs["point_cloud"], outputs["rgbs"])
#])
#plt.show()