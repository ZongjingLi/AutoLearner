import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root",               default = "")
parser.add_argument("--mode",               default = "geo")
parser.add_argument("--category",           default = "vase")
args = parser.parse_args()

def build_labels(h,voc):
    if h["label"] not in voc: voc.append(h["label"])
    if "children" in h:
        for child in h["children"]:
            build_labels(child, voc)

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
        return data

if args.mode == "geo":
    root = "/Users/melkor/Documents/GitHub/Hierarchical-Learner"
    data_root = "/Users/melkor/Documents/datasets"
    category = args.category
    train_split_path = data_root + "/partnethiergeo/{}_hier/train.txt".format(args.category)
    label_vocab = []
    with open(train_split_path,"r") as train_split:
        for index in tqdm(train_split):
            index = int(index.strip())
    
            hier_path = data_root + "/partnethiergeo/{}_hier/{}.json".format(category, index)
            hier_data = load_json(hier_path)

            build_labels(hier_data,label_vocab)
    with open(root + "/knowledge/structure_{}_concept_vocab.txt".format(category),"w") as knowledge_file:
        for label in label_vocab:
            knowledge_file.write(label+"\n")