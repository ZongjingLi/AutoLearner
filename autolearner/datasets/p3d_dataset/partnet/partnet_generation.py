import torch
import torch.nn as nn

import numpy as np

category = []
templates = []

default_nums = {"query":1,"count":1}
colors = ["red","blue","green","yellow"]
categories = []

test_case_scene_tree = {"start":{"category":"frame","color":"red"}}

def generate_question_answer_pairs(scene_tree, num_questions = "All"):
    question_answer_pairs = []
    if num_questions in ["all","All"]:
        for cat in default_nums:
            num_questions = default_nums[cat]
            for i in range(num_questions):
                pass
    if isinstance(num_questions,int):
        for i in range(num_questions):
            cat = np.random.choice(category)

    print(cat)
    return question_answer_pairs