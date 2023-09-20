'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:01:54
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:33
 # @ Description: This file is distributed under the MIT license.
'''


import numpy as np
import matplotlib.pyplot as plt

import pygame
import os
import json


root = "/Users/melkor/Documents/datasets/"

# load json data
numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def num2word(i):
    assert i < 10
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return numbers[i]

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

def random_category():
    return np.random.choice(["tower","boat","house"])

def random_template():
    tp = np.random.randint(0,1)
    return ["how many {} are there?",
    "count(filter(scene(),boat))",
    "one"]

def random_color():
    color = [0, 10, 0]
    color[np.random.choice([0,2])] = 200
    return color

def random_coord(margin,resolution = (128,128)):
    px = np.random.randint(margin, resolution[0] - margin)
    py = np.random.randint(margin, resolution[1] - margin)
    return px,py

def generate_toy_dataset(num, resolution = (128,128), questions = False):
    # Import and initialize the pygame library

    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode(resolution)
    bg1 = pygame.image.load("/Users/melkor/Documents/datasets/bg.webp").convert()
    bg2 = pygame.image.load("/Users/melkor/Documents/datasets/bg2.webp").convert()
    bg1 = bg2
    # Run until the user asks to quit
    running = True
    itr = 0
    all_questions = []
    while running:
        itr += 1

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if itr > num - 1: running = False

        # Fill the background with white
        screen.fill((255, 255, 255))
        background_image = np.random.choice([bg1,bg2])
        screen.blit(background_image, [0, 0])
        scene = []

        for _ in range(np.random.choice([0,1,2,3])):
            scale = np.random.randint(resolution[0]/12,resolution[0] / 9)
            # choose the color to draw
            color = random_color()
            px,py = random_coord(scale, resolution)

            # control the portion of different kind of objects generated
            category = np.random.choice([0,1,2], p = [0.2, 0.4, 0.4])
            
            if category == 0:
                # draw tower
                top_color = random_color()
                top_cat = np.random.choice([0,1], p = [0.2,0.8])
                tower_size = scale / 1.2 #np.random.randint(resolution[0]/12,resolution[0]/9)
                if top_cat == 0:
                    # draw the triangle
                    tri_pos = [[px,py],[px+tower_size,py],[px+tower_size/2,py-tower_size/2]]
                    pygame.draw.polygon(screen, top_color, tri_pos)
                if top_cat == 1:
                    pygame.draw.circle(screen, top_color, (px+tower_size/2, py-tower_size/2), tower_size / 2)
                pygame.draw.rect(screen, color, (px,py,tower_size,tower_size*1.5))
                scene.append(["tower", color])
                
            if category == 1:
                # draw boat
                top_color = random_color()
                tri_pos = [[px + scale/3, py],[px+scale/2 * 2.0,py - scale],[px+scale/3,py-scale*2]]
                pygame.draw.polygon(screen, top_color, tri_pos)
                pygame.draw.rect(screen, color, (px, py, scale, scale/2))
                scene.append(["boat", top_color])
                
            if category == 2:
                # draw a house
                top_color = random_color()
                scale *= 2
                margin = np.random.randint(0.2 * scale, 0.3 * scale)
                tri_pos = [[px,py],[px+scale,py],[px+scale/2,py-scale/2]]
                pygame.draw.polygon(screen, top_color, tri_pos)
                pygame.draw.rect(screen, color, (px + margin, py, scale - 2 * margin, scale - 2 * margin))
                scene.append(["house",color])

        # Flip the display
        pygame.display.flip()

        # generate the image of the scene produced 
        pygame.image.save(screen, "{}{}{}.png".format(root,"toy/images/",itr))

        # generate questions and programs for the scene
        questions_answer_pairs = []

        # single count
        # double count
        # single filteration
        # double filteration
        
        for i in range(3):
            # generate three concepts to perform 1st order filteration
            test_category = random_category()
            flag = False
            for bind in scene:
                if not flag and bind[0] == test_category:flag = True
            gt_ans = "yes" if flag else "no"

            template = ["is there any {} are there?".format(test_category),
            "exist(filter(scene(),{}))".format(test_category),
            gt_ans]

            questions_answer_pairs.append(
                {
            "question":template[0],
            "program":template[1],
            "answer":template[2]
                }
            )
        for i in range(2):
            # generate two conecpts to perform 
            count_category = random_category()
            count = 0
            for bind in scene:
                if  bind[0] == count_category:count += 1
            gt_ans = numbers[count]

            template = ["how many {} are there?".format(count_category),
            "count(filter(scene(),{}))".format(count_category),
            gt_ans]

            questions_answer_pairs.append(
                {
            "question":template[0],
            "program":template[1],
            "answer":template[2]
                }
            )

        template = ["how many objects are there?",
            "count(scene())",
            num2word(len(scene))]

        questions_answer_pairs.append(
                {
            "question":template[0],
            "program":template[1],
            "answer":template[2]
                }
            )

        all_questions.append(questions_answer_pairs)

    # Done! Time to quit.
    save_json(all_questions, root + "toy/" + "train_questions.json")
    pygame.quit()
    

if __name__ == "__main__":
    generate_toy_dataset(3500, [256,256], True)