import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from matplotlib.patches import Rectangle

def visualize_image_grid(images, row, save_name = "image_grid"):
    plt.figure(save_name, frameon = False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    comps_grid = torchvision.utils.make_grid(images,normalize=True,nrow=row)
    
    plt.imshow(comps_grid.cpu().detach().numpy())
    plt.savefig("outputs/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)



def visualize_psg(gt_img, scene_tree, effective_level = 1,):
    scale = gt_img.shape[1]
    assert len(scene_tree) >= effective_level,print("Effective Level Larger than Scene Graph")
    for i in range(effective_level):
        level_idx = effective_level - i
        masks = scene_tree[effective_level]["masks"]

        visualize_scores(scene_tree[effective_level]["masks"][0:1].cpu().detach(),"{}".format(level_idx) )

    if effective_level == 1:
        masks = scene_tree[effective_level]["masks"]
        for j in range(masks.shape[1]):
            save_name = "mask_{}_{}".format(level_idx,j+1)
            plt.figure(save_name, frameon = False);plt.cla()
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

            match = scene_tree[effective_level]["match"][0].detach().numpy()
            centers = scene_tree[effective_level - 1]["centroids"][0].detach().numpy()
            moments = scene_tree[effective_level - 1]["moments"][0].detach().numpy()
            plt.imshow(gt_img[0])

            """
            for k in range(match.shape[0]):
                center = centers[k]
                moment = moments[k]
                match_score = float(match[k,j].cpu().detach().numpy())

                lower_x = center[0] * scale 
                lower_y = scale - center[1] * scale

                x_edge = moment[0] * scale /2
                y_edge = moment[1] * scale /2
            """

            plt.scatter(centers[:,0] * scale, scale -centers[:,1] * scale, alpha = match[:,j], color = "purple")

            """
                plt.gca().add_patch(Rectangle((lower_x - x_edge,lower_y - y_edge),2 * x_edge,2 *y_edge,
                    alpha = 1.0,
                    edgecolor='red',
                    facecolor='red',
                    lw=4))
            """
                
            plt.savefig("outputs/details/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)



def visualize_scene(gt_img, scene_tree, effective_level = 0):
    assert len(scene_tree) >= effective_level,print("Effective Level Larger than Scene Graph")
    for i in range(effective_level):
        level_idx = effective_level - i
        masks = scene_tree[effective_level]["masks"]

        visualize_scores(scene_tree[effective_level]["masks"][0:1].cpu().detach(),"{}".format(level_idx) )

def visualize_tree(gt_img,scene_tree,effective_level):
    """
    only visualize single image case!
    """
    assert len(scene_tree) >= effective_level,print("Effective Level Larger than Scene Graph")
    for i in range(effective_level):
        level_idx = effective_level - i
        masks = scene_tree[level_idx]["local_masks"]

        for j in range(masks.shape[1]):
            save_name = "mask_{}_{}".format(level_idx,j+1)
            plt.figure(save_name, frameon = False);plt.cla()
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

  
            plt.imshow(masks[0,j,...].cpu().detach().numpy(), cmap="bone")
            
            plt.savefig("outputs/details/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)
    
            save_name = "comp_{}_{}".format(level_idx,j+1)
            plt.figure(save_name, frameon = False);plt.cla()
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

            plt.imshow(gt_img[0]/255 * (masks[0,j,...].unsqueeze(-1).cpu().detach().numpy()) )
            plt.savefig("outputs/details/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)

            visualize_scores(scene_tree[effective_level]["masks"][0:1])
        

def visualize_outputs(image, outputs):

    full_recon = outputs["full_recons"]
    recons     = outputs["recons"]
    masks      = outputs["masks"]

    num_slots = recons.shape[1]
    
    # [Draw Components]
    plt.figure("Components",frameon=False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    comps_grid = torchvision.utils.make_grid((recons*masks).cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    
    plt.imshow(comps_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/components.png")

    # [Draw Masks]
    plt.figure("Masks",frameon=False);plt.cla()
    masks_grid = torchvision.utils.make_grid(masks.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(masks_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/masks.png")

    # [Draw Recons]
    plt.figure("Recons",frameon=False);plt.cla()
    recon_grid = torchvision.utils.make_grid(recons.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(recon_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/recons.png")

    # [Draw Full Recon]
    plt.figure("Full Recons",frameon=False);plt.cla()
    grid = torchvision.utils.make_grid(full_recon.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=1)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/full_recons.png")

    # [Draw GT Image]
    plt.figure("GT Image",frameon=False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    gt_grid = torchvision.utils.make_grid(image.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=1)
    plt.imshow(gt_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/gt_image.png")

def visualize_distribution(values):
    plt.figure("answer_distribution", frameon = False)
    plt.cla()
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = False)
    keys = list(range(len(values)))
    plt.bar(keys,values)

def visualize_scores(scores, name = "set"):
    batch_size = scores.shape[0]
    score_size = scores.shape[1]

    row = batch_size * score_size 
    col = row / 4

    plt.figure("scores", frameon = False, figsize = (row,col))
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = False, bottom = False)
    plt.cla()
    
    for i in range(batch_size):
        plt.subplot(1,batch_size,i + 1,frameon=False)
        plt.cla()

        
        keys = list(range(score_size))
        plt.bar(keys,scores[i])
        plt.tick_params(left = False, right = False , labelleft = True ,
                labelbottom = False, bottom = False)

    plt.savefig("outputs/scores_{}.png".format(name))

def answer_distribution_num(count, target, name = "answer_distribution"):
    batch_size = 1
    score_size = 4

    row = batch_size * score_size 
    col = row / 2

    plt.figure("dists",frameon = False, figsize = (row,col))
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = True)
    plt.cla()

    x = np.linspace(0,5,100)
    y = np.exp( 0 - (x-target) * (x-target) / 2)
    plt.plot(x,y)
    plt.scatter(target,1)
    plt.scatter(count,np.exp( 0 - (target-count) * (target-count) / 2))

    plt.savefig("outputs/{}.png".format(name))
    

def answer_distribution_binary(score, name = "answer_distribution"):
    batch_size = 1
    score_size = 4

    row = batch_size * score_size 
    col = row / 2

    scores = [score, 1 - score]

    plt.figure("dists", frameon = False, figsize = (row,col))
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = True)
    plt.cla()
    
    for i in range(batch_size):
        plt.subplot(1,batch_size,i + 1,frameon=False)
        plt.cla()

        
        keys = ["yes","no"]
        plt.bar(keys,scores)
        plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = True)

    plt.savefig("outputs/{}.png".format(name))

# From SRN utils, just formats a flattened image for image writing
def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)

# Takes the pred img and clusters produced and writes them to a TF writer
