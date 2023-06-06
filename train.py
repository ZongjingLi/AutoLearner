# This file is somewhat hackish for now, will refactor later

import os,sys


import numpy as np
from skimage import color
import numpy as np
import tensorflow as tf # just for DeepMind's dataset
import torch

import torchvision
import torch_sparse
import torch_scatter
import torch_sparse
import torch_geometric
from torch_geometric.utils import to_dense_batch

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from psgnet import *
import datasets

tf_records_path = '/Users/melkor/Documents/datasets/objects_room_train.tfrecords'

batch_size = 1
imsize     = 128

model_name = "toy128_level3"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create train dataloader 


dataset = datasets.Clevr4()
dataset = datasets.BattlecodeImageData()
dataset = datasets.SpriteData()
dataset = datasets.MixSpriteData()

dataset1 = datasets.MineClip()
dataset2 = datasets.MineOut()
dataset3 = datasets.MineCrazy()


dataset = torch.utils.data.ConcatDataset([dataset1,dataset2,dataset3])
#dataset = datasets.SpriteData()
dataset = datasets.ToyData("train")
#dataset = datasets.PTRData("train")
train_dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = True)

#dataset = datasets.dataset(tf_records_path, 'train')
#train_dataloader = dataset.batch(batch_size)

# Should move these two functions below to another file

# From SRN utils, just formats a flattened image for image writing
def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)

# Takes the pred img and clusters produced and writes them to a TF writer

def log_imgs(pred_img,clusters,gt_img,iter_):
    
    # Write grid of output vs gt 
    grid = torchvision.utils.make_grid(
                          lin2img(torch.cat((pred_img.cpu(),gt_img.cpu()))),
                          normalize=True,nrow=batch_size)

    # Write grid of image clusters through layers
    cluster_imgs = []
    for i,(cluster,_) in enumerate(clusters):
        for cluster_j,_ in reversed(clusters[:i+1]): cluster = cluster[cluster_j]
        pix_2_cluster = to_dense_batch(cluster,clusters[0][1])[0]
        cluster_2_rgb = torch.tensor(color.label2rgb(
                    pix_2_cluster.detach().cpu().numpy().reshape(-1,imsize,imsize) 
                                    ))
        cluster_imgs.append(cluster_2_rgb)
    cluster_imgs = torch.cat(cluster_imgs)
    grid2=torchvision.utils.make_grid(cluster_imgs.permute(0,3,1,2),nrow=batch_size)
    writer.add_image("Clusters",grid2.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT",grid.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT Var",grid.detach().numpy(),iter_)
    

from torch_sparse import SparseTensor
from torch_scatter import scatter_max



# Create model

try:
    model = torch.load("checkpoints/{}.ckpt".format(model_name),map_location = device)
    print("QTR Model Loaded from ckpt")
    #model = PSGNet(imsize)
except:
    print("checkpoint is not found, creating a new instance")
    model = PSGNet(imsize)

model = model.to(device)
model.device = device

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
 
# Logs/checkpoint paths

logging_root = "./logs"
ckpt_dir     = os.path.join(logging_root, 'checkpoints')
events_dir   = os.path.join(logging_root, 'events')
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
if not os.path.exists(events_dir): os.makedirs(events_dir)

checkpoint_path = None
if checkpoint_path is not None:
    print("Loading model from %s" % checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    
writer = SummaryWriter(events_dir)
iter_  = 0
epoch  = 0
steps_til_ckpt = 1000

# Training loop

im_loss_weight = 100
loss_history = []

#writer.add_graph(model,torch.randn(2,imsize,imsize,3).float())

while True:
    for model_input in train_dataloader:
        
          gt_img = torch.tensor(model_input["image"].numpy()).float().to(device)/255


          optimizer.zero_grad()

        #try:
          outputs = model(gt_img)

          recons, clusters, all_losses = outputs["recons"],outputs["clusters"],outputs["losses"]

          img_loss = 0 

          for i,pred_img in enumerate(recons[:]):
              img_loss += torch.nn.functional.l1_loss(pred_img.flatten(), gt_img.flatten())

          pred_img = recons[-1]

          all_losses.append({"img_loss" : im_loss_weight*img_loss})

          total_loss = 0
          for i,losses in enumerate(all_losses):
              for loss_name,loss in losses.items():
                  total_loss += loss
                  writer.add_scalar(str(i)+loss_name, loss, iter_)
          writer.add_scalar("total_loss", total_loss, iter_)

          total_loss.backward()
        
          if iter_ % 10 == 0:
              torch.save(model,"checkpoints/{}.ckpt".format(model_name))
              log_imgs(pred_img.cpu().detach(), clusters, gt_img.reshape([batch_size,imsize ** 2,3]).cpu().detach(),iter_)



              plt.figure("Level Recons vs GT Img")
              recon_num = len(recons)
              for i in range(recon_num):
                  recon_batch = recons[i].reshape([batch_size,imsize,imsize,3]).detach()

                  for j in range(batch_size):

                      plt.subplot(recon_num+ 1,batch_size,j + 1 + i * batch_size)
                      plt.tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
                      plt.imshow(recon_batch[j].cpu().detach())
              gt_batch = gt_img.reshape(batch_size,imsize,imsize,3)
              for k in range(batch_size):
                  plt.subplot(len(recons) + 1,batch_size, 1 + k + (len(recons)) * batch_size)
                  plt.tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)
                  plt.imshow(gt_batch[k].cpu().detach())
              plt.pause(0.001)
              plt.figure("Connected")

              levels = outputs["levels"]
              for i,level in enumerate(levels):
                    plt.subplot(2,len(levels),i + 1)
                    plt.tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
                    plt.cla()
                    plt.imshow(level.clusters.reshape([imsize,imsize]).detach())
                    #render_level(level,"Namo",scale = imsize)

                    plt.subplot(2,len(levels),i+1 + len(levels))
                    plt.tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
                    plt.cla()
                    plt.imshow(recons[i].reshape([imsize,imsize,3]).detach())
                    render_level(level,"Namo",scale = imsize)
                    
              plt.pause(0.0001)

          optimizer.step()
          sys.stdout.write("\rIter %07d Epoch %03d   L_img %0.4f " %
                          (iter_, epoch, img_loss))

          iter_ += 1
        

    epoch += 1