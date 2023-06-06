from datasets import *
from psgnet import *
import torch
import torch.nn as nn

dataset = DynamicSprite()

vid = dataset[0]["video"]
batch_size = 1

model = torch.load("checkpoints/constant_qtr_model.ckpt")

imsize = 64

while True:
    for model_input in vid:
        
        gt_img = torch.tensor(model_input).unsqueeze(0).float()
        
        outputs = model(gt_img)
        
        recons, clusters, all_losses = outputs["recons"],outputs["clusters"],outputs["losses"]

        plt.figure("overall")
        plt.imshow(recons[-1][0].detach().reshape([imsize,imsize,3]).clip(0.0,1.0))
        plt.pause(0.001)
        plt.cla()


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

            plt.subplot(2,len(levels),i+1 + len(levels))
            plt.tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
            plt.cla()
            plt.imshow(recons[i].reshape([imsize,imsize,3]).detach())
            render_level(level,"Namo",scale = imsize)
        plt.pause(0.001)
        plt.cla()

