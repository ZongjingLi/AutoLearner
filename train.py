import warnings
warnings.filterwarnings("ignore")

import torch
import argparse 
import datetime
import time
import sys

from datasets import *

from config import *
from models import *
from visualize.answer_distribution import *

from torch.utils.tensorboard import SummaryWriter
import torchvision
from skimage import color

from legacy import *


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def log_imgs(imsize,pred_img,clusters,gt_img,writer,iter_):

    batch_size = pred_img.shape[0]
    
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

    visualize_image_grid(cluster_imgs[batch_size,...], row = 1, save_name = "val_cluster")
    visualize_image_grid(pred_img.reshape(batch_size,imsize,imsize,3)[0,...], row = 1, save_name = "val_recon")



def train_Valkyr(train_model, config, args):

    query = True if args.training_mode in ["joint", "query"] else False
    print("\nstart the experiment: {} query:[{}]".format(args.name,query))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    
    #[setup the training and validation dataset]
    if args.dataset == "ptr":
        train_dataset = PTRData("train", resolution = config.resolution)
        val_dataset =  PTRData("val", resolution = config.resolution)
    if args.dataset == "toy":
        if query:
            train_dataset = ToyDataWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = ToyData("train", resolution = config.resolution)
    if args.dataset == "Acherus":
        if query:
            print("Elbon Blade Crusade for You")
            train_dataset = AcherusDataset("train")
        else:
            train_dataset = AcherusImageDataset("train")
    if args.training_mode == "query":
        freeze_parameters(train_model.scene_perception.backbone)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle)

    # [joint training of perception and language]
    alpha = args.alpha
    beta  = args.beta
    if args.training_mode == "query":alpha = 0
    if args.training_mode == "perception":beta = 0
    

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)

    for epoch in range(args.epoch):
        epoch_loss = 0
        for sample in dataloader:
            
            # [perception module training]
            gt_ims = torch.tensor(sample["image"].numpy()).float().to(config.device)

            outputs = model.scene_perception(gt_ims)

            # get the components
            recons, clusters, all_losses = outputs["recons"],outputs["clusters"],outputs["losses"]
            
            masks    = outputs["abstract_scene"][-1]["masks"].permute([0,3,1,2]).unsqueeze(-1)
            scores   = outputs["abstract_scene"][-1]["scores"][0,...] - EPS
            scores   = torch.clamp(scores, min = EPS, max = 1)
            #print(scores)

            perception_loss = 0


            for i,pred_img in enumerate(recons[:]):
                perception_loss += torch.nn.functional.l1_loss(pred_img.flatten(), gt_ims.flatten())

            # [language query module training]
            language_loss = 0

            if query:
                for question in sample["question"]:
                    for b in range(len(question["program"])):
                        program = question["program"][b] # string program
                        answer  = question["answer"][b]  # string answer

                        abstract_scene  = outputs["abstract_scene"]
                        top_level_scene = abstract_scene[-1]

                        working_scene = [top_level_scene]

                        scores   = top_level_scene["scores"][b,...] - EPS
                        scores   = torch.clamp(scores, min = EPS, max = 1).reshape([-1])

                        #scores = scores.unsqueeze(0)

                        features = top_level_scene["features"][b].reshape([scores.shape[0],-1])


                        edge = 1e-6
                        if config.concept_type == "box":
                            features = torch.cat([features,edge * torch.ones(features.shape)],-1)#.unsqueeze(0)

                        kwargs = {"features":features,
                                  "end":scores }

                        q = train_model.executor.parse(program)
                        
                        o = train_model.executor(q, **kwargs)
                        #print("Batch:{}".format(b),q,o["end"],answer)
                        
                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            language_loss += F.mse_loss(int_num + 1,o["end"])
                            if itrs % args.checkpoint_itrs == 0:
                                print(q,answer)
                                visualize_scores(scores.reshape([args.batch_size,-1,1]).detach())
                                answer_distribution_num(o["end"].cpu().detach().numpy(),1+int_num.cpu().detach().numpy())
                        if answer in yes_or_no:
                            if answer == "yes":language_loss -= torch.log(torch.sigmoid(o["end"]))
                            else:language_loss -= torch.log(1 - torch.sigmoid(o["end"]))
                            if itrs % args.checkpoint_itrs == 0:
                                print(q,answer)
                                print(torch.sigmoid(o["end"]).cpu().detach().numpy())
                                visualize_scores(scores.reshape([args.batch_size,-1,1]).detach())
                                answer_distribution_binary(torch.sigmoid(o["end"]).cpu().detach().numpy())
            # [calculate the working loss]
            working_loss = perception_loss * alpha + language_loss * beta
            epoch_loss += working_loss.detach().numpy()

            # [backprop and optimize parameters]
            for i,losses in enumerate(all_losses):
                for loss_name,loss in losses.items():
                    writer.add_scalar(str(i)+loss_name, loss, itrs)

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            writer.add_scalar("working_loss", working_loss, itrs)
            writer.add_scalar("perception_loss", perception_loss, itrs)
            writer.add_scalar("language_loss", language_loss, itrs)

            if not(itrs % args.checkpoint_itrs):
                name = args.name
                expr = args.training_mode
                num_slots = masks.shape[1]
                torch.save(train_model, "checkpoints/{}_{}_{}_{}.ckpt".format(name,expr,config.domain,config.perception))
                log_imgs(config.imsize,pred_img.cpu().detach(), clusters, gt_ims.reshape([args.batch_size,config.imsize ** 2,3]).cpu().detach(),writer,itrs)
                
                visualize_image_grid(gt_ims.flatten(start_dim = 0, end_dim = 1).cpu().detach(), row = args.batch_size, save_name = "ptr_gt_perception")
                visualize_image_grid(gt_ims[0].cpu().detach(), row = 1, save_name = "val_gt_image")

                
                single_comps =  torchvision.utils.make_grid((masks*gt_ims)[0:1].cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots).permute(1,2,0)
                visualize_image_grid(single_comps.cpu().detach(), row = 1, save_name = "slot_masks")
                #visualize_psg(gt_ims[0:1].cpu().detach(), outputs["abstract_scene"], args.effective_level)

            itrs += 1

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Percept:{} Language:{}, Time: {}".format(epoch + 1, itrs, working_loss,perception_loss,language_loss,datetime.timedelta(seconds=time.time() - start)))
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
    print("\n\nExperiment {} : Training Completed.".format(args.name))


argparser = argparse.ArgumentParser()
# [general config of the training]
argparser.add_argument("--device",                  default = config.device)
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--epoch",                   default = 400)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 2e-4)
argparser.add_argument("--batch_size",              default = 1)
argparser.add_argument("--dataset",                 default = "toy")

# [perception and language grounding training]
argparser.add_argument("--training_mode",           default = "joint")
argparser.add_argument("--alpha",                   default = 10.00)
argparser.add_argument("--beta",                    default = 0.001)

# [additional training details]
argparser.add_argument("--warmup",                  default = True)
argparser.add_argument("--warmup_steps",            default = 100)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [curriculum training details]
argparser.add_argument("--effective_level",         default = 1)

# [checkpoint location and savings]
argparser.add_argument("--checkpoint_dir",          default = False)
argparser.add_argument("--checkpoint_itrs",         default = 50)
argparser.add_argument("--pretrain_perception",     default = False)

args = argparser.parse_args()

if args.checkpoint_dir:
    model = torch.load(args.checkpoint_dir, map_location = config.device)
else:
    print("No checkpoint to load and creating a new model instance")
    if args.name == "TBC":
        config.perception = "slot_attention"
        model = SceneLearner(config)
    if args.name == "Acherus":
        config.perception = "psgnet"
        model = SceneLearner(config)
    if args.name == "Elbon" or "PTR":
        config.domain = "toy"
        print("Elbon Blade Training Set")
        config.perception = "psgnet"
        args.dataset = "Elbon"
        model = SceneLearner(config)
    if args.name == "Valkyr":
        config.perception = "local_psgnet"
        args.dataset = "Elbon"
        print("Init the Local Scene Graph Net")
        model = SceneLearner(config)

if args.pretrain_perception:
    model.scene_perception = torch.load(args.pretrain_perception, map_location = config.device)

#model = torch.load("checkpoints/TBC_joint_toy_slot_attention.ckpt", map_location=args.device)
#model.scene_perception.slot_attention.iters = 10

if args.name == "Valkyr":
    print("Val'kyr start the training session.")
    args.dataset = "Acherus"
    train_Valkyr(model, config, args)

if args.name == "TBC":
    train_TBC(model, config, args)
elif args.name == "Acherus":
    print("Assault on New Avalon")
    config.perception = "psgnet"
    train_Archerus(model, config, args)
elif args.name == "Elbon" or "PTR":
    print("The Elbon Blade")
    args.dataset = "toy"
    if args.name == "PTR":
        args.dataset = "ptr"
    config.perception = "psgnet"
    train_Archerus(model, config, args)

