from mifafa import *
from config import *

import datasets
from tqdm import tqdm

psgnet = Mifafa(config)
psgnet = torch.load("checkpoints/qtrmc128.ckpt")
dataset = datasets.SpriteQA()
trainloader = torch.utils.data.DataLoader(dataset,batch_size = 1,shuffle = True)
optimizer = torch.optim.Adam(psgnet.parameters(), lr = 2e-4)

history = []

while True:
    total_loss = 0
    acc_count = 0
    total_count = 0
    for sample in tqdm(trainloader):
        
        working_loss = 0
        optimizer.zero_grad()

        ims = sample["image"] 
        programs = sample["program"]
        answers  = sample["answer"]

        outputs = psgnet.scene_perception(ims) # parser the image and other things

        working_programs = [p[0] for p in programs]
        working_answers  = [a[0] for a in answers]

        ground_answers = psgnet.joint_reason(working_programs,cast = False)

        recons = outputs["recons"];all_losses = outputs["losses"]
        for i,pred_img in enumerate(recons):
              working_loss += torch.nn.functional.l1_loss(pred_img.flatten(), ims.flatten())
        for i,losses in enumerate(all_losses):
              for loss_name,loss in losses.items():
                  working_loss += loss
        
        best_answer = [regular_result(t) for t in ground_answers]
        for i,term in enumerate(ground_answers):
            if len(term)>1:
                gt_answer = working_answers[i]
            
                key_answer = term["outputs"]
                pdf_answer = term["scores"]
 
                if gt_answer == best_answer[i]:acc_count += 1
                total_count += 1
                working_loss -= pdf_answer[key_answer.index(gt_answer)]

        working_loss.backward()
        optimizer.step()
        total_loss += working_loss.detach()
        #sys.stdout.write("\rLoss {} " .format(working_loss.detach()))
        
    torch.save(psgnet,"mk.ckpt")
    history.append(total_loss)
    print(1.0 * acc_count/total_count,total_loss)
    plt.cla()
    plt.plot(history)
    plt.pause(0.001)
