from config import *
from model import *
from datasets import *

demoparser = argparse.ArgumentParser()
demoparser.add_argument("--demo_name",          default = "knowledge")
demoparser.add_argument("--concept_dim",        default = 2)
demoparser.add_argument("--demo_epochs",        default = 113200)

democonfig = demoparser.parse_args()
config.concept_dim = democonfig.concept_dim

if democonfig.demo_name == "knowledge":
    executor = SceneProgramExecutor(config)
    p = "exist(filter(scene(),red))"
    q = executor.parse(p)
    EPS = 1e-4

    features = torch.tensor([
        [0.4,-0.3,EPS,EPS],
        [0.3,0.3,EPS,EPS],
    ])

    kwargs = {"end":torch.ones(2),
             "features":features}
    o = executor(q, **kwargs)
    print(o["end"])

    params = executor.parameters()
    optimizer = torch.optim.Adam(params, lr = 1e-2)

    statements_answers = [
        ("exist(filter(scene(),red))","yes"),
        ("exist(filter(scene(),green))","yes"),

        ("exist(filter(filter(scene(),green),red))","no"),
        #("exist(filter(filter(scene(),cyan),cone))","no"),
    ]

    if democonfig.concept_dim == 2:
        plt.figure("visualize knowledge", figsize=(6,6))
        plt.xlim(-0.5,0.5)
        plt.ylim(-0.5, 0.5)
        for epoch in range(democonfig.demo_epochs):
            concept_keys,concept_embs = executor.all_embeddings()
            concept_embs = [t.cpu().detach() for t in concept_embs]

            # [Visualize Boxes]
            plt.cla()
            plt.xlim(-0.5,0.5)
            plt.ylim(-0.5, 0.5)
            for i, concept in enumerate(concept_keys):
                center, offset = concept_embs[i][0][:2], concept_embs[i][0][2:]
                plt.text(center[0], center[1], concept)

                # [Create a Patch]
                corner = center - offset
                top = center + offset
                plt.plot([
                        corner[0],top[0],top[0],corner[0],corner[0]
                    ],[
                        corner[1],corner[1],top[1],top[1],corner[1]
                    ], color="red")
                #plt.plot([corner[0],top[0]],[corner[1],top[1]])
            for feat in features:
                plt.scatter(feat[0],feat[1],color="blue")
                
            # [Calculate Regular]
            language_loss = 0.0
            for pair in statements_answers:
                p,a = pair
                q = executor.parse(p)
                o = executor(q, **kwargs)

                if a in ["yes","no"]:
                    if a == "yes":
                        language_loss -= o["end"]
                        print("p:",torch.sigmoid(o["end"]))
                    else:
                        language_loss += torch.sigmoid(o["end"])
                        print("p:",1 - torch.sigmoid(o["end"]))

            optimizer.zero_grad()
            language_loss.backward()
            optimizer.step()
            plt.pause(0.01)