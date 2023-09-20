import json

def save_json(data,path):
    '''input the diction data and save it'''
    beta_file = json.dumps(data)
    file = open(path,'w')
    file.write(beta_file)
    return True

all_questions = []

for i in range(400):
    questions_answer_pairs = []
    if i <= 199:
        answer = "yes"
        count = "one"
        count_obj = "1"
    else:
        answer = "no"
        count = "zero"
        count_obj = "zero"
    questions_answer_pairs.append(
                {
            "question":"is there any house in the scene",
            "program":"exist(filter(scene(),house))",
            "answer":answer
                }
            )
    questions_answer_pairs.append(
                {
            "question":"how many house in the scene",
            "program":"count(filter(scene(),house))",
            "answer":count
                }
            )
    questions_answer_pairs.append(
                {
            "question":"how many object in the scene",
            "program":"count(scene())",
            "answer":count_obj
                }
            )
    all_questions.append(questions_answer_pairs)

root = "/Users/melkor/Documents/datasets/"

save_json(all_questions, root + "acherus/" + "train_questions.json")