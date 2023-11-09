# this is a transition script between each round of training, which does the following
# 1. check what model finished training, append to finished.txt and move folders into finished/
# 2. rename folders of experiments to be continued in the next round
# 3. replenish job array with remaining set of hyperparameter, save experiment specifiations in round_n.txt

import numpy as np
import pandas as pd
import os
import shutil

# construct some useful functions

def txt2list(filepath):
    """
    load text files, with each line as an element of list
    Args:
        filepath: a string indicating the path of a text file with each line is a literal tuple
    Outputs:
        lst: a list of tuple
    """
    lst = []
    f = open(filepath, "r")
    Lines = f.readlines()
    for line in Lines:
        lst.append(eval(line))
    return lst

def list2txt(lst,filepath):
    """
    save content of a list into text file
    Args:
        lst: a list of tuples
        filepath: a string indicating the path of a text file
    """
    # clear file content before writing
    if os.path.isfile(filepath):
        open(filepath, 'w').close()
    with open(filepath, 'w') as f:
        for line in lst:
            f.write(str(line)+"\n")

# define a whole set of hyperparameters to run experiments on

whole_set = []
for n_head in range(1,5):
    for n_layer in range(1,5):
        for random_ind in range(10):
            whole_set.append((n_head, n_layer, random_ind))

whole_set = set(whole_set)

# when running the script for the first time
if not os.path.isfile("finished.txt"):
    # train_finished = [(1,1,0),(1,1,1),(1, 3, 1),(2, 1, 0),(2, 1, 1),(2, 3, 0),(3, 1, 1),(3, 3, 0),(3, 4, 0),(4, 1, 1),(4, 3, 0)]
    train_finished = []
    list2txt(train_finished, "finished.txt")
else:
    train_finished = txt2list("finished.txt")

# check which round is it
# if not os.path.isfile("round_1.txt"):
#     lst = [(1, 1, 2),(1, 1, 3),(1, 2, 0),(1, 2, 1), (1, 3, 0), (1, 1, 4), (1,4, 0),(1, 4, 1),(1, 1, 5),(1, 1, 6),(2, 2, 0),(2, 2, 1),(1, 1, 7),(2, 3, 1),(2, 4, 0),(2, 4, 1),(3,1,0),(1, 1, 8),(3,2,0),(3, 2, 1),(1, 1, 9),(3, 3, 1),(1, 2, 2),(3, 4, 1),(4,1,0),(1, 2, 3),(4, 2, 0),(4, 2, 1), (1, 2, 4),(4, 3, 1),(4, 4, 0),(4, 4, 1)]
#     list2txt(lst, "round_1.txt")
# load models trained in the last round
latest_round = 0
res = []
# Iterate directory to find record of previous rounds
for file in os.listdir("./"):
    # check only text files
    if file.endswith('.txt') and file.startswith("round_"):
        res.append(file)
# when running helper script for the first time, rename train folders
if len(res) == 0:
    current_train_array = []
    for i in range(1,5):
        for j in range(1,5):
            for k in range(2):
                current_train_array.append((i,j,k))
    for i in range(32):
        train_path = f"./train-{i+1}"
        new_train_path = f"./train-1-{i+1}"
        os.rename(train_path, new_train_path)
# otherwise, load from the existing train_array
else:
    current_train_array = np.zeros(32)
    current_train_array = list(current_train_array)
    for s in res:
        lst = s.split("round_")
        lst = lst[1].split(".txt")
        round_n = int(lst[0])
        if round_n > latest_round:
            latest_round = round_n
    last_train_array = txt2list(f'round_{latest_round}.txt')
    # for all files starting with train-, check if model folder is empty
    for i in range(32):
        train_path = f"./train-{latest_round}-{i+1}"
        if len(os.listdir(train_path+"/model")) != 0:
            # if best model is saved, meaning training has been finished
            train_finished.append(last_train_array[i])
            shutil.move(train_path, "./finished/"+train_path)
            n_head, n_layer, rand_ind = last_train_array[i]
            new_train_path = f"./train-{n_head}-{n_layer}-{rand_ind}"
            os.rename("./finished/"+train_path,"./finished/"+new_train_path)
        else:
            current_train_array[i] = last_train_array[i]
            new_train_path =  f"./train-{latest_round+1}-{i+1}"
            os.rename(train_path, new_train_path)
    
    remaining = whole_set - set(train_finished) - set(last_train_array)
    remaining_lst = list(remaining)
    counter = 0
    for i in range(32):
        if current_train_array[i] == 0:
            try:
                current_train_array[i] = remaining_lst[counter]
                counter += 1
            except IndexError:
                print("Run out of sets of hyperparameters")
                break

list2txt(current_train_array, f"round_{latest_round+1}.txt")
list2txt(train_finished,"finished.txt")

# update index number, indicating which round is it
if os.path.isfile("index.txt"):
    open("index.txt", 'w').close()
with open("index.txt", 'w') as f:
    f.write(str(latest_round+1))

# not used in this script, helpful for inspection of checkpoints

def generate_path(i, j, k):
    # return path of the latest checkpoint json file for specified model
    ind = (i-1)*8+(j-1)*2+k
    s = "train-"+str(ind)
    lst = os.listdir(s+"/test")
    path = "/".join([s,"test",lst[-1],"trainer_state.json"])
    return path

# transform json files into dataframe
def read_log(filepath):
    # return a dataframe with columns "step", "loss", "eval_loss"
    js = pd.read_json(filepath)
    s = js["best_model_checkpoint"][0] # the directory of best checkpoint
    best_model_checkpoint = int(s.split("-")[-1]) # the number of steps at which the best metric is obtained
    lst = filepath.split("checkpoint-")
    lst = lst[1].split("/")
    total_steps = int(lst[0])
    df = pd.json_normalize(js["log_history"])
    n = len(df)
    # dataframe contianing eval loss
    odd_indx = np.array([i for i in range(n) if i%2 == 1])
    df1 = df.loc[odd_indx,:]
    # dataframe contianing train loss
    even_indx = np.array([i for i in range(n) if i%2 == 0])
    df2 = df.loc[even_indx,:]

    df1 = df1[["step", "eval_loss"]]
    df2 = df2[["step", "loss"]]
    df3 = df1.join(df2.set_index('step'), on='step')
    return df3,js["best_metric"][0],best_model_checkpoint, total_steps

def earlystop_detection(eval_loss, patience=10,tolarance=1e-5):
    # eval_loss: array of eval loss each eval step
    # n: integer, early stopping patience
    # lag_three = eval_loss[:-n]- eval_loss[n:] #should be positive if training continuously improves loss
    # lag_three = lag_three.reshape(-1)
    best_loss = eval_loss[0]
    counter = 0
    for i in range(1,len(eval_loss)):
        if best_loss - eval_loss[i] > tolarance:
            best_loss = eval_loss[i]
            # reset counter if validation loss improves
            counter = 0
        elif best_loss - eval_loss[i] < tolarance:
            counter += 1
        if counter >= patience:
            return True
