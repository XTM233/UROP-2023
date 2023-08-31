MAX_TIME = 300 # in minutes
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os
import numpy as np
import csv
import pickle

# structure of working dir
# ./models/*
# ./antibody-tokenizer
# ./sabdab_heavy.txt
# ./this script
# ./assest_0/*.txt, generated
# ./*.pkl, generated
# ./*.csv, generated
def recover_splits(rand_ind, seqs):
    random_seed = rand_ind + 2023
    # recover test datasets
    split_ratio = [0.8,0.1,0.1]
    def write2file(st, name):
        with open(f"./assets_{rand_ind}/"+name+'.txt', 'w') as f:
            for line in st:
                f.write(line+"\n")
    print("spliting")
    N = len(seqs)
    np.random.seed(random_seed)
    split_indices = np.random.permutation(N)
    train_indices = split_indices[:int(N*split_ratio[0])]
    # print(train_indices)
    val_indices = split_indices[int(N*split_ratio[0]):int(N*((split_ratio[0])+split_ratio[1]))]
    test_indices = split_indices[int(N*((split_ratio[0])+split_ratio[1])):]
    seqs = np.array(seqs)
    train_seqs = seqs[train_indices]
    val_seqs = seqs[val_indices]
    test_seqs = seqs[test_indices]
    write2file(train_seqs, "train_small")
    write2file(val_seqs, "val_small")
    write2file(test_seqs, "test_small")

# gather all sequences from the dataset
def load_FASTA(filename):
    count = 0
    current_seq = ''
    all_seqs = []
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '>':
                all_seqs.append(current_seq)
                current_seq = ''
            else:
                current_seq+=line[:-1]
                count+=1
        all_seqs.append(current_seq)
        #all_seqs=np.array(map(lambda x: [aadict[y] for y in x],all_seqs[1:]),dtype=int,order="c")
    return all_seqs    

def my_function():
    # Initialise the tokeniser
    tokenizer = RobertaTokenizer.from_pretrained("antibody-tokenizer")

    # Initialise the data collator, which is necessary for batching
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # I aligned and prepared a set of ~3000 antibodies from SabDab
    name_fasta='sabdab_heavy.txt'
    seqs_al  =load_FASTA(name_fasta)[1:]
        
    seqs=[]
    for s in range(len(seqs_al)):
        seqs.append(''.join([seqs_al[s][i] for i in range(len(seqs_al[s])) if seqs_al[s][i]!='-']))

    # assume we already gets a dictionary of tuples, and their respective path
    rows = []
    model_lst = os.listdir("./models")
    for d in model_lst:
        # for each model
        model = RobertaForMaskedLM.from_pretrained("./models/"+d)
        lst = d.split("-")[1:]
        lst = [int(s) for s in lst]
        model_set = tuple(lst) # (n_head, n_layer, rand_ind)
        rand_ind = model_set[-1]
        recover_splits(rand_ind=rand_ind, seqs=seqs)
        text_datasets = {
            "train": [f'assets_{rand_ind}/train_small.txt'],
            "eval": [f'assets_{rand_ind}/val_small.txt'],
            "test": [f'assets_{rand_ind}/test_small.txt']
        }
        dataset = load_dataset("text", data_files=text_datasets)
        tokenized_dataset = dataset.map(
            lambda z: tokenizer(
                z["text"],
                padding="max_length",
                truncation=True,
                max_length=150,
                return_special_tokens_mask=True,
            ),
            batched=True,
            num_proc=1,
            remove_columns=["text"],
        )
        # use a temprorary trainer for evaluation
        trainer = Trainer(
            model=model,
            # args=args,
            data_collator=collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"])
        output = trainer.predict(tokenized_dataset["test"])
        with open('output'+d[5:]+'.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(output, file)
        test_loss = output[2]["test_loss"]
        row = list(model_set)+[test_loss]
        rows.append(row)

    with open('test_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n_head", "n_layer", "rand_ind", "test_loss"])
        for row in rows:
            writer.writerow(row)

# set time_out

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


try:
    with time_limit(MAX_TIME*60):
        my_function()
except TimeoutException as e:
    print("Timed out!")

print("Finished.")
