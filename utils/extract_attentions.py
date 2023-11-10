MAX_TIME = 300 # in minutes
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
# from datasets import load_dataset
import os
import torch
import pickle

def write2file(st, name):
    with open("./assets/"+name+'.txt', 'w') as f:
        for line in st:
            f.write(line+"\n")

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
    name_fasta='sabdab_heavy.txt' #NOTE
    seqs_al  =load_FASTA(name_fasta)[1:]

    # remove gaps in seqs_al
    seqs=[]
    for s in range(len(seqs_al)):
        seqs.append(''.join([seqs_al[s][i] for i in range(len(seqs_al[s])) if seqs_al[s][i]!='-']))
    N = len(seqs_al)
    model = RobertaForMaskedLM.from_pretrained("./models/model-4-6-1")
    attentions_list = []
    for i in range(N):
        token_ids = tokenizer.encode(seqs_al[0], return_tensors='pt')
        out = model(token_ids)
        # attentions = out.attentions[0][0][0] # 130x130 array
        attentions_list.append(out.attentions)
        print(i)
    with open('attentions_4-6-1.pkl', 'wb') as file:     
        # A new file will be created
        pickle.dump(attentions_list, file) 
    model = RobertaForMaskedLM.from_pretrained("./models/model-4-6-5")
    attentions_list = []
    for i in range(N):
        token_ids = tokenizer.encode(seqs_al[0], return_tensors='pt')
        out = model(token_ids)
        # attentions = out.attentions[0][0][0] # 130x130 array
        attentions_list.append(out.attentions)
        print(i)
    with open('attentions_4-6-5.pkl', 'wb') as file:     
        # A new file will be created
        pickle.dump(attentions_list, file) 
        

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