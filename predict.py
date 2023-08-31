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
    prob_dict = {}
    count = 0
    for d in os.listdir("./models/"): # NOTE
        model_dir = "./models/"+d #NOTE
        model_set = d.split("-")
        n_layer = int(model_set[1]) #NOTE
        n_head = int(model_set[2])
        rand_ind = int(model_set[3])
        model = RobertaForMaskedLM.from_pretrained(model_dir)
        probs = []
        
        for s in range(N):
            token_ids = tokenizer.encode(seqs_al[s], return_tensors='pt') # to maintain a homogeneous shape
            #train
            # take the model and calculate the MaskedLMOutput, take its logits
            # which are in a matrix with size # tokens x # symbols 
            ## (note: it's a single-sequence embedding)
            out = model(token_ids)
            mat = out['logits'].squeeze()
            
            ## calculate attention of each token, which is a probability conditional on the rest of the sequence
            ## to obtain a probability, one normalizes over the column (possible symbols at that position)
            ## and flatten everything into a vector of length # tokens x # symbols
            prob = torch.nn.functional.softmax(mat, dim=1).detach().numpy()
            probs.append(prob)
        prob_dict[(n_head,n_layer,rand_ind)] = probs
        count += 1
        print(f"{count}/360 done")
    with open('probs_360.pkl', 'wb') as file:     
    # A new file will be created
        pickle.dump(prob_dict, file)

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