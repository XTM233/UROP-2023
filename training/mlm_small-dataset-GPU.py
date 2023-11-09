RUN_HPC = False
MAX_TIME = 2 # in minutes
# ## Setup of all the things we need
# Some imports 
import concurrent.futures
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import os
import torch
# import pandas as pd
import numpy as np
# %matplotlib inline
# import matplotlib.pyplot as plt

# TODO change this to import
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

    
# print('average seq length original dataset')
# lens = [len(s) for s in seqs]
# print(np.mean(lens))
# print('average seq length aligned dataset')
# lens = [len(s) for s in seqs_al]
# print(np.mean(lens))
# NA= int(np.mean(lens))
# print('length original dataset')
# print(len(seqs))
# print('length aligned dataset')
# print(len(seqs_al))

## here I verify that anarci can chop amino acids but only at the end##
# '''
# indices0 = file[file.columns[0]].values
# indices = [int(ii[1:]) for ii in indices0]
# final_indices = []
# final_indices_al = []
# for s in range(len(indices)):
#     if abs(len(seqs[indices[s]])-len([seqs_al[s][p] for p in range(len(seqs_al[s])) if seqs_al[s][p]!='-'])) < 2:
#         final_indices.append(indices[s])
#         final_indices_al.append(s)
# '''

# Mb= len(seqs_al)
# final_indices_al=list(np.arange(len(seqs_al)))
# final_indices=list(np.arange(len(seqs)))
        
# m=3
# print('example original seq')
# print(seqs[final_indices[m]])
# print('example aligned seq')
# print(seqs_al[final_indices_al[m]])

def my_function():
    # code goes here
    # Initialise the tokeniser
    tokenizer = RobertaTokenizer.from_pretrained("antibody-tokenizer")

    # Initialise the data collator, which is necessary for batching
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # identify which round of training is this
    
    if RUN_HPC:
        f = open("./index.txt", "r")
        round_ind = int(f.readline())

        current_train_array = txt2list(f"./round_{round_ind}.txt")
        ind = int(os.environ["PBS_ARRAY_INDEX"])-1
        n_head,n_layer,random_ind = current_train_array[ind] # will raise error if 0 inplaced 
    else:
        n_head,n_layer,random_ind = (1,1,0)
    random_seed = random_ind + 2023

    # prepare datasets
    # I aligned and prepared a set of ~3000 antibodies from SabDab
    name_fasta='sabdab_heavy.txt'
    seqs_al  =load_FASTA(name_fasta)[1:]
    f = open("sabdab_heavy_pos.txt", "r")
    # out = f.read()
    # imgt_num=out.splitlines()

    ## positions taken from IMGT templates ##
    # b_cdr1 = imgt_num.index('27')
    # e_cdr1 = imgt_num.index('38')

    # b_cdr2 = imgt_num.index('56')
    # e_cdr2 = imgt_num.index('65')

    # b_cdr3 = imgt_num.index('105')
    # e_cdr3 = imgt_num.index('117')
        
    seqs=[]
    for s in range(len(seqs_al)):
        seqs.append(''.join([seqs_al[s][i] for i in range(len(seqs_al[s])) if seqs_al[s][i]!='-']))

    # split datasets if not already done
    split_ratio = [0.8,0.1,0.1]

    if not os.path.isfile("./assets/train_small.txt"):
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
    text_datasets = {
    "train": ['assets/train_small.txt'],
    "eval": ['assets/val_small.txt'],
    "test": ['assets/test_small.txt']
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


    # estimate memory required to choose an optimal batch size to fully utilise memory
    # def count_parameters(model: torch.nn.Module) -> int:
    #     """ Returns the number of learnable parameters for a PyTorch model """
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # count_parameters(model) # return 7818265 for 1 hidden layer

    max_memory_per_batch = 7818265*n_layer*6 # consider mixed precision, 4*1.5 bytes for each parameter
    n = 24*1024**3/max_memory_per_batch
    batch_s = int(2**(np.floor(np.log2(n)))) # get a nearest power 2 as batch size

    model_name = "-".join([str(n_layer).zfill(2),
                        str(n_head).zfill(2),
                        str(random_ind).zfill(2)])


    # ## Model configuration

    # These are the cofigurations we've used for pre-training.
    antiberta_config = {
        #"num_hidden_layers": 12,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        #"num_attention_heads": 12,
        "hidden_size": 768,
        "d_ff": 3072, # feed-forward dimension (possible to change?)
        "vocab_size": 25, # 20 aa + 5 symbols, including masked, start-end
        "max_len": 150,
        "max_position_embeddings": 152, #?
        "batch_size": batch_s, # params to explore
        "max_steps": 225000, # params to explore
        "weight_decay": 0.01, # params to explore
        "peak_learning_rate": 0.0001, # params to explore
        "labels":torch
    }

    # Initialise the model
    model_config = RobertaConfig(
        vocab_size=antiberta_config.get("vocab_size"),
        hidden_size=antiberta_config.get("hidden_size"),
        max_position_embeddings=antiberta_config.get("max_position_embeddings"),
        num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
        num_attention_heads=antiberta_config.get("num_attention_heads", 12),
        type_vocab_size=1,
        output_attentions=True
    )
    model = RobertaForMaskedLM(model_config)

    steps=50 #greater save steps, faster training
    # construct training arguments
    # Huggingface uses a default seed of 42
    args = TrainingArguments(
        output_dir="test",
        overwrite_output_dir=True,
        per_device_train_batch_size=antiberta_config.get("batch_size", 32),
        per_device_eval_batch_size=antiberta_config.get("batch_size", 32),
        max_steps=antiberta_config.get("max_steps", 12),
        #save_steps=2500,
        save_steps=steps,
        eval_steps = steps,
        logging_steps= steps, # params to explore
        adam_beta2=0.98, # params to explore
        adam_epsilon=1e-6, # params to explore
        weight_decay=antiberta_config.get("weight_decay", 12),
        #warmup_steps = 10000, # params to explore
        warmup_steps = 2, # params to explore
        learning_rate=1e-4, # params to explore
        save_total_limit = 3,
        gradient_accumulation_steps=antiberta_config.get("gradient_accumulation_steps", 8),
        fp16=RUN_HPC, # True - CUDA
        #bf16=True, # True - CUDA
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        seed=random_seed
    )
    if args.eval_steps > args.max_steps:
        print('Please change eval steps')


    # Setup of the HuggingFace Trainer
    # model early stops if evaluation metric worsens for 10 eval steps
    MyCallback = EarlyStoppingCallback(10, 1e-5)

    # %%
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"] # TODO inject tokenizer in trainer
    )
    trainer.add_callback(MyCallback)
    if len(os.listdir("./test")) != 0:
        # check whether there is existing checkpoint
        print("Resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Start training")
        trainer.train()

    trainer.save_model("./model/"+model_name)
    return True

import threading

def run_with_timeout(func, args=(), timeout=10):
    result = []

    def target():
        result.append(func(*args))

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError('Function call timed out after {} seconds'.format(timeout))
	
    return result

try:
    result = run_with_timeout(my_function, timeout=MAX_TIME*60)
except TimeoutError:
    print("Timeout.")
print("finished")
