import os
import shutil
from shutil import copy2

retrain_lst = []
count = 0
for train_dir in os.listdir("./finished"):
    print(count/360)
    if not train_dir.startswith("train-"):
        print(f"{train_dir} is not a proper directory")
    else:
        try:
            lst = os.listdir("./finished/"+train_dir+"/model")
            if len(lst) == 0:
                print(f"{train_dir} model folder is empty")
            elif len(lst) > 1:
                print(f"{train_dir} saves more than one model, removed")
                train_set = train_dir.split("-")[1:]
                train_set = [int(s) for s in train_set]
                retrain_lst.append(tuple(train_set))
                os.rename("./finished/"+train_dir,"./finished/delete"+train_dir)
                # shutil.rmtree("./finished/"+train_dir)
            else:
                src = lst[0]
                lst = src.split("-")
                str_train = train_dir[6:]
                try:
                    [n_layer, n_head, rand_ind] = lst
                    model_set = [int(n_head), int(n_layer), int(rand_ind)]
                    dst = "-".join([str(i) for i in model_set])
                    if dst != str_train:
                        print(f"{train_dir} doesn't match the model {dst}, removed")
                        train_set = train_dir.split("-")[1:]
                        train_set = [int(s) for s in train_set]
                        retrain_lst.append(tuple(train_set))
                        os.rename("./finished/"+train_dir,"./finished/delete"+train_dir)
                        # shutil.rmtree("./finished/"+train_dir)
                    else:
                        src_dir = "./finished/"+train_dir+"/model/"+src
                        dst_dir = "./models/model-"+dst
                        # shutil.copytree(src_dir, dst_dir, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)
                        # print(f"Copied model {dst}")
                except IndexError:
                    print(f"Some naming issue with {train_dir} folders")
        except FileNotFoundError:
            print(f"{train_dir} is not a proper directory, model folder doesn't exist")

    count += 1
print("finished")
print(retrain_lst)
with open("TEST2.txt", "w") as f:
    for line in retrain_lst:
        f.write(str(line)+"\n")