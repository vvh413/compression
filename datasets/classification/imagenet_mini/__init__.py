import os

__DIR = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(__DIR, "train")
val_root = os.path.join(__DIR, "val")


with open(os.path.join(__DIR, "labels.txt")) as __f:
    labels_dict = dict([
        tuple(line.strip().split("\t", 1)) 
        for line in __f.readlines() if line != ""
    ])
