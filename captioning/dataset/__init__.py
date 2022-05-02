import os

__DIR = os.path.dirname(os.path.abspath(__file__))
coco = os.path.join(__DIR, "coco2017/")
coco_preprocessed = os.path.join(__DIR, "coco2017_preprocessed/")
annotations = os.path.join(__DIR, "annotations/")
vocab_path = os.path.join(__DIR, "vocab.bin")


def coco_tag(tag):
    return os.path.dirname(coco) + "_" + tag
