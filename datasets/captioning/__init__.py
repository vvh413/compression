import os

__DIR = os.path.dirname(os.path.abspath(__file__))
coco_raw = os.path.join(__DIR, "coco2017/")
coco_preprocessed = os.path.join(__DIR, "coco2017_preprocessed/")
annotations = os.path.join(__DIR, "annotations/")