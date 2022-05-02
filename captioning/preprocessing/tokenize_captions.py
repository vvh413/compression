from nltk.tokenize import TweetTokenizer
import json
import os
from datasets.captioning import annotations
from tqdm import tqdm

with open(os.path.join(annotations, "captions_train2017.json")) as f:
    json_data = json.load(f)

tokenizer = TweetTokenizer()

captions = json_data["annotations"]

tokenized = []
for capt in tqdm(captions):
    new_capt = capt
    new_capt["tokenized"] = " ".join(
        filter(len, tokenizer.tokenize(capt["caption"].lower()))
    )
    tokenized.append(new_capt)

json_data["annotations"] = tokenized

with open(os.path.join(annotations, "captions_train2017_tok.json"), "w") as f:
    json.dump(json_data, f)
