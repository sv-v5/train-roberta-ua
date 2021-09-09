import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./text/").glob("**/wiki*")]
# len(paths)   1500

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
# ['EsperBERTo/vocab.json', 'EsperBERTo/merges.txt']  when tokenizer.save_model("EsperBERTo")  is used.  not  tokenizer.save_model(".", "robertua")
os.mkdir("models/robertua")
tokenizer.save_model("models/robertua")

# TODO: wget https://huggingface.co/roberta-base/raw/main/config.json -P models/robertua