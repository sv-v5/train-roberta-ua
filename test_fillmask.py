#!/usr/bin/env python3.8

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./models/robertua-v1",
    tokenizer="./models/robertua-v1"
)

result = fill_mask("вони їдуть до <mask>.")
[print(x) for x in result]