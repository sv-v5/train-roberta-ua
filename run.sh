#!/usr/bin/env bash

# get training data and prepare it for training.  data_dir: ./text/
# wget https://dumps.wikimedia.org/ukwiki/latest/ukwiki-latest-pages-articles.xml.bz2
# bzip2 -d ukwiki-latest-pages-articles.xml.bz2
# pipenv run python -m wikiextractor.WikiExtractor ukwiki-latest-pages-articles.xml

### TODO: go over data preparation for wiki articles. is LineByLineTextDataset needed?


# train tokenizer and get roberta config.json
pipenv run python train_tokenizer.py
wget https://huggingface.co/roberta-base/raw/main/config.json -P models/robertua


# train lang model
# TODO: will this run on a CPU? how long would training take for a small dataset?
pipenv run python run_language_modeling.py \
--output_dir ./models/robertua-v1 \
--model_type roberta \
--mlm \
--config_name ./models/robertua \
--tokenizer_name ./models/robertua \
--train_data_files "./text/**/wiki*" \
--do_train \
--learning_rate 1e-6 \
--num_train_epochs 1 \
--save_total_limit 2 \
--save_steps 20000 \
--per_gpu_train_batch_size 4 \
--block_size=512 \
--seed 21


# test model mask fill. "test sentence with <mask>"
pipenv run python test_fillmask.py