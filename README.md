# Train Roberta Ukrainian Model from Scratch 
Credits to https://github.com/youscan/language-models for their documentation and their [roberta-ukrainian model](https://huggingface.co/youscan/ukr-roberta-base).  
This repository serves as a complete example of training-- from data downloading to testing the final model. The goal is to provide a working tutorial with instructions in English (and Ukrainian TODO) which locks the python dependencies and can be run on GNU/Linux and wherever Docker can be used (TODO).


## Main Process
Tested on `Ubuntu 20.04.3 LTS` and `Python 3.8.10`  
`./run.sh` will download a wiki dataset, train a tokenizer, train a roberta language model and test the model with a fillmask example


## Notes
using pipenv --python 3.8 (pipenv [guide](https://realpython.com/pipenv-guide/))


## Training Script
`run_language_modeling.py` is adapted from the 9-9-2021 [version](https://github.com/huggingface/transformers/blob/1c191efc3abc391072ff0094a8108459bc08e3fa/examples/legacy/run_language_modeling.py)


## Resources
- Ukrainian Roberta https://github.com/youscan/language-models  
- Transformers tutorial https://huggingface.co/blog/how-to-train  
- Colab https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb  
- Polish model article https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/  
