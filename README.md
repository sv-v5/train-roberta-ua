# Train Roberta Ukrainian Model from Scratch 
Credits to https://github.com/youscan/language-models for their documentation and their [roberta-ukrainian model](https://huggingface.co/youscan/ukr-roberta-base).  
This repository serves as a complete example of training--from downloading data to testing the final model. The goal is to provide a working tutorial with instructions in English (and Ukrainian TODO) which locks the python dependencies and can be run on GNU/Linux and on whichever system Docker is available (TODO).


## Requirements
[CUDA](https://developer.nvidia.com/cuda-downloads) supported GPU if not training on CPU  
using pipenv --python 3.8 (pipenv [guide](https://realpython.com/pipenv-guide/))  
Install Python 3.8  
`pip install pipenv && pipenv install`  
[Install CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) if training on a GPU
(or [nvidia docker installed](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#installdocker) and `docker run --gpus` TODO)  


## Main Process
Tested on `Ubuntu 20.04.3 LTS` and `Python 3.8.10`  
`./run.sh` will download the wiki dataset, train a tokenizer, train a roberta language model and test the model with a fillmask example


## Training Script
`run_language_modeling.py` is adapted from the 9-9-2021 [version](https://github.com/huggingface/transformers/blob/1c191efc3abc391072ff0094a8108459bc08e3fa/examples/legacy/run_language_modeling.py)


## Training Time
Duration of training will depend on the hardware used and dataset size. On a `GP104 GeForce GTX 1070 8 GB` training was estimated to complete in 30 hours for the `ukwiki-latest-pages-articles` dataset.  
Training time was ~3 minutes for the tiny dataset included in this repository (`text/AF/{wiki_00,wiki_03wiki_04wiki_06,wiki_08}`) with a batch_size of `4`


## Final Model
The trained model will be output in `./models/robertua-v1`

Example training output:
```shell
[INFO|trainer.py:1168] 2021-09-09 18:18:15,218 >> ***** Running training *****
[INFO|trainer.py:1169] 2021-09-09 18:18:15,218 >>   Num examples = 1247
[INFO|trainer.py:1170] 2021-09-09 18:18:15,218 >>   Num Epochs = 1
[INFO|trainer.py:1171] 2021-09-09 18:18:15,218 >>   Instantaneous batch size per device = 8
[INFO|trainer.py:1172] 2021-09-09 18:18:15,218 >>   Total train batch size (w. parallel, distributed & accumulation) = 4
[INFO|trainer.py:1173] 2021-09-09 18:18:15,218 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:1174] 2021-09-09 18:18:15,218 >>   Total optimization steps = 312
100%|██████████████████████████████████████████████████████████████████████████████████| 312/312 [02:40<00:00,  2.06it/s][INFO|trainer.py:1366] 2021-09-09 18:20:55,950 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 160.7356, 'train_samples_per_second': 7.758, 'train_steps_per_second': 1.941, 'train_loss': 10.262784517728365, 'epoch': 1.0}
100%|██████████████████████████████████████████████████████████████████████████████████| 312/312 [02:40<00:00,  1.94it/s]
[INFO|trainer.py:1935] 2021-09-09 18:20:55,954 >> Saving model checkpoint to ./models/robertua-v1
[INFO|configuration_utils.py:391] 2021-09-09 18:20:55,955 >> Configuration saved in ./models/robertua-v1/config.json
[INFO|modeling_utils.py:1001] 2021-09-09 18:20:56,417 >> Model weights saved in ./models/robertua-v1/pytorch_model.bin
[INFO|tokenization_utils_base.py:2020] 2021-09-09 18:20:56,417 >> tokenizer config file saved in ./models/robertua-v1/tokenizer_config.json
[INFO|tokenization_utils_base.py:2026] 2021-09-09 18:20:56,418 >> Special tokens file saved in ./models/robertua-v1/special_tokens_map.json
```


## Resources
- Ukrainian Roberta https://github.com/youscan/language-models  
- Transformers tutorial https://huggingface.co/blog/how-to-train  
- Colab https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb  
- Polish model article https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/  
- Docker using GPU https://towardsdatascience.com/how-to-properly-use-the-gpu-within-a-docker-container-4c699c78c6d1
- CUDA installation for Linux https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html and Windows https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html
