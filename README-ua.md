# Навчання RoBERTa Ukrainian Модель з нуля (language [en](./README.md) | [ua](./README-ua.md)) ![ci](https://github.com/sv-v5/train-roberta-ua/actions/workflows/ci.yaml/badge.svg)
Похвала https://github.com/youscan/language-models за їхню документацію та [roberta-ukrainian модель](https://huggingface.co/youscan/ukr-roberta-base).  
Це repository служить як повний приклад навчання--від скачання дані до тестування останньої моделі. Ціль тут є надати робочий підручник з інструкціями по англійській мові (та українській мові) який фіксує залежності від python і може бути запущений на GNU/Linux та на будь-якій системі де працює Docker.


## Передумови
[CUDA](https://developer.nvidia.com/cuda-downloads) підтримуваний GPU якщо навчання не є на CPU  
використовувати pipenv --python 3.8 (pipenv [покажчик](https://realpython.com/pipenv-guide/))  
Встановіть Python 3.8 (приклад для [системи Debian](https://linuxize.com/post/how-to-install-python-3-8-on-debian-10/), приклад через [apt](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/#installing-python-38-on-ubuntu-with-apt), приклад для [Windows](https://www.python.org/downloads/release/python-3810/) через Windows installer 64-bit (python3.8 має бути першим на $Path i зробіть `alias python3.8="python"`) )  
`python3.8 -m pip install pipenv && python3.8 -m pipenv install`  
[Встановіть CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) якщо навчання є на GPU  


## Основний процес
Перевірено на `Ubuntu 20.04.3 LTS` і `Python 3.8.10`  
`./run.sh` буде використовувати tiny wiki [dataset](./text/), тренувати tokenizer, тренувати мовну модель roberta і тестувати модель на fillmask приклад. Для використання повної wiki dataset, видаляйте папку `./text` і не-прокоментуйте рядки 13-15 в [run.sh](./run.sh) і запускаєте `./run.sh`  
<details><summary>Windows 10 64-bit і Python 3.8.10</summary><p>
    
в [git-bash](https://git-scm.com/download/win) shell запускаєте `./run.sh`. Повна wiki dataset може бути скачана з командами з `run.sh` якщо [wget для windows](https://eternallybored.org/misc/wget/1.19.4/32/wget.exe) є встановлене
</p></details>
<details><summary>Nvidia docker</summary><p>
    
[встановіть](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#installdocker) i запускаєте `docker build -t robertua-train . && docker run --gpus all robertua-train`. видаляєте `--gpus all` якщо використовуєте CPU
</p></details>


## Навчальний Скрипт
`run_language_modeling.py` адаптований з 2021 [версії](https://github.com/huggingface/transformers/blob/1c191efc3abc391072ff0094a8108459bc08e3fa/examples/legacy/run_language_modeling.py) transformers' мовного модель приклада


## Тривалість Навчання
Тривалість навчання буде залежати від устаткування комп'ютера та розміру dataset. На `GP104 GeForce GTX 1070 8 GB` навчання було оцінено зайняти 30 годин з `ukwiki-latest-pages-articles` dataset.  
Навчання тривало ~3 хвилини з tiny dataset з цього repository (`text/AF/{wiki_00,wiki_03,wiki_04,wiki_06,wiki_08}`) з розміром batch_size `4` на GPU , і ~33 хвилини з tiny dataset на `i7-10710U` CPU.
| Device                    |  Dataset  | Тривалість Навчання |
|:--------------------------|:---------:|:-------------------:|
| GeForce GTX 1070 8 GB GPU | full wiki |      30 годин       |
| GeForce GTX 1070 8 GB GPU | tiny wiki |      3 хвилини      |
| i7-10710U CPU             | tiny wiki |     33 хвилини      |


## Остання Модель
Навчена модель, розміром 487MB, буде випущена до папки `./models/robertua-v1/`

Приклад результатів навчання:
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
{'sequence': 'вони їдуть до..', 'score': 0.008971989154815674, 'token': 18, 'token_str': '.'}
{'sequence': 'вони їдуть до\n.', 'score': 0.002718620002269745, 'token': 203, 'token_str': '\n'}
{'sequence': 'вони їдуть до,.', 'score': 0.0021304022520780563, 'token': 16, 'token_str': ','}
{'sequence': 'вони їдуть донко.', 'score': 0.0002438406809233129, 'token': 11254, 'token_str': 'нко'}
{'sequence': 'вони їдуть до вірогід.', 'score': 0.0001972682512132451, 'token': 42912, 'token_str': ' вірогід'}
```


## Розширення Моделі
Використовувати новий dataset на розширення ukr-roberta-base модель.


## Ресурси
- Українська Roberta документація https://github.com/youscan/language-models і модель https://huggingface.co/youscan/ukr-roberta-base
- Transformers підручник https://huggingface.co/blog/how-to-train  
- Colab https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb  
- Польська модель стаття https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/  
- Docker з GPU https://towardsdatascience.com/how-to-properly-use-the-gpu-within-a-docker-container-4c699c78c6d1
- CUDA встановлення для Linux https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html і Windows https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html
