# syntax=docker/dockerfile:1
FROM python:3.8

# prepare virtual env and dependencies
COPY ./Pipfile* /
RUN python3.8 -m pip install pipenv && python3.8 -m pipenv install

# prepare data (mount data), scripts
COPY ./text/AF/wiki_08 /text/AF/
COPY ./*.py /
COPY ./run.sh /

# RUN 
CMD ./run.sh
