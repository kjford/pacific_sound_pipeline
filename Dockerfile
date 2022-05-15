FROM ubuntu:20.04

RUN apt-get update -y && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 && \
    apt-get clean

WORKDIR /code

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY humpback_model_cache/ humpback_model_cache/
