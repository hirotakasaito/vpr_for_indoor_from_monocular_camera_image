
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

COPY requirements.txt .
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3 python3-pip

RUN pip install --upgrade pip
RUN pip3 install PyYAML --ignore-installed
RUN pip3 install pytorch-lightning
RUN pip install -r requirements.txt && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


WORKDIR root
