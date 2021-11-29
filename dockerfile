FROM continuumio/miniconda3


WORKDIR /app

# RUN conda create -n fscount python=3.7 -y
# SHELL ["conda", "run", "-n", "fscount", "/bin/bash", "-c"]
# RUN pip install -r requirements.txt
# RUN conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch


# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash

# Fix libGL.so.1: cannot open shared object file
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Activate the environment, and make sure it's activated:
RUN echo "conda activate fscount" > ~/.bashrc

WORKDIR /workdir