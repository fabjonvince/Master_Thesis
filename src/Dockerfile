FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN mkdir ./workspace
WORKDIR /workspace

RUN apt-get update -y && \
    apt-get install -y curl \
                       git \
                       bash \
                       nano \
                       ssmtp \
                       subversion && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install transformers accelerate rouge_score
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install torch-geometric
RUN git clone https://github.com//ablodge/amr-utils
RUN pip install penman
RUN pip install ./amr-utils
RUN pip install datasets
RUN pip install gdown
RUN pip install simplejson==3.17.6
RUN pip install codecarbon sentence_transformers


# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
