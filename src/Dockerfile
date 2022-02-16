From python:3.8.8

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install setuptools wheel
RUN pip3 install scikit-build
RUN pip3 install cython
RUN pip3 install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]


CMD [ "service.py" ]