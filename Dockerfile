FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install \
  https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
  git+https://github.com/1adrianb/face-alignment@cc02cf6f879c652f4fffc7b35be85c951c5e580e \
  -r requirements.txt

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install curl git cmake ack g++ vim-youcompleteme tmux -yq
#RUN apt install xdg-utils -yq
RUN apt-get install g++-8 -yq
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
#ENV DISPLAY=:0
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/qooba/vim-python-ide/master/setup.sh)"
#COPY jupyter /root/.jupyter
#COPY . /app/
RUN pip install gdown ffmpeg-python imutils dlib
COPY . /app/

