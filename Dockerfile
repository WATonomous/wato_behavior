FROM python:3.8.16-slim-bullseye

COPY . /home/bolty/wato_behaviour

WORKDIR /home/bolty/wato_behaviour

RUN apt-get -y update && apt-get install -y git libglib2.0-0  \
libsm6 libxrender1 libxext6 libgl1 libgl1-mesa-glx

RUN git clone --depth 1 https://github.com/metadriverse/metadrive.git && \
cd metadrive && \
pip install -e . && \
pip install stable-baselines3 tensorboard do-mpc && \
DEBIAN_FRONTEND=noninteractive && \
python -m metadrive.pull_asset

WORKDIR /home/bolty/wato_behaviour

CMD /bin/bash -c "python main.py"