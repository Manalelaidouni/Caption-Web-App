FROM python:3.8-slim

ENV PUTHONUNBUFFERED 1

WORKDIR /myapp

COPY . /myapp/

RUN pip install "https://download.pytorch.org/whl/cpu/torch-1.10.1%2Bcpu-cp38-cp38-linux_x86_64.whl"
RUN pip install "https://download.pytorch.org/whl/cpu/torchvision-0.11.2%2Bcpu-cp38-cp38-linux_x86_64.whl"

RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:5000 app:app


