FROM ubuntu:latest
FROM python:3.9

WORKDIR /app

COPY . /app

RUN python -m pip install Tensorflow/models/research/
RUN pip install -r requirements.txt
RUN pip install protobuf~=3.20
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "-m", "flask", "run"]