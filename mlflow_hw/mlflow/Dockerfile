FROM python:3.7-slim

RUN pip install mlflow==1.10.0

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user \
  && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /home/new_user
