FROM python:3.7-slim
COPY requirements.txt ./
RUN apt-get update && pip install --requirement requirements.txt


ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user \
  && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /home/new_user

