FROM python:3.7-slim
COPY requirements.txt /tmp/
RUN apt-get update && pip install --requirement /tmp/requirements.txt
COPY . /tmp/

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user \
  && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENV PORT 5000
