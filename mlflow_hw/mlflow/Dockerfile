FROM python:3.7-slim

RUN pip install mlflow==1.10.0

EXPOSE 7777

#ARG USER_ID
#ARG GROUP_ID

#RUN addgroup --gid $GROUP_ID user \
#  && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
#USER user

#CMD mlflow server --backend-store-uri volume/meta --default-artifact-root volume/artifacts --host 0.0.0.0 --port 5000