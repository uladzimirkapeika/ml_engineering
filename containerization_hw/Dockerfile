FROM python:3.7-slim


WORKDIR /usr/src/

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends\
  wget \
  git \ 
  vim g++ make \
  && wget -O boost_1_63_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.tar.gz/download \
  && tar xzvf boost_1_63_0.tar.gz \
  && mv boost_1_63_0 /usr/local/bin \
  && git clone https://github.com/facebookresearch/Starspace.git

WORKDIR /usr/src/Starspace
RUN make

ENV input_file default_input_file
ENV output_file default_output_filedct

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user \
  && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

CMD ["sh", "-c", "./starspace train -trainFile ${input_file} -model ${output_file}"]
