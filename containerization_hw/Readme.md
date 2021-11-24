
# Dockerfile

Set starting working directory:

    WORKDIR /usr/src/


Install Boost:

    RUN wget -O boost_1_63_0.tar.gz https://sourceforge.net/projects/boost/files/boost/  1.63.0/boost_1_63_0.tar.gz/download
    RUN tar xzvf boost_1_63_0.tar.gz
    RUN mv boost_1_63_0 /usr/local/bin

Clone Starspace

    RUN git clone https://github.com/facebookresearch/Starspace.git

    WORKDIR /usr/src/Starspace
    RUN make

Create two environment variables for input and output files

    ENV input_file input.txt
    ENV output_file output.model 

Create non-root user
    ARG USER_ID	
    ARG GROUP_ID

    RUN addgroup --gid $GROUP_ID user
    RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
    USER user 

Run starspace training
    CMD ["sh", "-c", "./starspace train -trainFile ${input_file} -model ${output_file}"]

# Build

Build docker image:

    docker build -t myimage --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

# Run 
	docker run -it --name test --mount type=bind,source="$(pwd)"/volume,target=/usr/src/data -e input_file=/usr/src/data/input_file.txt -e output_file=/usr/src/data/starspace.model myimage:latest

The host volume folder was mounted into container's folder usr/src/data
Two environment variables point to this folder as a destination for loading and saving the data

