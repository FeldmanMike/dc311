#!/bin/bash

# Define the name of the Docker container
CONTAINER_NAME="dc311"

# Define the local project directory
LOCAL_DIR="$(pwd)/dc311"

# Define the directory inside the container
CONTAINER_DIR="/home/user/dc311"

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container is already running!"

# If the container exists and is not running, then re-start it
elif [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
    echo "Starting the existing stopped container..."
    docker start $CONTAINER_NAME

# Run the Docker container with volume mapping
else
    docker run -it --name $CONTAINER_NAME -v \
        $LOCAL_DIR:$CONTAINER_DIR dc-311-image

fi