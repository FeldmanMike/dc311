#!/bin/bash

# Define the name of the Docker container and the image
CONTAINER_NAME="dc311"
IMAGE_NAME="dc-311-image"

# Function to commit the container state
commit_container() {
    echo "Committing the container state to a new image..."
    docker commit $CONTAINER_NAME $IMAGE_NAME
    echo "Container state committed as image $IMAGE_NAME."
}

# Check if the container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping the running container..."
    docker stop $CONTAINER_NAME

# Check if the container exists (stopped or running)
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    # Commit the container state if required
    read -p "Do you want to commit the container state before removing it? \
    (y/n): " commit_choice
    if [ "$commit_choice" == "y" ]; then
        commit_container
    fi
    
    echo "Removing the container..."
    docker rm $CONTAINER_NAME
    echo "Container $CONTAINER_NAME has been removed."
else
    echo "Container $CONTAINER_NAME does not exist."
fi
