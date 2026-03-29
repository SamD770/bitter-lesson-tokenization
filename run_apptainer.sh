#!/bin/bash

# IMAGE_NAME="sams_favorite_build.sif"
IMAGE_NAME="mamba_container2.sif"

# Check if the local sif file exists, if not copy it from the network location
if [ ! -f "/scratch/sdauncey/$IMAGE_NAME" ]; then
    echo "$IMAGE_NAME not found locally, copying from network location..."
    cp /itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/$IMAGE_NAME /scratch/sdauncey/$IMAGE_NAME
    echo "Copy completed."
else
    echo "$IMAGE_NAME found locally."
fi

apptainer shell --nv --bind /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey /scratch/sdauncey/$IMAGE_NAME
# TODO: install matplotlib and jupyterrep