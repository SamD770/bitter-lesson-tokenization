#!/bin/bash

# Check if the local sif file exists, if not copy it from the network location
if [ ! -f "/scratch/sdauncey/sams_favorite_build.sif" ]; then
    echo "sams_favorite_build.sif not found locally, copying from network location..."
    cp /scratch_net/tikgpu10/sdauncey/sams_favorite_build.sif /scratch/sdauncey/sams_favorite_build.sif
    echo "Copy completed."
else
    echo "sams_favorite_build.sif found locally."
fi


apptainer shell --nv --bind /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey /scratch/sdauncey/sams_favorite_build.sif
# TODO: install matplotlib and jupyterrep