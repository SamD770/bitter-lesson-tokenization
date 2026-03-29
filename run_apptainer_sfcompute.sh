

CONTAINER_PATH=/root/bitter-lesson-tokenization/mamba_container_sfcompute.sif
SCRATCH_DIR=/data/scratch

apptainer shell --nv --bind $SCRATCH_DIR:/scratch/sdauncey $CONTAINER_PATH