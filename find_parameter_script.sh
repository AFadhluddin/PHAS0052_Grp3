#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request 48 hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:0

# Request 200 gigabtes of RAM (must be an integer followed by M, G, or T)
# 200 forces a better node, but the queue to get the job started might take longer
#$ -l mem=200G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N Simulation_Job

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/<your_UCL_id>/Scratch

# Your work should be done in $TMPDIR 
cd $TMPDIR

# Load the modules
module load python3/recommended

# Install tqdm library
pip3 install tqdm


# Runnning the job
python3 <path to main_loop.py file>/find_best_parameters.py

# Preferably, tar-up (archive) all output files onto the shared scratch area
tar -zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR

# Make sure you have given enough time for the copy to complete!
