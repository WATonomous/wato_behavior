#!/bin/bash
#SBATCH --job-name=wato_behaviour_training
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --gres tmpdisk:40960,shard:20480
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j-%x.out  # %j: job ID, %x: job name. Reference: https://slurm.schedmd.com/sbatch.html#lbAH
 
slurm-start-dockerd.sh
export DOCKER_HOST=unix:///tmp/run/docker.sock
docker build -q -t wato_behavior:latest . && docker run --env-file=behaviour.env --name wato_behaviour -v "$(pwd):/home/bolty/wato_behaviour" wato_behavior:latest