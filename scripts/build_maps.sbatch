#!/bin/bash
# Build the Predictive Graph of one layout on a SLURM cluster.
# Usage (from the repository root): sbatch scripts/build_maps.sbatch 153
# Add the partition, account or GPU type your cluster requires.
#SBATCH --job-name=build-map
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=build-map-%j.log

scripts/build_maps.sh "$@"
