#!/bin/bash
## Required PBS Directives --------------------------------------
#PBS -A ERDCV0089CDAG
#PBS -q standard
#PBS -l select=1:ncpus=22:mpiprocs=2:ngpus=1
#PBS -l walltime=24:00:00
#PBS -l application=novel-feature-detector
#PBS -j oe

# env | sort
# hostname
# uname -a
# lsb_release -a

## Execution Block ----------------------------------------------
# Environment Setup
# cd to your scratch directory in /work
cd $WORK/rditljtd/Novel-Feature-Detector

# create a job-specific subdirectory based on JOBID and cd to it
# JOBID=`echo ${PBS_JOBID} | cut -d '.' -f 1`
# if [ ! -d ${JOBID} ]; then
#   mkdir -p ${JOBID}
# fi
# cd ${JOBID}

## Launching -----------------------------------------------------

# copy executable from $HOME and submit it
#cp ${HOME}/oil/main.py .
#cp ${HOME}/oil/stats.py .
#cp ${HOME}/oil/neural_arch.py .

#cp ${HOME}/OilPanReplacement/mainStudy1_AllChoice.py .
#cp ${HOME}/OilPanReplacement/stats.py .

module load $PROJECTS_HOME/datools/modulefiles/anaconda/3
source $HOME/.bashrc
conda activate EC2

#ccmrun python ${HOME}/amrdec_oil/main.py ${JOBID} > out.dat

ccmrun python train_ae.py --dataset=cifar100 --experiment_name=cifar100_ae --novelty_interval=1 --test_accuracy_interval=1 --num_runs=5