#!/bin/bash
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=5:10:00
#PBS -N extract-attentions

module load anaconda3/personal
source ~/.bashrc
conda activate antiberv

cp $PBS_O_WORKDIR/antiberta/antiberta/*.txt $TMPDIR
cp -r $PBS_O_WORKDIR/antiberta/antiberta/antibody-tokenizer $TMPDIR
cp -r $PBS_O_WORKDIR/antiberta/models $TMPDIR
cp $PBS_O_WORKDIR/antiberta/antiberta/extract_attentions.py $TMPDIR
mkdir assets

python extract_attentions.py

find $TMPDIR -name \*.pkl -exec cp {} $PBS_O_WORKDIR/antiberta/output/ \;
