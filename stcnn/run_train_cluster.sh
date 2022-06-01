#!/bin/bash



logpath='/groups/card/cardlab/klapoetken/optic_glomeruli/logs/'

for i in {1..3}
do
job_prefix=$(date +'%Y%m%d_%H%M%S_')
job_suffix='lc_small_'$i
jobname=$job_prefix$job_suffix
bsub -P turaga \
-J $jobname \
-n 2 -gpu 'num=1' \
-q gpu_tesla \
-o $logpath$jobname.log \
python -u train.py expt_name=$job_suffix model=stcnn_exp_leakyrelu_bn batch_size=16 lr=0.0005 n_epochs=40
sleep 1 # sec
done
