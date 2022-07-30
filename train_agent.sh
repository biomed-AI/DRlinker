#!/usr/bin/env bash

dataset_name=ChEMBL
prior_model=${dataset_name}_prior_step_200000.pt
score_function=SIM_3D
score_type=QED
exp_id=5550
step=5000
beamsize=50
nbest=64
model=agent_${score_function}_${exp_id}
pathsave=case/${model}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
pathsave=checkpoints/${dataset_name}/${model}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi

agent_model=${dataset_name}_${model}_step_2000.pt

CUDA_VISIBLE_DEVICES=3 python train_agent.py -model checkpoints/${dataset_name}/${prior_model} \
                    -save_model checkpoints/${dataset_name}/${model}/${dataset_name}_${model} \
                    -src data/${dataset_name}/src-test-${exp_id}.txt \
                    -tgt data/${dataset_name}/tgt-test-${exp_id}.txt \
                    -output checkpoints/predictions_${model}_on_${dataset_name}_beam${nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $nbest \
                    -gpu 0 -log_probs -log_file log/agent_${exp_id}.txt \
                    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method fixed -warmup_steps 0  \
                    -rnn_size 256 -learning_rate 0.0001 -label_smoothing 0.0 -report_every 20 \
                    -max_grad_norm 0 -save_checkpoint_steps 1000 -train_steps $step \
                    -scoring_function $score_function -sigma 60 \
                    -score_function_num_processes 0

