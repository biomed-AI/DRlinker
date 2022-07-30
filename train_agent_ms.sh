#!/usr/bin/env bash

dataset_name=ChEMBL
model_type=L
score_function=SIM_3D
step=5000
beamsize=40
train_nbest=16
test_nbest=250

savetopiMol=10



chooseIdlst=(1 11100 13100 26700 27900)
gpuRunlst=(6 1 7 2)

for index in {0..0};
do
{

exp_id=${chooseIdlst[$index]}
chooseGpu=7

model=${model_type}_agent_${score_function}_${exp_id}
pathsave=case/${model_type}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
pathsave=case/${model_type}/${model}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
mkdir $pathsave
pathsave=checkpoints/${dataset_name}/${model_type}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
mkdir $pathsave
pathsave=checkpoints/${dataset_name}/${model_type}/${model}
if [ ! -d "$pathsave"]; then
  mkdir $pathsave
fi
mkdir $pathsave

CUDA_VISIBLE_DEVICES=$chooseGpu python visualmodule/oneCaseData.py \
    -id $exp_id \
    -testdatasrc data/${dataset_name}/${model_type}/src-test.txt \
    -testdatatgt data/${dataset_name}/${model_type}/tgt-test.txt \
    -outputsrc case/${model_type}/${model}/src-test-${exp_id}.txt \
    -outputtgt case/${model_type}/${model}/tgt-test-${exp_id}.txt


CUDA_VISIBLE_DEVICES=$chooseGpu python train_agent_ms.py -model checkpoints/${dataset_name}/${model_type}/SyntaLinker_${model_type}.pt \
                    -save_model checkpoints/${dataset_name}/${model_type}/${model}/${dataset_name}_${model} \
                    -src case/${model_type}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${model}/tgt-test-${exp_id}.txt \
                    -output log/${model}_batchsize${train_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size 50 -verbose -n_best 32 \
                    -gpu 0 -log_probs -log_file log/agent_${exp_id}.txt \
                    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method fixed -warmup_steps 0  \
                    -rnn_size 256 -learning_rate 0.00001 -label_smoothing 0.0 -report_every 50 \
                    -max_grad_norm 0 -save_checkpoint_steps 1000 -train_steps $step \
                    -scoring_function $score_function -score_function_num_processes 0 -sigma 60 \
                    -gpu_ranks 0


}
done


