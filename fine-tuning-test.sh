#!/usr/bin/env bash

dataset_name=ChEMBL
model_type=N
#choose scoring function: tanimoto, activity_model, NOS, CLOGP, linker_length, QED_SA, MW, QED, SIM_3D, SIM_3D1, SIM_3D2, M_SIM_QED
score_function=activity_model
clf=./data/clf_jak3_active.pkl
step=1000
beamsize=64
train_nbest=32
test_nbest=50
train_type=M
w=0.0
k=1.0
savetopiMol=10
rand_seed=1024


lrlst=(0.00001 0.00001 0.00001 0.0001 0.0001)
trainTypelst=(M M M M B B B B)
modelTypelst=(L L L L L L N N N N N N)
klst=(1.0 1.0 0.8 1.0 1.0 0.8 1.0 0.8)
rand_seedlst=(0 10 100 1000)
cur_steplst=(2800 2500 2800 3000)
gpulst=(1 0 2 3 0 1 2 3)
nohaved=0
wlst=(2.5 2.5 2.5 2.5 1 1 1 1 2 2 2 2 3 3 3 3)
sigm=60

for index in 126 ;
do
{
w=1
chooseGpu=3
exp_id=${index}
k=1.0
lr=0.00001
train_type=M
model_type=N
cur_step=0

model=${model_type}_agent_${train_type}_${score_function}_bs${train_nbest}_sigma${sigm}_seed${rand_seed}_k${k}_lr${lr}_${exp_id}
pathsave=log/${score_function}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi
pathsave=case/${model_type}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi
pathsave=case/${model_type}/${score_function}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi
pathsave=case/${model_type}/${score_function}/${model}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi
pathsave=checkpoints/${dataset_name}/${model_type}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi
pathsave=checkpoints/${dataset_name}/${model_type}/${score_function}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi
pathsave=checkpoints/${dataset_name}/${model_type}/${score_function}/${model}
if [ ! -d "$pathsave" ]; then
  mkdir $pathsave
fi

CUDA_VISIBLE_DEVICES=$chooseGpu python visualmodule/oneCaseData.py \
    -id $exp_id \
    -testdatasrc data/${dataset_name}/${model_type}/src-test-new.txt \
    -testdatatgt data/${dataset_name}/${model_type}/tgt-test-new.txt \
    -outputsrc case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
    -outputtgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt


if [ "$train_type" == 'B' ];then

echo "beam search for fine-tuning"
run_script=train_agent.py
else

echo "Multinomial sampling for fine-tuning"
run_script=train_agent_ms.py
fi
echo "training model"
echo ${run_script}
if (("$cur_step" == "$nohaved"));then

CUDA_VISIBLE_DEVICES=$chooseGpu python ${run_script} -model checkpoints/SyntaLinker_${model_type}.pt \
                    -save_model checkpoints/${dataset_name}/${model_type}/${score_function}/${model}/${dataset_name}_${model} \
                    -src case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
                    -output log/${score_function}/${model}_batchsize${train_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $train_nbest \
                    -gpu 0 -log_probs -log_file log/${score_function}/${model}.txt \
                    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method fixed -warmup_steps 0  \
                    -rnn_size 256 -learning_rate $lr -label_smoothing 0.0 -report_every 20 \
                    -max_grad_norm 0 -save_checkpoint_steps 1000 -train_steps $step \
                    -scoring_function $score_function -score_function_num_processes 0 -sigma $sigm \
                    -score_para_k $k -score_para_w $w -score_para_clf $clf \
                    -gpu_ranks 0 -src_type ${model_type}
else
CUDA_VISIBLE_DEVICES=$chooseGpu python ${run_script} -model checkpoints/SyntaLinker_${model_type}.pt checkpoints/${dataset_name}/${model_type}/${model}/${dataset_name}_${model}_step_${cur_step}.pt\
                    -save_model checkpoints/${dataset_name}/${model_type}/${score_function}/${model}/${dataset_name}_${model} \
                    -src case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
                    -output log/${score_function}/${model}_batchsize${train_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $train_nbest \
                    -gpu 0 -log_probs -log_file log/${score_function}/${model}.txt \
                    -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method fixed -warmup_steps 0  \
                    -rnn_size 256 -learning_rate $lr -label_smoothing 0.0 -report_every 20 \
                    -max_grad_norm 0 -save_checkpoint_steps 1000 -train_steps $step \
                    -scoring_function $score_function -score_function_num_processes 0 -sigma $sigm \
                    -score_para_k $k -score_para_w $w \
                    -gpu_ranks 0 -src_type ${model_type}
fi

echo "beam search using prior model"
CUDA_VISIBLE_DEVICES=$chooseGpu python translate.py -model checkpoints/${dataset_name}/${model_type}/SyntaLinker_${model_type}.pt \
                    -src case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
                    -output case/${model_type}/${score_function}/${model}/predictions_SyntaLinker_${model_type}_on_${dataset_name}_beam${test_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $test_nbest \
                    -gpu 0 -log_probs -log_file ''

echo "Multinomial sampling using prior model"
CUDA_VISIBLE_DEVICES=$chooseGpu python translate_ms.py -model checkpoints/${dataset_name}/${model_type}/SyntaLinker_${model_type}.pt \
                    -src case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
                    -output case/${model_type}/${score_function}/${model}/predictions_SyntaLinker_${model_type}_on_${dataset_name}_ms${test_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $test_nbest \
                    -gpu 0 -log_probs -log_file ''

echo "beam search using agent model"
CUDA_VISIBLE_DEVICES=$chooseGpu python translate.py -model checkpoints/${dataset_name}/${model_type}/${score_function}/${model}/${dataset_name}_${model}_step_${step}.pt \
                    -src case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
                    -output case/${model_type}/${score_function}/${model}/predictions_${model}_step_${step}_on_${dataset_name}_beam${test_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $test_nbest \
                    -gpu 0 -log_probs -log_file ''


echo "Multinomial sampling using agent model"
CUDA_VISIBLE_DEVICES=$chooseGpu python translate_ms.py -model checkpoints/${dataset_name}/${model_type}/${score_function}/${model}/${dataset_name}_${model}_step_${step}.pt \
                    -src case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
                    -tgt case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
                    -output case/${model_type}/${score_function}/${model}/predictions_${model}_step_${step}_on_${dataset_name}_ms${test_nbest}.txt \
                    -batch_size 1 -replace_unk -max_length 200 -beam_size $beamsize -verbose -n_best $test_nbest \
                    -gpu 0 -log_probs -log_file ''





echo "draw and save topi gen mol and draw Distribution between prior and agent"
CUDA_VISIBLE_DEVICES=$chooseGpu python visualmodule/calAndDraw_0225.py \
    -topi $savetopiMol \
    -src_case_path case/${model_type}/${score_function}/${model}/src-test-${exp_id}.txt \
    -tgt_case_path case/${model_type}/${score_function}/${model}/tgt-test-${exp_id}.txt \
    -prior_beam_output case/${model_type}/${score_function}/${model}/predictions_SyntaLinker_${model_type}_on_${dataset_name}_beam${test_nbest}.txt \
    -agent_beam_output case/${model_type}/${score_function}/${model}/predictions_${model}_step_${step}_on_${dataset_name}_beam${test_nbest}.txt \
    -output_path case/${model_type}/${score_function}/${model} \
    -beam_size $test_nbest \
    -score_function_type $score_function -src_type ${model_type} \
    -clf_path $clf \
    -id $exp_id
}
done



