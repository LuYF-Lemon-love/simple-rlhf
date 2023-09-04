#!/bin/bash
#
# DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
#
# git pull from DeepSpeed-Chat by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 31, 2023
#
# 训练脚本.

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 2 main.py \
   --model_name_or_path ../../Atom-7B \
   --data_path ../../rm-static-zh \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --lora_dim 8 \
   --lora_module_name layers. \
   --only_optimize_lora \
   --num_train_epochs 1 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log