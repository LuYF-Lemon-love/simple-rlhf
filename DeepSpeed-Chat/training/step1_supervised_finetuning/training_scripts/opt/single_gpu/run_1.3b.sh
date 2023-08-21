#!/bin/bash
#
# DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_1.3b.sh
#
# git pull from DeepSpeed-Chat by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 19, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on August 19, 2023
#
# 训练脚本.

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 2 main.py --model_name_or_path ../../opt-1.3b \
   --data_path ../../rm-static \
   --per_device_train_batch_size 4 \
   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
