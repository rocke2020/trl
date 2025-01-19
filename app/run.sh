set -eu

export CUDA_VISIBLE_DEVICES=$1
export HF_ENDPOINT=https://hf-mirror.com
file=app/trl/scripts/dpo.py
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > $file-nohup.log

#
nohup python $file \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path /data/model/Qwen/Qwen2.5-0.5B-Instruct \
    --torch_dtype bfloat16 \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 250 \
    --eval_strategy steps \
    --eval_steps 500 \
    --output_dir /data/t2s-data/models_verify/Qwen2.5-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 8 \
    >> $file-nohup.log 2>&1 &