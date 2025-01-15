gpu=$1
if [ -z $gpu ]; then
    gpu=2
fi
export CUDA_VISIBLE_DEVICES=$gpu
export HF_ENDPOINT=https://hf-mirror.com
#
file=app/trl/scripts/dpo.py
nohup python $file \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path /data/model/Qwen/Qwen2.5-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir /data/t2s-data/models_verify/Qwen2.5-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    > $file-gpu$gpu-nohup.log 2>&1 &