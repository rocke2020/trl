gpu=$1
if [ -z $gpu ]; then
    gpu=2
fi
export CUDA_VISIBLE_DEVICES=$gpu
# 
file=app/trl/scripts/dpo.py
nohup python $file \
    > $file-nohup.log 2>&1 &