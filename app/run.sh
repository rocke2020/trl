gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu
# 
file=
nohup python $file \
    > $file-gpu$gpu.log 2>&1 &