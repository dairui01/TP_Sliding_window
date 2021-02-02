module load cuda/8.0 cudnn/5.1-cuda-8.0
mkdir -p weights_$2
export PATH=/home/sdas/anaconda2/bin:$PATH
python train_attention.py --epochs $1 $2 $3
