config="/data/wangfei/project/nlp_cnn/data/config.yml"

# 定位conda
source /data/miniconda3/etc/profile.d/conda.sh

# 切换环境
conda activate wangfei

# 注册wandb
wandb login --host=https://api.wandb.ai

# 执行python训练脚本
CUDA_VISIBLE_DEVICES=4 python3 main.py ${config}


