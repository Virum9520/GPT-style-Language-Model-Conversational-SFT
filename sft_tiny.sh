#!/bin/bash

#SBATCH --job-name=sft_full
#SBATCH --account=eecs595f25_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=sft_full2.out

# The application(s) to execute along with its input arguments and options:
module purge
module load python/3.12
echo "Starting SFT training on full dataset..."

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-sft"
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=100
export TORCHDYNAMO_DISABLE=0

# Project paths
PROJECT_DIR="/home/brgolani/cse_595"
DATA_PATH="$PROJECT_DIR/Data"
MODEL_DIR="$PROJECT_DIR/models/full"
OUTPUT_DIR="$PROJECT_DIR/models/sft-models2"

# Activate your Python environment (must be created beforehand)
# Prefer a Python 3.12 environment for torch.compile (Dynamo unsupported on 3.13)
if [ -f "$HOME/venvs/e595-py312/bin/activate" ]; then
  source "$HOME/venvs/e595-py312/bin/activate"
elif [ -f "$HOME/venvs/e595/bin/activate" ]; then
  source "$HOME/venvs/e595/bin/activate"
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR" || exit 1

# Run SFT (train: Arrow packed, val: JSONL)
python sft_gpt.py \
  --train_data_path "$DATA_PATH/sft_data_packed.arrow" \
  --train_data_format arrow \
  --val_data_path "$DATA_PATH/smol-smoltalk-dev.jsonl.gz" \
  --val_data_format jsonl \
  --model_path "$MODEL_DIR/model_epoch_1.pt" \
  --context_length 1024 \
  --emb_dim 512 \
  --n_heads 8 \
  --n_layers 12 \
  --drop_rate 0.1 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --max_epochs 3 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 100 \
  --output_dir "$OUTPUT_DIR" \
  --save_every 10000 \
  --eval_every 500 \
  --wandb_project "gpt-sft-full" \
  --device "auto" \
  --num_workers 4 \
  --seed 42

echo "SFT training on full dataset finished."