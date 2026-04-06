#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=pretrain_full_temp
#SBATCH --account=eecs595f25_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=pretrain_full_temp.out

# The application(s) to execute along with its input arguments and options:
module purge
module load python/3.12

echo "Starting full GPT training..."
echo "=================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-pretraining"
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=100
export TORCHDYNAMO_DISABLE=0

# Project and data directories (adjust if you use /scratch instead of /home)
PROJECT_DIR="/home/brgolani/cse_595"
DATA_PATH="$PROJECT_DIR/Data"
OUTPUT_DIR="$PROJECT_DIR/models/full"

# Activate your Python environment (must be created beforehand)
# Prefer a Python 3.12 environment for torch.compile (Dynamo unsupported on 3.13)
if [ -f "$HOME/venvs/e595-py312/bin/activate" ]; then
  source "$HOME/venvs/e595-py312/bin/activate"
elif [ -f "$HOME/venvs/e595/bin/activate" ]; then
  source "$HOME/venvs/e595/bin/activate"
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Move to project directory to resolve relative imports (gpt.py, rope.py, etc.)
cd "$PROJECT_DIR" || exit 1

# Use these hyperparameters for your full TORCHDYNAMO_DISABLE

# Training hyperparameters for full model
python pretrain_gpt.py \
  --batch_size 4 \
  --learning_rate 1e-3 \
  --max_epochs 1 \
  --emb_dim 32 \
  --n_layers 2 \
  --n_heads 4 \
  --context_length 1024 \
  --drop_rate 0.0 \
  --weight_decay 0.01 \
  --max_docs 100 \
  --save_every 50 \
  --eval_every 500 \
  --device cpu \
  --data_path "$DATA_PATH/fineweb-edu-sample-1B-hf/" \
  --data_format arrow \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "gpt-pretraining-$(date +%Y%m%d-%H%M%S)" \
  --num_workers 1 \
  --eval_data_path $DATA_PATH/fineweb-edu-eval-3M.jsonl.gz \
  --eval_data_format jsonl

echo "Training completed!"
echo "Check the output directory for saved models and logs."