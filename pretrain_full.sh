#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=pretrain_full
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=pretrain_full.out

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

# Print Python and Torch versions for sanity
python -c "import sys; print('Python:', sys.version)" || true
python -c "import torch; print('Torch:', torch.__version__)" 2>/dev/null || true

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Move to project directory to resolve relative imports (gpt.py, rope.py, etc.)
cd "$PROJECT_DIR" || exit 1

# Use these hyperparameters for your full TORCHDYNAMO_DISABLE

# Training hyperparameters for full model
python pretrain_gpt.py \
  --batch_size 16 \
  --learning_rate 7e-4 \
  --max_epochs 2 \
  --emb_dim 512 \
  --n_layers 12 \
  --n_heads 8 \
  --context_length 1024 \
  --save_every 1000 \
  --eval_every 500 \
  --eval_max_docs_step 50 \
  --eval_max_docs_epoch 200 \
  --drop_rate 0.05 \
  --device cuda \
  --data_path "$DATA_PATH/fineweb-edu-sample-1B-hf" \
  --data_format arrow \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "gpt-pretraining-$(date +%Y%m%d-%H%M%S)" \
  --num_workers 1 \
  --eval_data_path "$DATA_PATH/fineweb-edu-eval-3M.jsonl.gz" \
  --eval_data_format jsonl

echo "Training completed!"
echo "Check the output directory for saved models and logs."