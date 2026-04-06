#!/bin/zsh
source .venv/bin/activate  # your mac venv
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=100
export TORCHDYNAMO_DISABLE=1  # compile is slow/unreliable on MPS

PROJECT_DIR="/Users/bgolani/Documents/CSE 595/Homework_3_export"
DATA_PATH="$PROJECT_DIR/Data"
OUTPUT_DIR="$PROJECT_DIR/models/full"
cd "$PROJECT_DIR" || exit 1

python pretrain_gpt.py \
  --batch_size 1 \
  --eval_batch_size 1 \
  --learning_rate 6e-4 \
  --max_epochs 1 \
  --emb_dim 512 \
  --n_layers 8 \
  --n_heads 8 \
  --context_length 1024 \
  --save_every 1000 \
  --eval_every 200 \
  --eval_max_docs_step 5 \
  --eval_max_docs_epoch 20 \
  --device mps \
  --data_path "$DATA_PATH/fineweb-edu-sample-1B-hf" \
  --data_format arrow \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "eecs595-gpt-pretraining" \
  --wandb_run_name "gpt-pretraining-local-$(date +%Y%m%d-%H%M%S)" \
  --num_workers 0 \
  --eval_data_path "$DATA_PATH/fineweb-edu-eval-3M.jsonl.gz" \
  --eval_data_format jsonl