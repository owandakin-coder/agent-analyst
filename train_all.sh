#!/bin/bash
# train_all.sh — runs full training pipeline:
# 1. Train base Transformer model (Optuna + final 500k steps)
# 2. Train Ensemble (3 models × 500k steps each)
# Total estimated time: ~5-6 hours

set -e
cd "$(dirname "$0")"

echo "========================================"
echo " Phase 1: Transformer base model"
echo "========================================"
python -u main.py --mode train --optuna-trials 10

echo ""
echo "========================================"
echo " Phase 2: Ensemble (3 seeds)"
echo "========================================"
python -u main.py --mode train_ensemble

echo ""
echo "========================================"
echo " Training complete!"
echo " Models saved:"
echo "   models/final_model.zip"
echo "   models/ensemble_0.zip / 1 / 2"
echo "========================================"
