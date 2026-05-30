#!/bin/bash
LOG="/root/experiment_followup.log"
cd /root
echo "===== FOLLOWUP START: $(date) =====" | tee $LOG

echo "[1/3] convnext_tiny_ft (60 epochs, fair baseline)" | tee -a $LOG
/root/miniconda3/bin/python run_experiments.py \
  --datasets cifar100 --models convnext_tiny_ft --batch_size 64 \
  2>&1 | tee -a $LOG

echo "[2/3] Ablation (30 epochs each, 4 variants)" | tee -a $LOG
/root/miniconda3/bin/python run_experiments.py \
  --datasets cifar100 --ablation --batch_size 64 --epochs 30 \
  2>&1 | tee -a $LOG

echo "[3/3] Flowers-102 (4 key models, 45 epochs)" | tee -a $LOG
/root/miniconda3/bin/python run_experiments.py \
  --datasets flowers102 \
  --models convnext_tiny convnext_tiny_ft mspa_convnext se_resnet50 \
  --batch_size 64 --epochs 45 \
  2>&1 | tee -a $LOG

echo "===== ALL DONE: $(date) =====" | tee -a $LOG
