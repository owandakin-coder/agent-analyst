@echo off
cd /d "C:\Users\Ea Arage\Downloads\agent analyst"
echo [%date% %time%] Starting 15-stock training (6 Optuna trials)... >> train_15stocks.log
python main.py --mode train --optuna-trials 6 >> train_15stocks.log 2>&1
echo [%date% %time%] Training finished. >> train_15stocks.log
