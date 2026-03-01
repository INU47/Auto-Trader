import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AI_Brain.models import HybridModel
from AI_Brain.training_pipeline import get_training_data, QuantDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OptunaTuner")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    window_size = trial.suggest_categorical("window_size", [24, 32, 48, 64])

    # Fetch MTF data
    df = asyncio.run(get_training_data(n_candles=30000, timeframe_label=None))
    if df is None: return 0.0
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_dataset = QuantDataset(train_df, window_size=window_size)
    val_dataset = QuantDataset(val_df, window_size=window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = HybridModel(input_size=27, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for gaf, seq, l_cls, l_reg in train_loader:
            gaf, seq = gaf.to(device), seq.to(device)
            l_cls, l_reg = l_cls.to(device), l_reg.to(device)
            
            optimizer.zero_grad()
            logits, trend, _ = model(gaf, seq)
            loss = criterion_cls(logits, l_cls) + criterion_reg(trend.squeeze(), l_reg)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for gaf, seq, l_cls, l_reg in val_loader:
                gaf, seq = gaf.to(device), seq.to(device)
                l_cls, l_reg = l_cls.to(device), l_reg.to(device)
                logits, trend, _ = model(gaf, seq)
                loss = criterion_cls(logits, l_cls) + criterion_reg(trend.squeeze(), l_reg)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_val_loss

if __name__ == "__main__":
    logger.info("Starting Optuna Hyperparameter Search...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
    
    logger.info("Search Complete!")
    try:
        best_params = study.best_params
        logger.info(f"Best Trial: {best_params}")
        
        with open("Config/best_hyperparams.json", "w") as f:
            json.dump(best_params, f)
        
        logger.info("Best parameters saved to Config/best_hyperparams.json")
    except ValueError:
        logger.warning("⚠️ No successful trials found. Config was not updated.")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
