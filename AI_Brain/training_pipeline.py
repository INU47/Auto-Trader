import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import logging
import MetaTrader5 as mt5
from datetime import datetime
try:
    from preprocessor import GAFTransformer
    from models import PatternCNN, TrendLSTM
    from llm_advisor import LLMRewardAdvisor
except ImportError:
    from AI_Brain.preprocessor import GAFTransformer
    from AI_Brain.models import PatternCNN, TrendLSTM
    from AI_Brain.llm_advisor import LLMRewardAdvisor

# Configure Logger
logger = logging.getLogger("Trainer")

# --- 1. Real Data Ingestion (MT5) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Target Device: {device}")

def get_mt5_data(n_candles=200000, timeframe=mt5.TIMEFRAME_H1):
    # Load Credentials
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', 'mt5_config.json')
    if not os.path.exists(config_path):
        logger.error("Config file not found!")
        return None
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    logger.info(f"Initializing MT5 [{config['server']}]...")
    if not mt5.initialize(login=config['login'], server=config['server'], password=config['password']):
        logger.error(f"MT5 Initialize Failed: {mt5.last_error()}")
        return None

    # Get Symbols from config or default to list
    symbols = config.get('symbols', ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    all_data = []

    for symbol in symbols:
        logger.info(f"Fetching {n_candles} candles for {symbol}...")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data received for {symbol}!")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['symbol'] = symbol # Track source
        all_data.append(df)
        
    mt5.shutdown()
    
    if not all_data:
        logger.error("No data received from any symbol!")
        return None

    final_df = pd.concat(all_data, ignore_index=True)
    # Sort by time to ensure time-series order if needed, or shuffle in loader
    # For time-series split, we should split per symbol, but simpler: shuffle in training
    # For now, just return concatenated.
    logger.info(f"Total Data Points: {len(final_df)} candles across {len(symbols)} markets.")
    return final_df

# --- 2. Dataset Class (OHLCV Upgrade) ---
class QuantDataset(Dataset):
    def __init__(self, data, window_size=32, prediction_horizon=5, mode='train'):
        self.data = data
        self.window_size = window_size
        self.horizon = prediction_horizon
        self.gaf_transformer = GAFTransformer(image_size=window_size)
        
        self.X_gaf = []
        self.X_seq = []
        self.y_cls = []   
        self.y_reg = []  
        self.raw_prices = [] # For Backtesting
        
        self._prepare_data()

    def _detect_pattern(self, window_feat):
        """
        Detects specific candlestick patterns in the LAST candle of the window.
        Returns label (0-9)
        """
        if len(window_feat) < 2: return 0
        
        # Get last 2 candles
        curr = window_feat[-1] # o, h, l, c, v
        prev = window_feat[-2]
        
        o, h, l, c, v = curr
        po, ph, pl, pc, pv = prev
        
        body = abs(c - o)
        full_range = h - l if (h - l) > 0 else 0.0001
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        # 9: Doji (Indecision)
        if body <= 0.1 * full_range:
            return 9
            
        # 3: Hammer (Bullish Pin Bar)
        if lower_shadow >= 2 * body and upper_shadow <= 0.2 * full_range:
            return 3
            
        # 4: Shooting Star (Bearish Pin Bar)
        if upper_shadow >= 2 * body and lower_shadow <= 0.2 * full_range:
            return 4
            
        # 5: Bullish Engulfing
        if c > o and pc < po and c > po and o < pc:
            return 5
            
        # 6: Bearish Engulfing
        if c < o and pc > po and c < po and o > pc:
            return 6
            
        # Reversal Stars (Needs 3 bars)
        if len(window_feat) >= 3:
            prev2 = window_feat[-3]
            p2o, p2h, p2l, p2c, p2v = prev2
            
            # 7: Morning Star (Reversal Bull)
            if p2c < p2o and abs(pc - po) < 0.3 * abs(p2c - p2o) and c > o and c > (p2c + p2o)/2:
                return 7
            
            # 8: Evening Star (Reversal Bear)
            if p2c > p2o and abs(pc - po) < 0.3 * abs(p2c - p2o) and c < o and c < (p2c + p2o)/2:
                return 8

        # Fallback to general direction
        if c > o: return 1 # Bullish Momentum
        if c < o: return 2 # Bearish Momentum
        
        return 0 # Neutral

    def _prepare_data(self):
        # Extract features for LSTM
        # MT5 columns: time, open, high, low, close, tick_volume, spread, real_volume
        features = self.data[['open', 'high', 'low', 'close', 'tick_volume']].values
        close_prices = self.data['close'].values
        # Handle Symbol Multipliers
        # Default 100,000 for Forex, 100 for XAUUSD/Gold
        if 'symbol' in self.data.columns:
            symbols = self.data['symbol'].values
        else:
            symbols = ['EURUSD'] * len(self.data)
            
        n = len(features)
        
        logger.info("Preprocessing data (GAF + OHLCV)...")
        for i in range(self.window_size, n - self.horizon):
            window_feat = features[i-self.window_size : i] # [32, 5]
            window_close = close_prices[i-self.window_size : i] # [32]
            
            future_price = close_prices[i + self.horizon]
            current_price = close_prices[i - 1]
            current_symbol = symbols[i-1]
            
            # Determine Multiplier
            if 'XAU' in current_symbol or 'Gold' in current_symbol:
                multiplier = 100.0
            elif 'JPY' in current_symbol:
                multiplier = 1000.0 # Approximate for JPY pips (0.01) -> 100,000 * 0.01 = 1000 per pip? No.
                # Standard Lot: 100,000 units. 
                # USDJPY 150.00. 1 pip = 0.01.
                # Value = 100,000 * 0.01 / 150.00 ~ $6.6.
                # Direct price diff: 150.01 - 150.00 = 0.01.
                # 0.01 * 1000 = 10. Close enough to $10 standard.
                # Let's use 100000 / current_price roughly? Or just stick to 100000/100 split for simplified PnL
                # For simplified "Quote Currency" PnL:
                # Forex (USD quote): Price Diff * 100,000.
                # JPY pairs: Price Diff * 1000 (approx to USD).
                multiplier = 1000.0 
            else:
                multiplier = 100000.0

            # GAF (Using Close Price only for Image)
            gaf_img = self.gaf_transformer.transform(window_close)
            self.X_gaf.append(gaf_img)
            
            # Seq (OHLCV Normalization)
            min_vals = window_feat.min(axis=0)
            max_vals = window_feat.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            
            seq = (window_feat - min_vals) / range_vals
            self.X_seq.append(seq) # [32, 5]
            
            # Labels (Advanced Pattern Recognition)
            # Use pattern detection on the current window
            label = self._detect_pattern(window_feat)
            
            diff = future_price - current_price
            # Scale diff to "Points" to prevent vanishing gradients/outputs
            scaled_diff = diff * 100000.0
            
            self.y_cls.append(label)
            self.y_reg.append(scaled_diff)
            self.raw_prices.append((current_price, multiplier)) # Store tuple

    def __len__(self):
        return len(self.X_gaf)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_gaf[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.X_seq[idx], dtype=torch.float32),
            torch.tensor(self.y_cls[idx], dtype=torch.long),
            torch.tensor(self.y_reg[idx], dtype=torch.float32)
        )

# --- 3. Backtester ---
class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.trades = [] # {'type': 'BUY', 'entry': 1.0, 'exit': 1.1, 'pnl': 100}

    def run_backtest(self, model_cnn, model_lstm, val_loader, val_dataset):
        logger.info("Running Financial Backtest on Validation Set...")
        model_cnn.eval()
        model_lstm.eval()
        
        position = None # {'type': 'BUY', 'price': 1.2, 'size': 1.0}
        
        # Iterating directly over the validation dataset
        for i in range(len(val_dataset)):
            # Unpack the tuple from __getitem__
            gaf, seq, _, _ = val_dataset[i]
            # Access raw price and multiplier
            price, multiplier = val_dataset.raw_prices[i]
            
            # Inference
            gaf = gaf.unsqueeze(0).to(device) # [1, 1, 32, 32]
            seq = seq.unsqueeze(0).to(device) # [1, 32, 5]
            
            logits = model_cnn(gaf)
            probs = torch.softmax(logits, dim=1)
            conf, pred_cls = torch.max(probs, 1)
            pred_cls = pred_cls.item()
            
            trend = model_lstm(seq).item()
            
            if i < 20:
                logger.info(f"Sample {i}: Class={pred_cls}, Conf={conf.item():.4f}, Trend={trend:.6f}")

            # Logic
            signal = "HOLD"
            # Strong Signal Logic (Rely on CNN primarily for this test)
            # Labels: 1=Bull, 3=Hammer, 5=BullEngulf, 7=MorningStar
            if pred_cls in [1, 3, 5, 7] and conf.item() > 0.5:
                signal = "BUY"
            # Labels: 2=Bear, 4=Star, 6=BearEngulf, 8=EveningStar
            elif pred_cls in [2, 4, 6, 8] and conf.item() > 0.5:
                signal = "SELL"
            
            # Execution Simulation (Simple)
            if position is None:
                if signal == "BUY":
                    position = {'type': 'BUY', 'price': price, 'size': 0.1, 'mult': multiplier} # 0.1 Lot
                    logger.info(f"OPEN BUY at {price} (Sample {i})")
                elif signal == "SELL":
                    position = {'type': 'SELL', 'price': price, 'size': 0.1, 'mult': multiplier}
                    logger.info(f"OPEN SELL at {price} (Sample {i})")
            else:
                # Close Logic (e.g. Opposite signal or simple horizon exit)
                pnl = 0
                closed = False
                
                # Signal Exit
                if position['type'] == 'BUY' and signal == 'SELL':
                    pnl = (price - position['price']) * position['mult'] * position['size'] 
                    closed = True
                elif position['type'] == 'SELL' and signal == 'BUY':
                    pnl = (position['price'] - price) * position['mult'] * position['size']
                    closed = True
                
                # Force Close at End
                if i == len(val_dataset) - 1 and not closed:
                    if position['type'] == 'BUY':
                        pnl = (price - position['price']) * position['mult'] * position['size']
                    else:
                        pnl = (position['price'] - price) * position['mult'] * position['size']
                    closed = True
                    logger.info(f"FORCE CLOSE at {price} (End of Data)")
                
                if closed:
                    self.balance += pnl
                    self.trades.append(pnl)
                    logger.info(f"CLOSE {position['type']} | PnL: ${pnl:.2f}")
                    position = None
                    
        # Report
        wins = len([t for t in self.trades if t > 0])
        total = len(self.trades)
        win_rate = (wins/total*100) if total > 0 else 0
        profit = self.balance - self.initial_balance
        
        logger.info(f"=== Backtest Report ===")
        logger.info(f"Final Balance: ${self.balance:.2f}")
        logger.info(f"Total Trades: {total} | Win Rate: {win_rate:.1f}%")
        
        # Save to performance log
        self.save_performance_log(win_rate, profit, total, wins)
        
        return {
            'balance': self.balance,
            'profit': profit,
            'trades': total,
            'wins': wins,
            'win_rate': win_rate
        }
    
    def save_performance_log(self, win_rate, profit, total_trades, wins):
        """Saves backtest performance to a log file"""
        log_path = "AI_Brain/performance_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load AI mode from config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', 'server_config.json')
        ai_mode = "UNKNOWN"
        exploration_rate = 0.0
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                ai_mode = config.get('ai_mode', 'UNKNOWN')
                exploration_rate = config.get('exploration_rate', 0.0)
        
        # Format log entry
        log_entry = f"""
{'='*60}
[BACKTEST REPORT] {timestamp}
{'='*60}
AI Mode: {ai_mode}
Exploration Rate: {exploration_rate}
Win Rate: {win_rate:.2f}%
Total Trades: {total_trades}
Winning Trades: {wins}
Losing Trades: {total_trades - wins}
Net Profit: ${profit:.2f}
Final Balance: ${self.balance:.2f}
{'='*60}

"""
        
        # Append to log file
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        logger.info(f"Performance logged to {log_path}")

# --- 4. Main Training ---
def train_and_backtest():
    WINDOW_SIZE = 32
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.001
    
    if not os.path.exists("AI_Brain/weights"): os.makedirs("AI_Brain/weights")

    # 1. Fetch
    df = get_mt5_data()
    if df is None: return

    # 2. Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # 3. Train
    train_dataset = QuantDataset(train_df, window_size=WINDOW_SIZE)
    # Validate with Batch=1 for accurate backtesting sequence
    val_dataset = QuantDataset(val_df, window_size=WINDOW_SIZE) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Calculate Class Weights to handle Imbalance
    all_labels = [y for y in train_dataset.y_cls]
    class_counts = np.bincount(all_labels, minlength=10)
    # Avoid div by zero
    weights = 1.0 / (class_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Models (Input=5)
    cnn = PatternCNN().to(device)
    lstm = TrendLSTM(input_size=5, hidden_size=64, dropout=0.3).to(device)
    
    criterion_cls = nn.CrossEntropyLoss(weight=weights)
    criterion_reg = nn.MSELoss()
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=LR)
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=LR)
    
    # Simple Train Loop
    for epoch in range(EPOCHS):
        cnn.train(); lstm.train()
        total_loss = 0
        for gaf, seq, l_cls, l_reg in train_loader:
            gaf, seq = gaf.to(device), seq.to(device)
            l_cls, l_reg = l_cls.to(device), l_reg.to(device)
            
            optimizer_cnn.zero_grad(); optimizer_lstm.zero_grad()
            
            loss = criterion_cls(cnn(gaf), l_cls) + criterion_reg(lstm(seq).squeeze(), l_reg)
            loss.backward()
            
            optimizer_cnn.step(); optimizer_lstm.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    torch.save(cnn.state_dict(), "AI_Brain/weights/cnn_model.pt")
    torch.save(lstm.state_dict(), "AI_Brain/weights/lstm_model.pt")
    
    # 4. Backtest
    backtester = Backtester(initial_balance=10000)
    backtester.run_backtest(cnn, lstm, None, val_dataset)

def run_backtest_only():
    WINDOW_SIZE = 32
    BATCH_SIZE = 64
    
    # 1. Fetch
    df = get_mt5_data()
    if df is None: return

    # 2. Split (Same split as training)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:]
    val_dataset = QuantDataset(val_df, window_size=WINDOW_SIZE)
    
    # 3. Load Models
    cnn = PatternCNN()
    lstm = TrendLSTM(input_size=5, hidden_size=64, dropout=0.3)
    
    try:
        cnn.load_state_dict(torch.load("AI_Brain/weights/cnn_model.pt"))
        lstm.load_state_dict(torch.load("AI_Brain/weights/lstm_model.pt"))
        logger.info("Loaded trained weights.")
    except FileNotFoundError:
        logger.error("Weights not found! Train first.")
        return

    # 4. Run Backtest
    backtester = Backtester(initial_balance=10000)
    backtester.run_backtest(cnn, lstm, None, val_dataset)

# --- RL Training Classes ---
class RLExperienceDataset(Dataset):
    def __init__(self, experiences, window_size=32):
        self.experiences = experiences
        self.window_size = window_size
        self.gaf_transformer = GAFTransformer(image_size=window_size)
        
    def __len__(self):
        return len(self.experiences)
        
    def __getitem__(self, idx):
        exp = self.experiences[idx]
        # state is list of 32 candles
        df_state = pd.DataFrame(exp['state'])
        
        # GAF Image
        gaf_img = self.gaf_transformer.transform(df_state['close'].values)
        
        # Seq (OHLCV)
        window_feat = df_state[['open', 'high', 'low', 'close', 'tick_volume']].values
        min_vals = window_feat.min(axis=0)
        max_vals = window_feat.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        seq = (window_feat - min_vals) / range_vals
        
        return (
            torch.tensor(gaf_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(exp['action'], dtype=torch.long),
            torch.tensor(exp['reward'], dtype=torch.float32)
        )

async def train_rl_mode(experiences, advisor=None, db=None, epochs=10, lr=0.0001):
    """
    Reinforcement Learning update with LLM Reward Shaping.
    """
    if not experiences:
        logger.warning("No experiences for RL training.")
        return

    # 1. Reward Shaping with LLM Advisor
    if advisor and db:
        logger.info("Shaping rewards using LLM Advisor...")
        for exp in experiences:
            try:
                # Get Score and Reasoning
                score, reason = await advisor.get_quality_score(exp)
                await asyncio.sleep(1) # Proactive throttling for free-tier quota
                
                # Formula: Adjusted = PnL * (Score / 50.0)
                # 50 is neutral. 100 doubles reward. 0 nullifies reward.
                raw_reward = exp['reward']
                adjusted_reward = raw_reward * (score / 50.0)
                
                # Update Experience
                exp['reward'] = adjusted_reward
                
                # Log to DB for audit
                await db.log_llm_reward(exp['id'], score, reason, adjusted_reward)
            except Exception as e:
                logger.error(f"Reward shaping failed for trade {exp.get('id')}: {e}")

    logger.info(f"Starting RL Training session with {len(experiences)} samples...")
    
    WINDOW_SIZE = 32
    BATCH_SIZE = 16
    
    dataset = RLExperienceDataset(experiences, window_size=WINDOW_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    cnn = PatternCNN()
    lstm = TrendLSTM(input_size=5, hidden_size=64, dropout=0.3)
    
    # Load current weights if available
    try:
        cnn.load_state_dict(torch.load("AI_Brain/weights/cnn_model.pt"))
        lstm.load_state_dict(torch.load("AI_Brain/weights/lstm_model.pt"))
        logger.info("Loaded previous weights for RL refinement.")
    except:
        logger.warning("No existing weights found. Starting RL from scratch.")
        
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=lr)
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=lr)
    
    cnn.train(); lstm.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for gaf, seq, action, reward in loader:
            optimizer_cnn.zero_grad()
            optimizer_lstm.zero_grad()
            
            # 1. Forward Pass
            logits = cnn(gaf)
            probs = torch.softmax(logits, dim=1)
            
            # 2. Select prob of the action taken
            # action is 1 (BUY) or 2 (SELL)
            m = torch.distributions.Categorical(probs)
            log_prob = m.log_prob(action)
            
            # 3. Policy Gradient Loss (Maximizing expected reward)
            # Standard: minimize -log_prob * reward
            # Scale reward to be more stable (e.g. normalize or clamp)
            norm_reward = reward / 10.0 # Small scale
            loss = - (log_prob * norm_reward).mean()
            
            loss.backward()
            optimizer_cnn.step()
            optimizer_lstm.step()
            
            epoch_loss += loss.item()
            
        logger.info(f"RL Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.6f}")

    # Save upgraded weights
    if not os.path.exists("AI_Brain/weights"): os.makedirs("AI_Brain/weights")
    torch.save(cnn.state_dict(), "AI_Brain/weights/cnn_model.pt")
    torch.save(lstm.state_dict(), "AI_Brain/weights/lstm_model.pt")
    logger.info("RL Training Complete. Weights updated.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_and_backtest() 
    # run_backtest_only()
