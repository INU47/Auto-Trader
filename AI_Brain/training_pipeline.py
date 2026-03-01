import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import asyncio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import logging
import MetaTrader5 as mt5
from datetime import datetime
import hashlib
try:
    from preprocessor import GAFTransformer
    from models import HybridModel, get_best_hyperparams
    from llm_advisor import LLMRewardAdvisor
except ImportError:
    from AI_Brain.preprocessor import GAFTransformer
    from AI_Brain.models import HybridModel, get_best_hyperparams
    from AI_Brain.llm_advisor import LLMRewardAdvisor

logger = logging.getLogger("Trainer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Target Device: {device}")

async def get_training_data(n_candles=100000, timeframe_label=None):
    """
    Fetches M1, M5, and H1 data for balanced training across 4 pairs.
    Synchronizes using M1 as the anchor.
    """
    from Database.db_handler import DBHandler
    db = DBHandler()
    await db.connect()
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    all_data = []

    for symbol in symbols:
        logger.info(f"🔄 Syncing MTF Data for {symbol}...")
        
        # 1. Fetch M1 (Anchor)
        await db.ensure_data_continuity(symbol, 'M1', target_candles=n_candles)
        m1_candles = await db.get_candles(symbol, 'M1', limit=n_candles)
        if not m1_candles: continue
        df_m1 = pd.DataFrame(m1_candles)
        df_m1['time'] = pd.to_datetime(df_m1['time'], unit='s')
        df_m1 = df_m1.rename(columns={c: f'm1_{c}' for c in df_m1.columns if c != 'time' and c != 'symbol'})
        
        # 2. Fetch M5
        m5_candles = await db.get_candles(symbol, 'M5', limit=n_candles//5 + 1000)
        df_m5 = pd.DataFrame(m5_candles)
        df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
        df_m5 = df_m5.rename(columns={c: f'm5_{c}' for c in df_m5.columns if c != 'time' and c != 'symbol'})
        
        # 3. Fetch H1
        h1_candles = await db.get_candles(symbol, 'H1', limit=n_candles//60 + 500)
        df_h1 = pd.DataFrame(h1_candles)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        df_h1 = df_h1.rename(columns={c: f'h1_{c}' for c in df_h1.columns if c != 'time' and c != 'symbol'})
        
        # 4. Synchronize MTF
        # We merge M5 and H1 onto M1 using a left join on 'time'
        # Then we forward fill to handle the gaps between M5/H1 bars
        df_sync = pd.merge(df_m1, df_m5.drop(columns=['symbol'], errors='ignore'), on='time', how='left')
        df_sync = pd.merge(df_sync, df_h1.drop(columns=['symbol'], errors='ignore'), on='time', how='left')
        
        # Forward fill the MTF columns
        mtf_cols = [c for c in df_sync.columns if c.startswith('m5_') or c.startswith('h1_')]
        df_sync[mtf_cols] = df_sync[mtf_cols].ffill()
        
        # Fill remaining (initial) NaNs
        df_sync.fillna(method='bfill', inplace=True)
        
        all_data.append(df_sync)
        
    await db.close()
    
    if not all_data:
        logger.error("No data received from Database!")
        return None

    final_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ MTF Sync Complete: {len(final_df)} synchronized rows (27+ columns).")
    return final_df

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

CACHE_DIR = "AI_Brain/cache"

class QuantDataset(Dataset):
    def __init__(self, data, window_size=32, prediction_horizon=5, mode='train'):
        self.data = data.copy()
        self.window_size = window_size
        self.horizon = prediction_horizon
        self.mode = mode
        self.gaf_transformer = GAFTransformer(image_size=window_size)
        
        start_ts = data['time'].iloc[0].timestamp() if not data.empty else 0
        end_ts = data['time'].iloc[-1].timestamp() if not data.empty else 0
        data_info = f"{data.shape}_{start_ts}_{end_ts}_{window_size}_{prediction_horizon}_{mode}"
        self.cache_id = hashlib.md5(data_info.encode()).hexdigest()
        self.cache_path = os.path.join(CACHE_DIR, f"precalc_{self.cache_id}.npz")
        
        self.X_gaf = []
        self.X_seq = []
        self.y_cls = []   
        self.y_reg = []  
        self.raw_prices = []
        self.symbols = []
        
        if os.path.exists(self.cache_path):
            try:
                logger.info(f"🚀 Loading Preprocessed Cache: {self.cache_path}")
                loaded = np.load(self.cache_path, allow_pickle=True)
                
                self.X_gaf = list(loaded['X_gaf'])
                self.X_seq = list(loaded['X_seq'])
                self.y_cls = list(loaded['y_cls'])
                self.y_reg = list(loaded['y_reg'])
                self.raw_prices = list(loaded['raw_prices'])
                self.symbols = list(loaded['symbols']) if 'symbols' in loaded else ['EURUSD'] * len(self.X_gaf)
                
                # Validate all lengths
                expected_len = len(self.data) - self.window_size - self.horizon
                valid = (len(self.X_gaf) == expected_len and 
                         len(self.X_seq) == expected_len and 
                         len(self.y_cls) == expected_len and 
                         len(self.y_reg) == expected_len)
                
                if not valid:
                    logger.warning(f"⚠️ Cache Corrupted or Mismatched for {self.mode}. Purging.")
                    self.X_gaf = []; self.X_seq = []; self.y_cls = []; self.y_reg = []; self.raw_prices = []; self.symbols = []
                else:
                    logger.info(f"✅ Cache Validated for {self.mode} ({len(self.X_gaf)} items)")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Falling back to preprocessing.")
                self.X_gaf = []; self.X_seq = []; self.y_cls = []; self.y_reg = []; self.raw_prices = []; self.symbols = []

        self._prepare_data()
        self._save_cache()

    def _save_cache(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        try:
            np.savez_compressed(
                self.cache_path,
                X_gaf=np.array(self.X_gaf),
                X_seq=np.array(self.X_seq),
                y_cls=np.array(self.y_cls),
                y_reg=np.array(self.y_reg),
                raw_prices=np.array(self.raw_prices),
                symbols=np.array(self.symbols, dtype=object)
            )
            logger.info(f"📁 Preprocessed data saved to cache: {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _detect_pattern(self, window_feat):
        """
        Detects specific candlestick patterns in the LAST candle of the window.
        Returns label (0-9)
        """
        if len(window_feat) < 2: return 0
        
        curr = window_feat[-1] 
        prev = window_feat[-2]
        
        o, h, l, c, v = curr[:5]
        po, ph, pl, pc, pv = prev[:5]
        
        body = abs(c - o)
        full_range = h - l if (h - l) > 0 else 0.0001
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        if body <= 0.1 * full_range:
            return 9
            
        if lower_shadow >= 2 * body and upper_shadow <= 0.2 * full_range:
            return 3
            
        if upper_shadow >= 2 * body and lower_shadow <= 0.2 * full_range:
            return 4
            
        if c > o and pc < po and c > po and o < pc:
            return 5
            
        if c < o and pc > po and c < po and o > pc:
            return 6
            
        if len(window_feat) >= 3:
            prev2 = window_feat[-3]
            p2o, p2h, p2l, p2c, p2v = prev2[:5]
            
            if p2c < p2o and abs(pc - po) < 0.3 * abs(p2c - p2o) and c > o and c > (p2c + p2o)/2:
                return 7
            
            if p2c > p2o and abs(pc - po) < 0.3 * abs(p2c - p2o) and c < o and c < (p2c + p2o)/2:
                return 8

        if c > o: return 1
        if c < o: return 2
        
        return 0

    def _prepare_data(self):
        """
        Processes synchronized MTF data.
        Features are organized as [M1 features..., M5 features..., H1 features...]
        """
        # Ensure Indicators are calculated for M1 if missing
        for tf in ['m1', 'm5', 'h1']:
            close_col = f'{tf}_close'
            if close_col not in self.data.columns: continue
            
            # RSI
            delta = self.data[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            self.data[f'{tf}_rsi'] = (100 - (100 / (1 + rs))) / 100.0
            
            # MACD
            ema12 = self.data[close_col].ewm(span=12, adjust=False).mean()
            ema26 = self.data[close_col].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            self.data[f'{tf}_macd_sig'] = macd_line.ewm(span=9, adjust=False).mean()
            self.data[f'{tf}_macd_hist'] = macd_line - self.data[f'{tf}_macd_sig']
            
            # ATR (M1 only for SL/TP purposes)
            if tf == 'm1':
                high_low = self.data['m1_high'] - self.data['m1_low']
                high_close = (self.data['m1_high'] - self.data['m1_close'].shift()).abs()
                low_close = (self.data['m1_low'] - self.data['m1_close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                self.data['m1_atr'] = tr.rolling(window=14).mean()

        if 'm1_sentiment' not in self.data.columns: self.data['m1_sentiment'] = 0.0
        if 'm5_sentiment' not in self.data.columns: self.data['m5_sentiment'] = 0.0
        if 'h1_sentiment' not in self.data.columns: self.data['h1_sentiment'] = 0.0

        self.data.ffill(inplace=True)
        self.data.fillna(0, inplace=True)

        # Build feature list (27 columns)
        feature_cols = []
        for tf in ['m1', 'm5', 'h1']:
            feature_cols.extend([
                f'{tf}_open', f'{tf}_high', f'{tf}_low', f'{tf}_close', 
                f'{tf}_tick_volume', f'{tf}_rsi', f'{tf}_macd_sig', f'{tf}_macd_hist', f'{tf}_sentiment'
            ])
            
        features = self.data[feature_cols].values
        m1_close = self.data['m1_close'].values
        m1_atr = self.data['m1_atr'].values
        symbols = self.data['symbol'].values if 'symbol' in self.data.columns else ['EURUSD'] * len(self.data)
            
        n = len(features)
        
        logger.info(f"Preprocessing MTF data ({len(feature_cols)} features)...")
        for i in range(self.window_size, n - self.horizon):
            window_feat = features[i-self.window_size : i]
            window_m1_close = m1_close[i-self.window_size : i]
            
            future_price = m1_close[i + self.horizon]
            current_price = m1_close[i - 1]
            current_symbol = symbols[i-1]
            
            multiplier = 100000.0
            if 'XAU' in current_symbol or 'Gold' in current_symbol: multiplier = 100.0
            elif 'JPY' in current_symbol: multiplier = 1000.0

            # GAF uses H1 (or the largest TF) usually, but here we'll use M1 as the "base" sequence
            gaf_img = self.gaf_transformer.transform(window_m1_close)
            self.X_gaf.append(gaf_img)
            
            # Normalize sequence
            min_vals = window_feat.min(axis=0)
            max_vals = window_feat.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            seq = (window_feat - min_vals) / range_vals
            self.X_seq.append(seq)
            
            # Detect pattern on M1
            m1_window = window_feat[:, :9] # First 9 are M1
            label = self._detect_pattern(m1_window)
            
            diff = future_price - current_price
            scaled_diff = diff * 100000.0 # Standardize for training
            
            self.y_cls.append(label)
            self.y_reg.append(scaled_diff)
            
            self.raw_prices.append((current_price, multiplier, m1_atr[i-1]))
            self.symbols.append(current_symbol)

    def __len__(self):
        return len(self.X_gaf)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_gaf[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.X_seq[idx], dtype=torch.float32),
            torch.tensor(self.y_cls[idx], dtype=torch.long),
            torch.tensor(self.y_reg[idx], dtype=torch.float32)
        )

class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.trades = []

    def run_backtest(self, model, val_loader, val_dataset):
        logger.info("Running Enhanced Financial Backtest (Symbol-Aware + SL/TP)...")
        model.eval()
        
        position = None 
        backtest_log = []
        
        RISK_REWARD = 2.0
        MAX_HOLD_BARS = 48 
        
        symbol_stats = {}
        
        current_symbol = None
        bars_held = 0
        
        max_conf_seen = 0.0
        for i in range(len(val_dataset)):
            window_size = getattr(val_dataset, 'window_size', 32)
            data_idx = i + window_size - 1
            
            row_data = {}
            if hasattr(val_dataset, 'data') and data_idx < len(val_dataset.data):
                row_data = val_dataset.data.iloc[data_idx]
            else:
                logger.warning(f"🚨 Backtest Loop Break: indexing mismatch at i={i}")
                break
                
            item_symbol = row_data.get('symbol', 'UNKNOWN')
            if current_symbol is None: current_symbol = item_symbol
            force_close = (item_symbol != current_symbol) or (i == len(val_dataset) - 1)
            
            gaf, seq, _, _ = val_dataset[i]
            price, multiplier, _ = val_dataset.raw_prices[i]
            
            if position is not None:
                bars_held += 1
                pnl = 0
                closed = False
                reason = ""
                
                if position['type'] == 'BUY':
                    if price <= position['sl']:
                        pnl = (position['sl'] - position['price']) * position['mult'] * position['size']
                        closed = True; reason = "STOP LOSS"
                    elif price >= position['tp']:
                        pnl = (position['tp'] - position['price']) * position['mult'] * position['size']
                        closed = True; reason = "TAKE PROFIT"
                elif position['type'] == 'SELL':
                    if price >= position['sl']:
                        pnl = (position['price'] - position['sl']) * position['mult'] * position['size']
                        closed = True; reason = "STOP LOSS"
                    elif price <= position['tp']:
                        pnl = (position['price'] - position['tp']) * position['mult'] * position['size']
                        closed = True; reason = "TAKE PROFIT"
                
                if not closed:
                    if bars_held >= MAX_HOLD_BARS:
                        closed = True; reason = "TIME EXIT"
                    elif force_close:
                        closed = True; reason = "SYMBOL CHANGE/END"
                    
                    if closed:
                        if position['type'] == 'BUY':
                            pnl = (price - position['price']) * position['mult'] * position['size']
                        else:
                            pnl = (position['price'] - price) * position['mult'] * position['size']

                if closed:
                    self.balance += pnl
                    self.trades.append(pnl)
                    sym = position['symbol']
                    if sym not in symbol_stats: symbol_stats[sym] = {'trades': [], 'wins': 0}
                    symbol_stats[sym]['trades'].append(pnl)
                    if pnl > 0: symbol_stats[sym]['wins'] += 1
                    
                    backtest_log.append({
                        'symbol': sym, 'type': position['type'], 
                        'entry': position['price'], 'exit': price, 
                        'pnl': pnl, 'reason': reason, 'bars': bars_held
                    })
                    position = None
                    bars_held = 0

            if position is None and not force_close:
                current_symbol = item_symbol
                gaf_in = gaf.unsqueeze(0).to(device)
                seq_in = seq.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, _, _ = model(gaf_in, seq_in)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_cls = torch.max(probs, 1)
                    pred_cls = pred_cls.item()
                    if conf.item() > max_conf_seen: max_conf_seen = conf.item()
                
                if conf.item() > 0.5:
                    _, multiplier, atr = val_dataset.raw_prices[i]
                    sl_offset = (atr * 1.5) if (atr and atr > 0) else (price * 0.002)
                    tp_offset = sl_offset * RISK_REWARD
                    
                    if pred_cls in [1, 3, 5, 7]:
                        position = {'type': 'BUY', 'price': price, 'sl': price - sl_offset, 'tp': price + tp_offset, 'size': 0.1, 'mult': multiplier, 'symbol': current_symbol}
                    elif pred_cls in [2, 4, 6, 8]:
                        position = {'type': 'SELL', 'price': price, 'sl': price + sl_offset, 'tp': price - tp_offset, 'size': 0.1, 'mult': multiplier, 'symbol': current_symbol}
                    
                    if position and current_symbol not in symbol_stats:
                        symbol_stats[current_symbol] = {'trades': [], 'wins': 0}

        wins = len([t for t in self.trades if t > 0])
        total = len(self.trades)
        win_rate = (wins/total*100) if total > 0 else 0
        profit = self.balance - self.initial_balance
        
        try:
            pd.DataFrame(backtest_log).to_csv("AI_Brain/backtest_detailed_trades.csv", index=False)
            logger.info("📝 Detailed backtest log saved: AI_Brain/backtest_detailed_trades.csv")
        except Exception as e:
            logger.error(f"Failed to save detailed log: {e}")

        logger.info(f"=== Enhanced Backtest Report ===")
        logger.info(f"Final Balance: ${self.balance:.2f} | Net Profit: ${profit:.2f}")
        logger.info(f"Total Trades: {total} | Win Rate: {win_rate:.1f}%")
        logger.info(f"Diagnostic: Max Confidence Seen: {max_conf_seen:.4f}")
        
        logger.info("--- Symbol Breakdown ---")
        for sym, stats in symbol_stats.items():
            s_total = len(stats['trades'])
            if s_total > 0:
                s_wr = (stats['wins'] / s_total) * 100
                s_profit = sum(stats['trades'])
                logger.info(f"[{sym}] Trades: {s_total} | WR: {s_wr:.1f}% | Profit: ${s_profit:.2f}")
        
        self.save_performance_log(win_rate, profit, total, wins)
        return {'balance': self.balance, 'profit': profit, 'trades': total, 'win_rate': win_rate}
    
    def save_performance_log(self, win_rate, profit, total_trades, wins):
        """Saves backtest performance to a log file"""
        log_path = "AI_Brain/performance_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config', 'server_config.json')
        ai_mode = "UNKNOWN"
        exploration_rate = 0.0
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                ai_mode = config.get('ai_mode', 'UNKNOWN')
                exploration_rate = config.get('exploration_rate', 0.0)
        
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
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        logger.info(f"Performance logged to {log_path}")

def train_and_backtest(epochs=50):
    from models import get_best_hyperparams
    params = get_best_hyperparams()
    
    WINDOW_SIZE = params['window_size']
    BATCH_SIZE = params['batch_size']
    HIDDEN_SIZE = params['hidden_size']
    DROPOUT = params['dropout']
    EPOCHS = epochs
    LR = params['lr']
    
    logger.info(f"🚀 Training with Optimized Parameters: {params} | Epochs per Fold: {EPOCHS}")
    
    if not os.path.exists("AI_Brain/weights"): os.makedirs("AI_Brain/weights")

    df = asyncio.run(get_training_data(n_candles=100000, timeframe_label='H1'))
    if df is None: return

    fold_results = []
    best_stats = {'win_rate': -1.0, 'profit': -float('inf')}
    
    splits = [
        (0, int(len(df)*0.6), int(len(df)*0.8)),
        (0, int(len(df)*0.8), len(df))
    ]
    
    for fold_idx, (start, mid, end) in enumerate(splits):
        logger.info(f"[Fold {fold_idx+1}] Starting Walk-Forward Fold...")
        train_df = df.iloc[start:mid]
        val_df = df.iloc[mid:end]
        
        train_dataset = QuantDataset(train_df, window_size=WINDOW_SIZE, mode=f'fold_{fold_idx}_train')
        val_dataset = QuantDataset(val_df, window_size=WINDOW_SIZE, mode=f'fold_{fold_idx}_val')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        all_labels = [y for y in train_dataset.y_cls]
        class_counts = np.bincount(all_labels, minlength=10)
        weights = 1.0 / (class_counts + 1e-6)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        model = HybridModel(input_size=27, hidden_size=HIDDEN_SIZE).to(device)
        weights_path = "AI_Brain/weights/hybrid_model.pt"
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=device)
                # Verify input size match
                if state_dict['lstm.lstm.weight_ih_l0'].shape[1] == 27:
                    model.load_state_dict(state_dict)
                    logger.info(f"[Checkpoint] Continuing training from existing MTF weights for Fold {fold_idx+1}")
                else:
                    logger.warning(f"[Incompatible] Found old 9-feature weights at {weights_path}. Skipping load.")
            except Exception as e:
                logger.warning(f"Could not load weights for continuation: {e}")
        
        criterion_cls = nn.CrossEntropyLoss(weight=weights)
        criterion_reg = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        # User Requirement: LR Decay when EarlyStopping counter reaches 5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for gaf, seq, l_cls, l_reg in train_loader:
                gaf, seq = gaf.to(device), seq.to(device)
                l_cls, l_reg = l_cls.to(device), l_reg.to(device)
                optimizer.zero_grad()
                logits, trend, _ = model(gaf, seq)
                loss = criterion_cls(logits, l_cls) + criterion_reg(trend.squeeze(), l_reg)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            val_loader_internal = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            with torch.no_grad():
                for gaf, seq, l_cls, l_reg in val_loader_internal:
                    gaf, seq = gaf.to(device), seq.to(device)
                    l_cls, l_reg = l_cls.to(device), l_reg.to(device)
                    logits, trend, _ = model(gaf, seq)
                    loss = criterion_cls(logits, l_cls) + criterion_reg(trend.squeeze(), l_reg)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader_internal)
            # Step scheduler based on performance
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"AI_Brain/weights/best_hybrid_fold_{fold_idx}.pt")
                torch.save(model.state_dict(), "AI_Brain/weights/best_hybrid_model.pt")

            # User Requirement: Backtest every 10 epochs (or if loss doesn't improve for 10)
            # We'll stick to fixed 10-epoch interval + Fold completion
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Fold {fold_idx+1} | Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.8f}")
                
                backtester = Backtester()
                results = backtester.run_backtest(model, None, val_dataset)
                
                if results['trades'] >= 10 and results['win_rate'] > best_stats.get('win_rate', -1):
                    best_stats['win_rate'] = results['win_rate']
                    torch.save(model.state_dict(), "AI_Brain/weights/weights_guardian.pt")
                    logger.info(f"[GUARDIAN] New Checkpoint (Fold {fold_idx+1}) | WR: {results['win_rate']:.2f}%")
                
                if results['profit'] > best_stats.get('profit', -float('inf')):
                    best_stats['profit'] = results['profit']
                    torch.save(model.state_dict(), "AI_Brain/weights/weights_explorer.pt")
                    logger.info(f"[EXPLORER] New Checkpoint (Fold {fold_idx+1}) | Profit: ${results['profit']:.2f}")

            if early_stopping(avg_val_loss, model):
                break
        
        model.load_state_dict(torch.load(f"AI_Brain/weights/best_hybrid_fold_{fold_idx}.pt"))
        backtester = Backtester()
        final_res = backtester.run_backtest(model, None, val_dataset)
        fold_results.append(final_res)
        logger.info(f"[SUCCESS] Fold {fold_idx+1} complete. Net Profit: ${final_res['profit']:.2f} | Win Rate: {final_res['win_rate']:.2f}%")
    try:
        best_path = "AI_Brain/weights/best_hybrid_model.pt"
        prod_path = "AI_Brain/weights/hybrid_model.pt"
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path))
            torch.save(model.state_dict(), prod_path)
            logger.info(f"Training Complete. Best model promoted to production.")
        else:
            torch.save(model.state_dict(), prod_path)
            logger.warning("No best model file found to promote. Saving current model state.")
    except Exception as e:
        logger.error(f"Failed to promote best model: {e}")
        torch.save(model.state_dict(), "AI_Brain/weights/hybrid_model.pt")
    
    backtester = Backtester()
    backtester.run_backtest(model, None, val_dataset)
    
    return fold_results

def run_backtest_only():
    from models import get_best_hyperparams
    params = get_best_hyperparams()
    
    WINDOW_SIZE = params['window_size']
    HIDDEN_SIZE = params['hidden_size']
    
    df = asyncio.run(get_training_data(n_candles=100000, timeframe_label='H1'))
    if df is None: return

    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:]
    val_dataset = QuantDataset(val_df, window_size=WINDOW_SIZE)
    
    model = HybridModel(input_size=27, hidden_size=HIDDEN_SIZE).to(device)
    
    weights_path = "AI_Brain/weights/hybrid_model.pt"
    if not os.path.exists(weights_path):
        weights_path = "AI_Brain/weights/best_hybrid_model.pt"
        
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info(f"Loaded trained Hybrid model from {weights_path}.")
    else:
        logger.error("Hybrid weights not found! Train first (looked for hybrid_model.pt and best_hybrid_model.pt).")
        return

    backtester = Backtester(initial_balance=10000)
    backtester.run_backtest(model, None, val_dataset)

class RLExperienceDataset(Dataset):
    def __init__(self, experiences, window_size=32):
        self.experiences = experiences
        self.window_size = window_size
        self.gaf_transformer = GAFTransformer(image_size=window_size)
        self.cache = {}
        
    def __len__(self):
        return len(self.experiences)
        
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        exp = self.experiences[idx]
        
        m1_data = exp['state'].get('M1')
        if not m1_data:
            m1_data = list(exp['state'].values())[0] if exp['state'] else [{}]*self.window_size
            
        df_state = pd.DataFrame(m1_data)
        
        delta = df_state['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df_state['rsi'] = (100 - (100 / (1 + rs))) / 100.0
        
        ema12 = df_state['close'].ewm(span=12, adjust=False).mean()
        ema26 = df_state['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_state['macd_sig'] = signal_line
        df_state['macd_hist'] = macd_line - signal_line
        
        if 'sentiment' not in df_state.columns:
            df_state['sentiment'] = exp.get('sentiment', 0.0)

        df_state.ffill(inplace=True)
        df_state.fillna(0, inplace=True)

        gaf_img = self.gaf_transformer.transform(df_state['close'].values)
        
        window_feat = df_state[['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd_sig', 'macd_hist', 'sentiment']].values
        min_vals = window_feat.min(axis=0)
        max_vals = window_feat.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        seq = (window_feat - min_vals) / range_vals
        
        result = (
            torch.tensor(gaf_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(exp['action'], dtype=torch.long),
            torch.tensor(exp['reward'], dtype=torch.float32)
        )
        # Note: In MTF mode, the RL experiences should ideally contain 27 features.
        # This implementation assumes exp['state'] has been updated by the live system to match.
        self.cache[idx] = result
        return result

async def train_rl_mode(experiences, advisor=None, db=None, epochs=50, lr=0.0001):
    """
    Reinforcement Learning update with LLM Reward Shaping.
    """
    if not experiences:
        logger.warning("No experiences for RL training.")
        return

    logger.info(f"Starting RL Training session with {len(experiences)} samples for {epochs} epochs...")
    
    params = get_best_hyperparams()
    HIDDEN_SIZE = params['hidden_size']
    WINDOW_SIZE = params['window_size']
    lr = lr if lr else params['lr']
    
    dataset = RLExperienceDataset(experiences, window_size=WINDOW_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = HybridModel(input_size=27, hidden_size=HIDDEN_SIZE).to(device)
    
    if os.path.exists("AI_Brain/weights/hybrid_model.pt"):
        try:
            model.load_state_dict(torch.load("AI_Brain/weights/hybrid_model.pt"))
            logger.info("Loaded previous Hybrid weights for RL refinement.")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Starting RL from scratch.")
    else:
        logger.warning("No existing weights found. Starting RL from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    model.train()
    
    best_policy_score = -float('inf') 
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_score = 0
        
        for gaf, seq, action, reward in loader:
            gaf, seq = gaf.to(device), seq.to(device)
            action, reward = action.to(device), reward.to(device)
            
            optimizer.zero_grad()
            
            logits, _, _ = model(gaf, seq)
            probs = torch.softmax(logits, dim=1)
            
            if len(reward) > 1:
                std = reward.std()
                norm_reward = (reward - reward.mean()) / (std + 1e-8)
            else:
                norm_reward = reward / 10.0
            
            m = torch.distributions.Categorical(probs)
            log_prob = m.log_prob(action)
            loss = - (log_prob * norm_reward).mean()
            
            with torch.no_grad():
                batch_score = (probs.gather(1, action.unsqueeze(1)).squeeze() * reward).mean()
                epoch_score += batch_score.item()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        avg_score = epoch_score / len(loader)
        
        scheduler.step(avg_score)

        if avg_score > best_policy_score:
            best_policy_score = avg_score
            if not os.path.exists("AI_Brain/weights"): os.makedirs("AI_Brain/weights")
            torch.save(model.state_dict(), "AI_Brain/weights/best_hybrid_model.pt")
            logger.info(f"New BEST RL model saved (Epoch {epoch+1}) | Score: {avg_score:.6f} | Loss: {avg_loss:.6f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"RL Epoch {epoch+1}/{epochs} | Score: {avg_score:.6f} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    try:
        model.load_state_dict(torch.load("AI_Brain/weights/best_hybrid_model.pt"))
        torch.save(model.state_dict(), "AI_Brain/weights/hybrid_model.pt")
        logger.info("RL Training Complete. Best model promoted.")
    except Exception as e:
        logger.error(f"Failed to promote best RL model: {e}")
        torch.save(model.state_dict(), "AI_Brain/weights/hybrid_model.pt")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    import argparse
    parser = argparse.ArgumentParser(description="AI Training & Backtest Pipeline")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on current weights only")
    parser.add_argument("--train", action="store_true", default=True, help="Run the iterative refinement loop (Default)")
    args, unknown = parser.parse_known_args()

    if args.backtest:
        logger.info("🧪 Running Backtest-Only Mode...")
        run_backtest_only()
    else:
        # User Requirement: 50 blocks max or performance target
        MAX_BLOCKS = 50
        BLOCK_EPOCHS = 50
        current_block = 0
        target_win_rate = 50.0
        
        logger.info("[START] Starting MTF Iterative Auto-Refinement Process...")
        
        while current_block < MAX_BLOCKS:
            logger.info(f"[BLOCK {current_block + 1}] Training Block: Epochs {(current_block * BLOCK_EPOCHS)} to {(current_block + 1) * BLOCK_EPOCHS}...")
            
            results = train_and_backtest(epochs=BLOCK_EPOCHS) 
            
            current_block += 1
            
            if not results:
                logger.error("Training failed to return results. Aborting refinement.")
                break
                
            last_fold = results[-1]
            wr = last_fold['win_rate']
            profit = last_fold['profit']
            
            logger.info(f"📊 Block {current_block} Results | WR: {wr:.2f}% | Profit: ${profit:.2f}")
            
            if wr >= target_win_rate and profit > 0:
                logger.info(f"[SUCCESS] Performance targets met at block {current_block}! Stopping refinement.")
                break
            else:
                logger.warning(f"📉 Targets not met. Continuing training for block {current_block + 1}...")
                
        logger.info(f"🏁 Refinement Process Finished. Total Blocks: {current_block}")
