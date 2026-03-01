import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger("Preprocessor")

class CandleAggregator:
    def __init__(self, timeframe_seconds=3600):
        self.tf_seconds = timeframe_seconds
        self.current_candle = None
        self.last_closed_candle = None
        
    def add_tick(self, tick_data):
        """
        Ingest a tick and update current candle.
        Returns: completed_candle (dict) or None
        tick_data: {'bid': float, 'time': int (ms timestamp)}
        """
        price = tick_data['bid']
        ts_ms = tick_data.get('time', int(datetime.now().timestamp() * 1000))
        ts_sec = ts_ms / 1000.0
        
        candle_start_time = int(ts_sec // self.tf_seconds) * self.tf_seconds
        
        completed_candle = None
        if self.current_candle is None:
            self._init_candle(candle_start_time, price, tick_data)
        elif candle_start_time > self.current_candle['time']:
            completed_candle = self.current_candle.copy()
            self.last_closed_candle = completed_candle
            self._init_candle(candle_start_time, price, tick_data)
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['tick_volume'] += 1
            if 'sentiment' in tick_data:
                self.current_candle['sentiment'] = tick_data['sentiment']
            
        return completed_candle

    def _init_candle(self, start_time, price, tick_data=None):
        self.current_candle = {
            'time': start_time,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'tick_volume': 1,
            'sentiment': tick_data.get('sentiment', 0.0) if tick_data else 0.0
        }
    
    def get_current_candle(self):
        return self.current_candle

    def get_last_closed_candle(self):
        return self.last_closed_candle
class SlidingWindowBuffer:
    def __init__(self, window_size=32, features=['open', 'high', 'low', 'close', 'tick_volume', 'sentiment']):
        self.window_size = window_size
        self.features = features
        self.data = pd.DataFrame(columns=features + ['rsi', 'macd_sig', 'macd_hist', 'atr'])

    def _calculate_indicators(self):
        if len(self.data) < 26: return
        
        close = self.data['close']
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.data['rsi'] = (100 - (100 / (1 + rs))) / 100.0
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        self.data['macd_sig'] = signal_line
        self.data['macd_hist'] = macd_line - signal_line
        
        prev_close = close.shift(1)
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - prev_close)
        tr3 = abs(self.data['low'] - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=14).mean()
        
        self.data.ffill(inplace=True)
        self.data.fillna(0, inplace=True)

    def add_candle(self, candle_dict):
        filtered = {k: candle_dict[k] for k in self.features if k in candle_dict}
        new_row = pd.DataFrame([filtered])
        
        if not new_row.empty:
            if self.data.empty:
                self.data = new_row
            else:
                self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        if len(self.data) > self.window_size:
            self.data = self.data.iloc[-self.window_size:]
        
        if len(self.data) >= 26:
            self._calculate_indicators()
            
    def get_data(self):
        return self.data.copy()
    
    def is_full(self):
        return len(self.data) >= self.window_size

class MTFManager:
    def __init__(self, timeframes=[60, 300, 3600], window_size=32):
        self.timeframes = timeframes
        self.aggregators = {tf: CandleAggregator(timeframe_seconds=tf) for tf in timeframes}
        self.buffers = {tf: SlidingWindowBuffer(window_size=window_size) for tf in timeframes}
        self.ready_signals = {tf: False for tf in timeframes}

    def add_tick(self, tick_data):
        closed_tfs = []
        for tf, agg in self.aggregators.items():
            new_candle = agg.add_tick(tick_data)
            if new_candle:
                self.buffers[tf].add_candle(new_candle)
                closed_tfs.append(tf)
                
        return closed_tfs

    def get_data_for_tf(self, tf):
        return self.buffers[tf].get_data()

    def is_tf_ready(self, tf):
        return self.buffers[tf].is_full()

class GAFTransformer:
    def __init__(self, image_size=32):
        self.image_size = image_size

    def transform(self, series):
        min_val = np.min(series)
        max_val = np.max(series)
        
        if max_val == min_val:
            scaled_series = np.zeros_like(series)
        else:
            scaled_series = ((series - min_val) / (max_val - min_val)) * 2 - 1
            
        scaled_series = np.clip(scaled_series, -1.0, 1.0)
        phi = np.arccos(scaled_series)
        
        phi_i = phi.reshape(-1, 1)
        phi_j = phi.reshape(1, -1)
        
        gaf_image = np.cos(phi_i + phi_j)
        return gaf_image
