import logging
import torch
import numpy as np
import math

logger = logging.getLogger("DecisionEngine")

class ConfidenceCalibrator:
    def __init__(self, db_handler):
        self.db = db_handler
        self.stats = {}

    async def update_stats(self):
        if not self.db: return
        try:
            trades = await self.db.get_rl_training_data(limit=500)
            if not trades: return
            
            brackets = {}
            for t in trades:
                conf = t.get('cnn_confidence', 0.0)
                bracket = round(math.floor(conf * 10) / 10.0, 1)
                if bracket not in brackets: brackets[bracket] = []
                brackets[bracket].append(1 if t['reward'] > 0 else 0)
                
            for b, results in brackets.items():
                self.stats[b] = sum(results) / len(results)
                
            logger.info(f"📊 Confidence Stats Updated: {self.stats}")
        except Exception as e:
            logger.error(f"Calibration update failed: {e}")

    def get_calibration_factor(self, confidence):
        bracket = round(math.floor(confidence * 10) / 10.0, 1)
        win_rate = self.stats.get(bracket, 0.5)
        
        if win_rate < 0.45:
            return 0.7
        if win_rate > 0.60:
            return 1.2
        return 1.0

class DecisionEngine:
    def __init__(self, model, calibrator=None, atr_period=14):
        self.model = model
        self.model.eval()
        self.calibrator = calibrator
        self.atr_period = atr_period

    def analyze(self, gaf_image, raw_series):
        with torch.no_grad():
            logits, trend, attn = self.model(gaf_image, raw_series)
            probs = torch.softmax(logits, dim=1)
            conf, pred_class = torch.max(probs, 1)
            
            return {
                'class': pred_class.item(),
                'conf': conf.item(),
                'trend': trend.item(),
                'attn_weights': attn
            }

    def analyze_mtf(self, mtf_sync_data, ai_mode="CONSERVATIVE", exploration_rate=0.0, symbol="Unknown"):
        """
        New MTF analysis logic for the 27-feature model.
        mtf_sync_data: (gaf_image, seq_27_features)
        """
        pattern_names = {
            0: "Neutral / Consolidation",
            1: "Bullish Momentum",
            2: "Bearish Momentum",
            3: "Hammer (Bullish Pin Bar) - Bullish Reversal",
            4: "Shooting Star (Bearish Pin Bar) - Bearish Reversal",
            5: "Bullish Engulfing - Strong Bullish Reversal",
            6: "Bearish Engulfing - Strong Bearish Reversal",
            7: "Morning Star - Bullish Reversal Pattern",
            8: "Evening Star - Bearish Reversal Pattern",
            9: "Doji - Indecision / Pivot Point"
        }
        
        gaf, seq = mtf_sync_data
        res = self.analyze(gaf, seq)
        
        # M1 features are index 0-8. RSI is 5, Sentiment is 8.
        rsi = seq[0, -1, 5].item() 
        sent = seq[0, -1, 8].item()
        m5_trend_est = seq[0, -1, 12].item() # Example: m5_close or similar (not exact but for logging)
        
        signal = {'action': 'HOLD', 'confidence': 0.0, 'report': ''}
        
        # RL Exploration
        if np.random.random() < exploration_rate:
            action = 'BUY' if np.random.random() > 0.5 else 'SELL'
            logger.info(f"🎲 RL Exploration Triggered: {action}")
            pattern_text = pattern_names.get(res['class'], "Unknown")
            return {
                'action': action, 'confidence': 0.5, 'reason': 'RL Exploration',
                'report': f"🔍 **Exploration Mode**: ลองทดสอบคำสั่ง {action} (Exploration Rate: {exploration_rate})",
                'analyst_metadata': {'pattern': pattern_text, 'confidence': res['conf']},
                'raw_cnn_class': res['class'], 'raw_lstm_trend': res['trend'], 'ai_mode': ai_mode
            }

        # Strategy Logic
        is_bullish = res['class'] in [1, 3, 5, 7] and res['trend'] > 0
        is_bearish = res['class'] in [2, 4, 6, 8] and res['trend'] < 0
        
        # Exhaustion Filters
        is_exhausted = False
        exhaustion_reason = ""
        if (is_bullish and rsi > 0.85) or (is_bearish and rsi < 0.15):
            is_exhausted = True
            exhaustion_reason = f"Extreme RSI Exhaustion ({rsi:.2f})"
        elif res['class'] == 9:
            is_exhausted = True
            exhaustion_reason = "Doji Indecision"
        
        final_conf = res['conf']
        min_conf = 0.6 if ai_mode == "CONSERVATIVE" else 0.5
        
        # Apply Filters directly to confidence
        if is_bullish and (rsi > 0.75 or sent < -0.3): final_conf *= 0.7
        if is_bearish and (rsi < 0.25 or sent > 0.3): final_conf *= 0.7

        if self.calibrator:
            final_conf *= self.calibrator.get_calibration_factor(final_conf)

        pattern_text = pattern_names.get(res['class'], "Unknown")
        analyst_data = {
            'pattern': pattern_text, 'confidence': final_conf,
            'rsi': rsi, 'sent': sent, 'trend': res['trend']
        }

        if is_bullish and final_conf >= min_conf:
            signal = {
                'action': 'BUY', 'confidence': final_conf, 
                'reason': f'MTF Bullish {ai_mode}', 'analyst_metadata': analyst_data,
                'raw_cnn_class': res['class'], 'raw_lstm_trend': res['trend'],
                'ai_mode': ai_mode, 'is_exhausted': is_exhausted, 'exhaustion_reason': exhaustion_reason
            }
        elif is_bearish and final_conf >= min_conf:
            signal = {
                'action': 'SELL', 'confidence': final_conf, 
                'reason': f'MTF Bearish {ai_mode}', 'analyst_metadata': analyst_data,
                'raw_cnn_class': res['class'], 'raw_lstm_trend': res['trend'],
                'ai_mode': ai_mode, 'is_exhausted': is_exhausted, 'exhaustion_reason': exhaustion_reason
            }
            
        return signal

            pattern_text = pattern_names.get(m1['class'], "Unknown")
            is_uptrend = m1['trend'] > 0
            future_outlook = "ขาขึ้นแข็งแกร่ง" if is_uptrend else "แนวโน้มขาลง"

            analyst_data = {
                'pattern': pattern_text,
                'future_outlook': future_outlook,
                'is_uptrend': is_uptrend,
                'confidence': final_conf,
                'm1_conf': m1['conf'],
                'm1_rsi': m1.get('rsi', 0.5),
                'm1_sentiment': m1.get('sentiment', 0.0),
                'm5_trend': m5['trend'] if m5 else 0,
                'h1_trend': h1['trend'] if h1 else 0
            }
            
            rsi = m1.get('rsi', 0.5) 
            sent = m1.get('sentiment', 0.0)
            
            if is_bullish:
                if rsi > 0.75:
                    final_conf *= 0.7
                    logger.info(f"🛡️ RSI Bull Filter: {symbol} RSI {rsi:.2f} is Overbought. Reducing confidence.")
                if sent < -0.4:
                    final_conf *= 0.5
                    logger.info(f"🛡️ Sentiment Bull Filter: {symbol} Sentiment {sent:.2f} is Negative. Reducing confidence.")
            
            elif is_bearish:
                if rsi < 0.25:
                    final_conf *= 0.7
                    logger.info(f"🛡️ RSI Bear Filter: {symbol} RSI {rsi:.2f} is Oversold. Reducing confidence.")
                if sent > 0.4:
                    final_conf *= 0.5
                    logger.info(f"🛡️ Sentiment Bear Filter: {symbol} Sentiment {sent:.2f} is Positive. Reducing confidence.")

            if self.calibrator:
                c_factor = self.calibrator.get_calibration_factor(final_conf)
                if c_factor != 1.0:
                    old_conf = final_conf
                    final_conf *= c_factor
                    logger.info(f"⚖️ Confidence Calibrated: {symbol} {old_conf:.2f} -> {final_conf:.2f} (Factor: {c_factor})")

            trend_val = m1['trend']
            lstm_conf = min(abs(trend_val) / 50.0, 1.0)

            if is_bullish and bull_confirmed and final_conf >= min_conf:
                signal = {
                    'action': 'BUY', 
                    'confidence': final_conf, 
                    'reason': f'MTF Bullish {ai_mode} (W-Conf: {final_conf:.2f})',
                    'analyst_metadata': analyst_data,
                    'raw_cnn_class': m1['class'],  
                    'raw_lstm_trend': m1['trend'],
                    'raw_lstm_conf': lstm_conf,
                    'ai_mode': ai_mode,
                    'is_exhausted': is_exhausted,
                    'exhaustion_reason': exhaustion_reason
                }
            elif is_bearish and bear_confirmed and final_conf >= min_conf:
                signal = {
                    'action': 'SELL', 
                    'confidence': final_conf, 
                    'reason': f'MTF Bearish {ai_mode} (W-Conf: {final_conf:.2f})',
                    'analyst_metadata': analyst_data,
                    'raw_cnn_class': m1['class'],  
                    'raw_lstm_trend': m1['trend'],
                    'raw_lstm_conf': lstm_conf,
                    'ai_mode': ai_mode,
                    'is_exhausted': is_exhausted,
                    'exhaustion_reason': exhaustion_reason
                }
            else:
                if final_conf < min_conf:
                    logger.info(f"⚠️ Signal Skipped: {symbol} Pattern {pattern_text} | W-Conf {final_conf:.2f} < {min_conf}")
                signal['analyst_metadata'] = analyst_data
                signal['raw_cnn_class'] = m1['class']
                signal['raw_lstm_trend'] = m1['trend']
                signal['raw_lstm_conf'] = lstm_conf
                signal['is_exhausted'] = is_exhausted
                signal['exhaustion_reason'] = exhaustion_reason
                
        return signal

    def extract_rl_features(self, gaf_image, raw_series):
        with torch.no_grad():
            cnn_logits = self.cnn(gaf_image)
            lstm_pred = self.lstm(raw_series)
            state = torch.cat([cnn_logits.flatten(), lstm_pred.flatten()])
            return state.cpu().numpy()

class RiskManager:
    def __init__(self, risk_per_trade=0.01):
        self.risk_per_trade = risk_per_trade 

    def calculate_sl_tp(self, symbol, action, entry_price, atr=None, stop_loss_pips=20, reward_ratio=2.0, point=0.00001, tick_size=0.00001, digits=5, commission_buffer_pips=2.0, confidence=None):
        s_upper = symbol.upper()
        if "XAU" in s_upper or "GOLD" in s_upper:
            pip_value = 0.1
        elif "JPY" in s_upper or digits in [2, 3]:
            pip_value = 0.01
        elif digits in [0, 1]:
            pip_value = 1.0
        else:
            pip_value = 10 * point
            
        if atr is not None and atr > 0:
            sl_offset = atr * 1.5
        else:
            sl_offset = stop_loss_pips * pip_value
            
        if confidence is not None:
            conf_clamped = max(0.55, min(0.85, confidence))
            reward_ratio = 1.0 + ((conf_clamped - 0.55) / 0.30) * 1.0
        
        tp_offset = (sl_offset * reward_ratio) + (commission_buffer_pips * pip_value)
        
        if action.upper() == "BUY":
            sl = entry_price - sl_offset
            tp = entry_price + tp_offset
        else:
            sl = entry_price + sl_offset
            tp = entry_price - tp_offset
            
        def normalize(price):
            return round(price / tick_size) * tick_size
            
        return normalize(sl), normalize(tp)

    def calculate_lot_size(self, account_equity, sl_price_distance, confidence=0.5, tick_value=1.0, tick_size=0.00001):
        if sl_price_distance <= 0: return 0.01
        
        risk_amount = account_equity * self.risk_per_trade
        
        confidence_factor = max(0.5, min(1.5, confidence / 0.7)) 
        risk_amount *= confidence_factor
        
        ticks_at_risk = sl_price_distance / tick_size
        dollar_risk_per_lot = ticks_at_risk * tick_value
        
        if dollar_risk_per_lot <= 0: return 0.01
        
        raw_lots = risk_amount / dollar_risk_per_lot
        
        return round(max(0.01, raw_lots), 2)

    def calculate_max_affordable_lots(self, symbol, action, free_margin, mt5_module=None):
        if mt5_module is None:
            import MetaTrader5 as mt5_module

        order_type = mt5_module.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5_module.ORDER_TYPE_SELL
        
        margin_per_lot = mt5_module.order_calc_margin(order_type, symbol, 1.0, mt5_module.symbol_info_tick(symbol).ask)
        
        if margin_per_lot is None or margin_per_lot <= 0:
            logger.warning(f"Could not calculate margin for {symbol}. Falling back to 0.01 lots.")
            return 0.01

        affordable_margin = free_margin * 0.9
        max_lots = affordable_margin / margin_per_lot
        
        max_lots = math.floor(max_lots * 100) / 100.0
        
        symbol_info = mt5_module.symbol_info(symbol)
        if symbol_info:
            max_lots = min(max_lots, symbol_info.volume_max)
            max_lots = max(max_lots, symbol_info.volume_min)
            
        return round(max_lots, 2)

class ConsensusEngine:
    def __init__(self, explorer_brain, guardian_brain, calibrator=None):
        self.explorer = explorer_brain
        self.guardian = guardian_brain
        self.calibrator = calibrator
        
    def analyze_mtf_consensus(self, mtf_data, exploration_rate=0.0, symbol="Unknown"):
        explorer_engine = DecisionEngine(self.explorer, self.calibrator)
        guardian_engine = DecisionEngine(self.guardian, self.calibrator)
        
        exp_signal = explorer_engine.analyze_mtf(mtf_data, ai_mode="EXPLORER", exploration_rate=exploration_rate, symbol=symbol)
        guard_signal = guardian_engine.analyze_mtf(mtf_data, ai_mode="CONSERVATIVE", exploration_rate=0.0, symbol=symbol)
        
        final_signal = exp_signal.copy()
        
        if exp_signal['action'] != 'HOLD':
            guard_conf = guard_signal.get('confidence', 0.0)
            
            mismatch = (exp_signal['action'] == 'BUY' and guard_signal['action'] == 'SELL') or \
                       (exp_signal['action'] == 'SELL' and guard_signal['action'] == 'BUY')
            
            GUARDIAN_THRESHOLD = 0.55
            
            is_vetoed = False
            veto_reason = ""
            
            if mismatch:
                is_vetoed = True
                veto_reason = "Directional Mismatch (Guardian disagrees)"
            elif guard_signal['action'] == 'HOLD' and guard_conf < GUARDIAN_THRESHOLD:
                is_vetoed = True
                veto_reason = f"Guardian Veto: High Uncertainty ({guard_conf:.2f} < {GUARDIAN_THRESHOLD})"
            elif guard_signal['action'] != exp_signal['action'] and guard_signal['action'] != 'HOLD':
                is_vetoed = True
                veto_reason = "Guardian Contradiction"

            if is_vetoed:
                logger.info(f"🛡️ CONSENSUS VETO for {symbol}: {exp_signal['action']} rejected. Reason: {veto_reason}")
                final_signal['action'] = 'HOLD'
                final_signal['reason'] = f"Vetoed: {veto_reason}"
                final_signal['report'] = f"🛡️ **Guardian Veto**: ปฏิเสธ {exp_signal['action']} เนื่องจาก {veto_reason}"
            else:
                final_signal['reason'] = f"Consensus Approved ({exp_signal['action']})"
                final_signal['report'] = f"🤝 **Consensus Approved**: ทั้ง 2 ขุมพลังเห็นพ้อง (Conf: {exp_signal['confidence']:.2f})"

        return final_signal
