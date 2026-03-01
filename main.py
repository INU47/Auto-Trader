import asyncio
import time
import httpx
import logging
import json
import torch
import datetime
import os
import MetaTrader5 as mt5
from AI_Brain.preprocessor import SlidingWindowBuffer, GAFTransformer, CandleAggregator, MTFManager
from AI_Brain.models import HybridModel, get_best_hyperparams
from AI_Brain.sentiment_analyzer import SentimentAnalyzer
from AI_Brain.decision_engine import DecisionEngine, RiskManager
from Database.db_handler import DBHandler
from ZMQ_Bridge.telegram_notifier import TelegramNotifier
import traceback
import subprocess
from AI_Brain.llm_advisor import LLMRewardAdvisor
from AI_Brain.training_pipeline import train_rl_mode

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if root_logger.hasHandlers():
    root_logger.handlers.clear()

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

file_handler = logging.FileHandler("quant_system.log", mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

err_handler = logging.FileHandler("errors.log", mode='a', encoding='utf-8')
err_handler.setFormatter(formatter)
err_handler.setLevel(logging.ERROR)
root_logger.addHandler(err_handler)

trade_all_handler = logging.FileHandler("trades_all.log", mode='a', encoding='utf-8')
trade_all_handler.setFormatter(formatter)
trade_logger = logging.getLogger("Trade")
trade_logger.setLevel(logging.INFO)
trade_logger.addHandler(trade_all_handler)

trade_exe_handler = logging.FileHandler("trades_executed.log", mode='a', encoding='utf-8')
trade_exe_handler.setFormatter(formatter)
trade_exe_logger = logging.getLogger("Trade.Executed")
trade_exe_logger.setLevel(logging.INFO)
trade_exe_logger.addHandler(trade_exe_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("Main")
logger.info("="*50)
logger.info("--- SYSTEM STARTUP ---")
logger.info("="*50)

dashboard_client = None

async def post_to_dashboard(data):
    """Sends tick, candle, or signal data to the web dashboard server using shared client"""
    global dashboard_client
    if dashboard_client is None:
        dashboard_client = httpx.AsyncClient(timeout=2.0)
    
    try:
        url = "http://127.0.0.1:8000/push"
        resp = await dashboard_client.post(url, json=data)
        if resp.status_code != 200:
            logger.warning(f"Dashboard push failed: {resp.status_code}")
    except Exception:
        pass

def get_filling_type(symbol):
    """Dynamically detects the supported filling mode for a symbol/broker"""
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return mt5.ORDER_FILLING_IOC
        
    filling_mode = symbol_info.filling_mode
    if filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def get_pip_size(symbol):
    info = mt5.symbol_info(symbol)
    if not info: return 0.0001
    
    s_upper = symbol.upper()
    if "XAU" in s_upper or "GOLD" in s_upper: return 0.1
    if "JPY" in s_upper: return 0.01
    
    if info.digits in [2, 3]: return 0.01
    if info.digits in [0, 1]: return 1.0 
    
    return 0.0001

def is_market_open(symbol):
    """
    Checks if the market for a specific symbol is open for full trading.
    Returns: (bool, message)
    """
    terminal = mt5.terminal_info()
    if not terminal or not terminal.trade_allowed:
        return False, "Broker/Terminal trade permission denied (Global)."

    info = mt5.symbol_info(symbol)
    if info is None:
        return False, f"Symbol {symbol} info not found."
    
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return False, "Market is completely CLOSED for this symbol."
    elif info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        return False, "Market is CLOSE-ONLY (No new entries allowed)."
        
    now = datetime.datetime.now()
    if now.weekday() == 4:
        if now.hour >= 23:
            return False, "Friday Night Protection: Market nearing weekly close. Volatility too high."
    elif now.weekday() == 5:
        return False, "Market is closed for the weekend."
    
    if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        return False, f"Market restricted (Trade Mode: {info.trade_mode})."

    return True, "Market is open."

async def execute_mt5_order(user_id, symbol, action, volume, sl=0, tp=0, notifier=None, signal_data=None, db=None):
    """Executes a trade on MT5 for a specific user and logs it globally."""
    
    try:
        if not mt5.terminal_info():
            msg = "❌ **MT5 Error**: Terminal not connected or initialized."
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None

        if not mt5.symbol_select(symbol, True):
            msg = f"❌ **MT5 Error**: Symbol {symbol} not found or cannot be selected."
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None

        is_open, mkt_msg = is_market_open(symbol)
        if not is_open:
            msg = f"🕰️ **Market Closed**: {symbol}\n📝 {mkt_msg}"
            logger.warning(msg)
            if notifier: await notifier.send_message(msg)
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            msg = f"❌ **MT5 Error**: Could not get price for {symbol}."
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None
            
        print(f"DEBUG: Preparing order for {symbol} {action} {volume}")
        order_type = mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action.upper() == "BUY" else tick.bid
        
        filling_type = get_filling_type(symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl), 
            "tp": float(tp),
            "magic": 123456,
            "comment": f"QuantAI_{user_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"❌ **Order Failed**: {result.comment if result else 'None'} (Code: {result.retcode if result else 'N/A'})"
            logger.error(msg)
            trade_logger.error(f"{symbol} execution failed: {result.comment if result else 'None'}")
            if notifier: await notifier.send_message(msg)
            return None
        else:
            reason = signal_data.get('reason', 'N/A') if signal_data else 'Manual/Unknown'
            msg = (
                f"✅ **Order Executed**\n\n"
                f"🎫 **Ticket**: {result.order}\n"
                f"🎯 **Action**: {action}\n"
                f"📦 **Volume**: {volume}\n"
                f"💰 **Price**: {result.price}\n"
                f"🛑 **SL**: {sl}\n"
                f"🏁 **TP**: {tp}\n"
                f"📝 **Reason**: {reason}"
            )
            logger.info(msg)
            trade_exe_logger.info(f"ORDER_{result.order}: {action} {volume} {symbol} @ {result.price} | SL: {sl} | TP: {tp} | Reason: {reason}")
            if notifier: await notifier.send_message(msg)
        
        if db and signal_data:
            await asyncio.sleep(1.0)
            deal_ticket = result.deal
            position_ticket = result.order
            
            history_deal = mt5.history_deals_get(ticket=deal_ticket)
            if history_deal:
                position_ticket = history_deal[0].position_id
                logger.info(f"🔗 Linked Deal {deal_ticket} to Position {position_ticket}")
            else:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for p in positions:
                        if p.magic == 123456 and abs(p.volume - float(volume)) < 0.0001:
                            position_ticket = p.ticket
                            break
            
            logger.info(f"📥 Logging Entry: {symbol} | Ticket: {position_ticket} (User: {user_id})")
            await db.log_trade_entry(user_id, symbol, action, volume, result.price, signal_data, ticket=position_ticket)
            
        return result
    except Exception as e:
        logger.error(f"Failed to execute {action} for {symbol}: {e}")
        if notifier: await notifier.send_message(f"⚠️ **Execution Crash**: {str(e)}")
        return None

async def close_all_positions(user_id, symbol, action_type=None, notifier=None, db=None):
    """Closes all open positions for a symbol and a specific user."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    for p in positions:
        if p.magic != 123456:
            continue
            
        if action_type is not None and p.type != action_type:
            continue

        tick = mt5.symbol_info_tick(symbol)
        order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": p.ticket,
            "symbol": symbol,
            "volume": p.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Close_{user_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_type(symbol),
        }

        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            await asyncio.sleep(0.5)
            deals = mt5.history_deals_get(position=p.ticket)
            total_profit = sum([d.profit + d.swap + d.commission for d in deals]) if deals else 0.0
            
            msg = (
                f"🎯 **Position Closed**\n\n"
                f"🎫 **Ticket**: {p.ticket}\n"
                f"📌 **Symbol**: {symbol}\n"
                f"📦 **Volume**: {p.volume}\n"
                f"💰 **P/L**: ${total_profit:.2f}\n"
                f"🚪 **Exit Price**: {res.price}\n"
                f"📝 **Reason**: { 'REVERSAL' if action_type is not None else 'MANUAL' }"
            )
            logger.info(msg + f" (User: {user_id})")
            trade_exe_logger.info(f"CLOSE_{'REVERSAL' if action_type is not None else 'MANUAL'}: {symbol} | Ticket: {p.ticket} | Vol: {p.volume} | P/L: ${total_profit:.2f} (User: {user_id})")
            if notifier: await notifier.send_message(msg)
            
            if db:
                await db.log_trade_exit_by_ticket(user_id, p.ticket, res.price, total_profit, total_profit, reason="REVERSAL" if action_type is not None else "MANUAL")
        else:
            logger.error(f"Failed to close {p.ticket} for User {user_id}: {res.comment}")

async def manage_trailing_stop(symbol, activation_pips=10, trailing_pips=10, db=None, mtf_managers=None):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return
        
    point = symbol_info.point
    pip_size = get_pip_size(symbol)
    
    mgr = mtf_managers.get(symbol) if mtf_managers else None
    current_atr = 0.0002
    if mgr:
        df = mgr.get_data_for_tf(60)
        if not df.empty and 'atr' in df.columns:
            current_atr = df['atr'].iloc[-1]
            
    trail_offset = max(trailing_pips * pip_size, current_atr * 1.5)
    activation_offset = activation_pips * pip_size
    be_buffer = 1.0 * pip_size
    
    for p in positions:
        if p.magic != 123456:
            continue
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        
        new_sl = None
        if p.type == mt5.POSITION_TYPE_BUY:
            profit_distance = tick.bid - p.price_open
            
            if profit_distance >= activation_offset:
                be_level = p.price_open + be_buffer
                if p.sl < be_level:
                    new_sl = be_level
                    logger.info(f"🛡️ BE Triggered: {symbol} Ticket {p.ticket} moved to BE.")
            
            if profit_distance >= trail_offset:
                candidate_trail_sl = tick.bid - trail_offset
                if candidate_trail_sl > (new_sl if new_sl else p.sl) + (point * 10):
                    new_sl = candidate_trail_sl

        elif p.type == mt5.POSITION_TYPE_SELL:
            profit_distance = p.price_open - tick.ask
            
            if profit_distance >= activation_offset:
                be_level = p.price_open - be_buffer
                if p.sl == 0 or p.sl > be_level:
                    new_sl = be_level
                    logger.info(f"🛡️ BE Triggered: {symbol} Ticket {p.ticket} moved to BE.")
            
            if profit_distance >= trail_offset:
                candidate_trail_sl = tick.ask + trail_offset
                if p.sl != 0 and (candidate_trail_sl < (new_sl if new_sl else p.sl) - (point * 10)):
                    new_sl = candidate_trail_sl
                elif p.sl == 0:
                    new_sl = candidate_trail_sl

        if new_sl is not None:
            new_sl = round(new_sl / symbol_info.trade_tick_size) * symbol_info.trade_tick_size
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": symbol,
                "sl": float(new_sl),
                "tp": float(p.tp),
            }
            res = mt5.order_send(request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.debug(f"📈 SL Updated: {symbol} Ticket {p.ticket} | New SL: {new_sl:.5f}")
            else:
                logger.warning(f"⚠️ SL Update Failed: {symbol} Ticket {p.ticket} | {res.comment if res else 'No response'}")

async def sync_historical_data(managers_dict, db, window_size=32):
    """Fetches history from Local DB (Optimized for speed). Fallback to MT5 if empty."""
    DAYS = 7
    logger.info(f"🔄 Syncing History: Fetching last {DAYS} days of data from Database...")
    history_batch = []
    
    config_path = "Config/mt5_config.json"
    if not os.path.exists(config_path): return []
    with open(config_path, "r") as f: config = json.load(f)
    symbols = config.get('symbols', ["EURUSD"])
    
    for symbol in symbols:
        if symbol not in managers_dict:
            managers_dict[symbol] = MTFManager(timeframes=[60, 300, 3600], window_size=window_size)
            
        for tf_sec, tf_label in {60: 'M1', 300: 'M5', 3600: 'H1'}.items():
            await db.ensure_data_continuity(symbol, tf_label, target_candles=5000)
            
            candles = await db.get_candles(symbol, tf_label, days=DAYS)
            for i, candle in enumerate(candles):
                if len(candles) - i <= 500:
                    managers_dict[symbol].buffers[tf_sec].add_candle(candle)
                
                history_batch.append({"type": "candle", **candle})
            
            logger.info(f"  > Synced {len(candles)} candles for {symbol} {tf_label}")

    return history_batch

async def run_trading_engine():
    """Main Orchestrator for Multi-User Trading System."""
    logger.info("Starting Multi-User Trading Orchestrator...")
    
    db = DBHandler()
    await db.connect()
    
    with open("Config/server_config.json", "r") as f: srv_config = json.load(f)
    
    notifier = TelegramNotifier()
    gaf_transformer = GAFTransformer()
    sentiment_analyzer = SentimentAnalyzer()
    
    hparams = get_best_hyperparams()
    hidden_size = hparams['hidden_size']
    window_size = hparams['window_size']
    dropout = hparams['dropout']
    logger.info(f"✨ Initializing System with Optimized Hyperparameters: {hparams}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    explorer_model = HybridModel(input_size=27, hidden_size=hidden_size).to(device)
    guardian_model = HybridModel(input_size=27, hidden_size=hidden_size).to(device)
    
    exp_path = "AI_Brain/weights/weights_explorer.pt"
    guard_path = "AI_Brain/weights/weights_guardian.pt"
    fallback_path = "AI_Brain/weights/best_hybrid_model.pt"
    
    if os.path.exists(exp_path):
        explorer_model.load_state_dict(torch.load(exp_path, map_location=device))
        logger.info("🚀 Explorer Brain Loaded.")
    elif os.path.exists(fallback_path):
        explorer_model.load_state_dict(torch.load(fallback_path, map_location=device))
        logger.info("🚀 Explorer Brain Loaded (Fallback).")
        
    if os.path.exists(guard_path):
        guardian_model.load_state_dict(torch.load(guard_path, map_location=device))
        logger.info("🛡️ Guardian Brain Loaded.")
    elif os.path.exists(fallback_path):
        guardian_model.load_state_dict(torch.load(fallback_path, map_location=device))
        logger.info("🛡️ Guardian Brain Loaded (Fallback).")

    from AI_Brain.decision_engine import ConsensusEngine, ConfidenceCalibrator, RiskManager
    calibrator = ConfidenceCalibrator(db)
    await calibrator.update_stats()
    
    consensus_engine = ConsensusEngine(explorer_model, guardian_model, calibrator=calibrator)
    risk_manager = RiskManager(risk_per_trade=0.02)
    advisor = LLMRewardAdvisor(
        api_key_pool=srv_config["gemini_api_key_pool"],
        model_pool=srv_config.get("gemini_model_pool")
    ) if srv_config.get("llm_enabled") else None
    
    mtf_managers = {}
    active_workers = {} 
    
    async def command_worker():
        """Handles global Telegram commands."""
        while True:
            try:
                cmds = await notifier.check_commands()
                for cmd in cmds:
                    logger.info(f"📩 Telegram Command received: {cmd}")
                    if cmd == "/info":
                        users = await db.get_active_users()
                        total_trades = await db.count_total_trades()
                        open_trades = await db.count_open_trades()
                        closed_trades = await db.count_closed_trades()
                        
                        llm_status = "✅ ENABLED" if advisor else "❌ DISABLED"
                        llm_health = "🟢 HEALTHY" if advisor and advisor.is_healthy() else "🔴 UNHEALTHY"
                        
                        ai_mode = srv_config.get("ai_mode", "EXPLORER")
                        exploration_rate = srv_config.get("exploration_rate", 0.2)
                        
                        msg = f"""📊 **SYSTEM INFO**

                                🔧 **Status**: {'ACTIVE' if state.get("trading_enabled", True) else 'PAUSED'}
                                👥 **Users**: {len(users)} active
                                ⚙️ **Workers**: {len(active_workers)} running

                                📈 **Trade Statistics**
                                ├─ Total: {total_trades}
                                ├─ Open: {open_trades}
                                └─ Closed: {closed_trades}

                                🧠 **AI Intelligence**
                                ├─ Mode: {ai_mode} (Consensus Ensemble)
                                ├─ Explorer: 🟢 ACTIVE
                                ├─ Guardian: 🛡️ ACTIVE
                                ├─ Exploration: {exploration_rate * 100:.0f}%
                                └─ LLM: {llm_status} {llm_health if advisor else ""}

                                💾 **Database**: {'🟢 CONNECTED' if await db.is_healthy() else '🔴 DISCONNECTED'}
                                """
                        
                        await notifier.send_message(msg)
                    
                    elif cmd == "/on":
                        state["trading_enabled"] = True
                        await notifier.send_message("✅ **Trading Enabled**: The system is now active and monitoring for signals.")
                        logger.info("System ENABLED via Telegram.")
                        
                    elif cmd == "/off":
                        state["trading_enabled"] = False
                        await notifier.send_message("🛑 **Trading Disabled**: New trade execution is now paused.")
                        logger.info("System DISABLED via Telegram.")
                        
                    elif cmd == "/history":
                        trades = await db.get_rl_training_data(limit=5)
                        if not trades:
                            await notifier.send_message("📊 **History**: No recent trades found in database.")
                        else:
                            history_msg = "📊 **Recent History**\n\n"
                            for t in trades:
                                icon = "🟢" if t['reward'] > 0 else "🔴"
                                history_msg += f"{icon} {t['symbol']} | {t['action']} | PnL: ${t['reward']:.2f}\n"
                            await notifier.send_message(history_msg)

            except Exception as e: logger.error(f"Command worker error: {e}")
            await asyncio.sleep(5)

    async def db_monitor_worker():
        """Checks DB health every 5 minutes and alerts if down"""
        while True:
            await asyncio.sleep(300)
            if not await db.is_healthy():
                await notifier.send_message("🚨 **DATABASE ALERT**: Connection lost! RL data collection is paused.")
                logger.error("Database health check failed.")

    state = {
        "trading_enabled": True,
        "last_rl_order_count": 0,
        "main_menu": notifier.get_main_menu()
    }

    async def retraining_worker():
        """Shared RL retraining logic."""
        persistent_count = await db.get_metadata("last_retrain_trade_count")
        
        if persistent_count is not None:
            state["last_rl_order_count"] = int(persistent_count)
            logger.info(f"Retraining worker started. Loaded persistent count: {state['last_rl_order_count']}")
        else:
            current_closed = await db.count_closed_trades()
            state["last_rl_order_count"] = current_closed
            await db.set_metadata("last_retrain_trade_count", str(current_closed))
            logger.info(f"Retraining worker started. Baseline set to current closed trades: {current_closed}")
        
        while True:
            await asyncio.sleep(600)
            
            try:
                current_closed = await db.count_closed_trades()
                new_trades = current_closed - state["last_rl_order_count"]
                
                if new_trades >= 30:
                    logger.info(f"🕒 RL Trigger: {new_trades} new trades detected. Starting RL retraining...")
                    await notifier.send_message(f"🧠 **RL Retraining Started**\n\nLearning from {new_trades} new trade results...")
                    
                    experiences = await db.get_rl_training_data(limit=500) 
                    
                    if experiences:
                        await train_rl_mode(experiences, advisor=advisor, db=db)
                        
                        checkpoint = torch.load("AI_Brain/weights/hybrid_model.pt", map_location=device)
                        explorer_model.load_state_dict(checkpoint)
                        guardian_model.load_state_dict(checkpoint) 
                        
                        state["last_rl_order_count"] = current_closed
                        
                        await db.set_metadata("last_retrain_trade_count", str(current_closed))
                        
                        logger.info("♻️ System updated with new RL weights and persistent count.")
                        
                        backtest_process = await asyncio.create_subprocess_exec(
                            "python", "-c",
                            "from AI_Brain.training_pipeline import run_backtest_only; run_backtest_only()",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=os.getcwd()
                        )
                        await backtest_process.communicate()
                        
                        try:
                            with open("AI_Brain/performance_log.txt", 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                recent_report = ''.join(lines[-12:]) if len(lines) >= 12 else ''.join(lines)
                                await notifier.send_message(f"🧠 **RL Learning Complete**\nModel refined based on real profit/loss data.\n\n```\n{recent_report}\n```")
                        except Exception as e:
                            logger.error(f"Failed to read backtest log: {e}")
                            await notifier.send_message("🧠 **RL Learning Complete**: Bot has successfully refined its model based on recent trade performance.")
            except Exception as e:
                logger.error(f"⚠️ Retraining Exception: {e}")

    async def reward_shaping_worker():
        """Shared LLM reward shaping logic."""
        logger.info("Reward shaping worker started.")
        while True:
            await asyncio.sleep(300)
            if not advisor: continue
            
            try:
                unrated = await db.get_unrated_trades(limit=5)
                if not unrated: continue
                
                logger.info(f"LLM Batch: Shaping rewards for {len(unrated)} trades...")
                for exp in unrated:
                    score, reason = await advisor.get_quality_score(exp)
                    
                    raw_reward = exp['reward']
                    
                    if raw_reward >= 0:
                        adjusted_reward = raw_reward * (score / 50.0)
                    else:
                        penalty_factor = (150 - score) / 50.0
                        adjusted_reward = raw_reward * penalty_factor
                    
                    await db.log_llm_reward(exp['id'], score, reason, adjusted_reward)
                    await asyncio.sleep(2)
                
                logger.info("LLM Batch: Reward shaping successful.")
            except Exception as e:
                logger.error(f"Reward shaping worker error: {e}")

    async def position_monitor_worker():
        """
        Background worker to monitor closed positions and sync them to DB.
        If a position is no longer in MT5 open positions but is in DB as OPEN,
        it means it was closed (TP/SL/Manual). We need to get the deal details.
        """
        logger.info("Position monitor worker started.")
        known_tickets = set()
        retry_counts = {}
        max_retries = 3

        while True:
            try:
                await asyncio.sleep(15)
                users = await db.get_active_users()
                
                for user in users:
                    user_id = user['id']
                    username = user['username']
                    
                    db_positions = await db.get_open_positions(user_id)
                    if not db_positions: continue
                    
                    if not mt5.initialize(path=user['mt5_path']): continue
                    if not mt5.login(login=user['mt5_login'], password=user['mt5_password'], server=user['mt5_server']): continue
                    
                    mt5_positions = mt5.positions_get()
                    if mt5_positions is None: mt5_positions = []
                    mt5_tickets = [p.ticket for p in mt5_positions]
                    
                    for db_pos in db_positions:
                        ticket = db_pos['ticket']
                        
                        if ticket not in mt5_tickets:
                            logger.info(f"🔍 Detected closed position for ticket {ticket} ({db_pos['symbol']}) for user {username}. Syncing exit...")
                            
                            if retry_counts.get(ticket, 0) >= max_retries:
                                logger.warning(f"⚠️ Max retries reached for ticket {ticket}. Skipping sync.")
                                continue
                            history = mt5.history_deals_get(position=ticket)
                            
                            if history and len(history) > 0:
                                exit_deal = history[-1]
                                profit = exit_deal.profit
                                exit_price = exit_deal.price
                                exit_time = datetime.datetime.fromtimestamp(exit_deal.time)
                                reason_code = exit_deal.reason
                                closure_reason = "MANUAL"
                                if reason_code == mt5.DEAL_REASON_TP:
                                    closure_reason = "TAKE PROFIT"
                                elif reason_code == mt5.DEAL_REASON_SL:
                                    closure_reason = "STOP LOSS"
                                elif reason_code == mt5.DEAL_REASON_EXPERT:
                                    closure_reason = "EXPERT/BOT"

                                await db.log_trade_exit_by_ticket(
                                    user_id, 
                                    ticket, 
                                    exit_price, 
                                    profit, 
                                    profit, 
                                    reason=closure_reason
                                )
                                logger.info(f"✅ Synced exit for Ticket {ticket}: Profit {profit}, Price {exit_price}, Reason {closure_reason}")
                                
                                msg = (
                                    f"🎯 **Position Closed ({username})**\n\n"
                                    f"🎫 **Ticket**: {ticket}\n"
                                    f"📌 **Symbol**: {db_pos['symbol']}\n"
                                    f"🎯 **Action**: {db_pos['action']}\n"
                                    f"📦 **Volume**: {db_pos['lot_size']}\n"
                                    f"💰 **P/L**: ${profit:.2f}\n"
                                    f"🚪 **Exit Price**: {exit_price}\n"
                                    f"📝 **Reason**: {closure_reason}"
                                )
                                await notifier.send_message(msg)
                                
                                if ticket in retry_counts: del retry_counts[ticket]
                                
                            else:
                                logger.warning(f"Could not find deal history for ticket {ticket} yet for user {username}.")
                                retry_counts[ticket] = retry_counts.get(ticket, 0) + 1
                                
            except Exception as e:
                logger.error(f"Position Monitor Error: {e}")
                await asyncio.sleep(5)

    async def shared_market_poller():
        """Aggregates ticks into candles for all symbols."""
        with open("Config/mt5_config.json", "r") as f: mt5_cfg = json.load(f)
        symbols = mt5_cfg.get('symbols', ["EURUSD"])
        
        config_path = "Config/mt5_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                mt5_cfg_init = json.load(f)
            if not mt5.initialize(login=mt5_cfg_init['login'], server=mt5_cfg_init['server'], password=mt5_cfg_init['password']):
                logger.error(f"MT5 Initialization Failed for Poller: {mt5.last_error()}")
                return
        else:
            logger.error("MT5 Config missing for Poller.")
            return
        
        logger.info(f"📊 Global Poller Started for symbols: {symbols}")
        last_sentiment_fetch = 0
        cached_sentiment = {s: 0.0 for s in symbols}

        while True:
            try:
                current_time = time.time()
                if current_time - last_sentiment_fetch > 300:
                    for symbol in symbols:
                        cached_sentiment[symbol] = await sentiment_analyzer.get_latest_market_sentiment(symbol)
                    last_sentiment_fetch = current_time
                    logger.info(f"📰 Sentiment Updated for all symbols: {cached_sentiment}")

                for symbol in symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick: continue
                    if symbol not in mtf_managers: mtf_managers[symbol] = MTFManager(timeframes=[60, 300, 3600], window_size=32)
                    mgr = mtf_managers[symbol]
                    
                    closed_tfs = mgr.add_tick({
                        'symbol': symbol, 
                        'bid': tick.bid, 
                        'time': int(tick.time_msc),
                        'sentiment': cached_sentiment.get(symbol, 0.0)
                    })
                    
                    for tf_sec, label in {60: 'M1', 300: 'M5', 3600: 'H1'}.items():
                        current_c = mgr.aggregators[tf_sec].get_current_candle()
                        if current_c:
                            await post_to_dashboard({
                                "type": "candle", 
                                "symbol": symbol, 
                                "timeframe": label, 
                                "sentiment": cached_sentiment.get(symbol, 0.0), 
                                **current_c
                            })

                    for tf_sec in closed_tfs:
                        label = {60: 'M1', 300: 'M5', 3600: 'H1'}.get(tf_sec)
                        if label:
                            c = mgr.aggregators[tf_sec].get_last_closed_candle()
                            if c:
                                await db.log_candle(symbol, label, c)
                        if tf_sec == 60:
                            await manage_trailing_stop(symbol, activation_pips=10, trailing_pips=10, db=db, mtf_managers=mtf_managers)
                await asyncio.sleep(0.1)
            except Exception as e: logger.error(f"Poller error: {e}"); await asyncio.sleep(1)

    async def run_user_worker(user):
        """Dedicated worker for one user's trading lifecycle."""
        user_id, username, login, password, server, path = user['id'], user['username'], user['mt5_login'], user['mt5_password'], user['mt5_server'], user['mt5_path']
        logger.info(f"👤 Starting Worker for {username} (ID: {user_id})")
        
        symbols = srv_config.get('symbols', ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])

        worker_state = {
            "trading_enabled": srv_config.get("trading_enabled", True),
            "ai_mode": srv_config.get("ai_mode", "EXPLORER"),
            "pending_signals": {},
            "last_signal_candle": {s: None for s in symbols}
        }
        
        if not mt5.initialize(path=path):
            logger.error(f"❌ Worker {username}: MT5 init failed at {path}")
            return
            
        if not mt5.login(login=login, password=password, server=server):
            logger.error(f"❌ Worker {username}: Login failed for {login}")
            mt5.shutdown()
            return

        logger.info(f"✅ Worker {username}: Logged into {server}")
        
        
        try:
            while True:
                if not state.get("trading_enabled", True):
                    await asyncio.sleep(5)
                    continue
                    
                for symbol in symbols:
                    mgr = mtf_managers.get(symbol)
                    if not mgr or not mgr.is_tf_ready(60): continue
                    
                    pending_sig = worker_state["pending_signals"].get(symbol)
                    if pending_sig:
                        tick = mt5.symbol_info_tick(symbol)
                        if not tick: continue
                        
                        current_time = time.time()
                        if current_time - pending_sig['time'] > 60:
                            logger.warning(f"💓 Heartbeat Timeout: {symbol} signal discarded (No-Pulse).")
                            del worker_state["pending_signals"][symbol]
                            continue
                        
                        atr = pending_sig.get('atr', 0.0001)
                        confirmation_move = atr * 0.1
                        
                        if pending_sig['action'] == 'BUY':
                            if tick.ask >= pending_sig['entry'] + confirmation_move:
                                confirmed = True
                        else:
                            if tick.bid <= pending_sig['entry'] - confirmation_move:
                                confirmed = True
                                
                        if confirmed:
                            logger.info(f"✅ Heartbeat Confirmed: {symbol} @ {tick.ask if pending_sig['action'] == 'BUY' else tick.bid}. Executing.")
                            await execute_mt5_order(
                                user_id, symbol, pending_sig['action'], pending_sig['lots'], 
                                sl=pending_sig['sl'], tp=pending_sig['tp'], 
                                notifier=notifier, signal_data=pending_sig['signal_data'], db=db
                            )
                            del worker_state["pending_signals"][symbol]
                        continue

                    is_open, _ = is_market_open(symbol)
                    if not is_open: continue
                    
                    current_candle = mgr.aggregators[60].get_current_candle()
                    if current_candle and worker_state["last_signal_candle"].get(symbol) == current_candle['time']:
                        continue

                    MAX_CONCURRENT = srv_config.get("max_concurrent_trades_per_symbol", 5)
                    positions = mt5.positions_get(symbol=symbol)
                    if positions and len([p for p in positions if p.magic == 123456]) >= MAX_CONCURRENT:
                        continue

                    # MTF Synchronization for 27-feature Model
                    if mgr.is_tf_ready(60) and mgr.is_tf_ready(300) and mgr.is_tf_ready(3600):
                        df_m1 = mgr.get_data_for_tf(60)
                        df_m5 = mgr.get_data_for_tf(300)
                        df_h1 = mgr.get_data_for_tf(3600)
                        
                        # Use M1 as anchor, Left Join M5 and H1
                        # Note: In live mode, we only need the VERY LAST row for prediction
                        last_m1 = df_m1.iloc[-32:] # We need window_size=32
                        
                        # Simple sync for live: just take the last 32 of each and align
                        # (In live, they should already be roughly aligned by the manager)
                        
                        def normalize_df(df):
                            feat_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'macd_sig', 'macd_hist', 'sentiment']
                            # Ensure columns exist
                            for c in feat_cols:
                                if c not in df.columns: df[c] = 0.0
                            feats = df[feat_cols].tail(32).values.astype(float)
                            min_v = feats.min(axis=0)
                            max_v = feats.max(axis=0)
                            rng = (max_v - min_v)
                            rng[rng==0] = 1.0
                            return (feats - min_v) / rng

                        seq_m1 = normalize_df(df_m1)
                        seq_m5 = normalize_df(df_m5)
                        seq_h1 = normalize_df(df_h1)
                        
                        # Concatenate into 27 features
                        sync_seq = np.concatenate([seq_m1, seq_m5, seq_h1], axis=1) # [32, 27]
                        
                        gaf_img = gaf_transformer.transform(df_m1['close'].tail(32).values.astype(float))
                        
                        mtf_input_sync = (
                            torch.tensor(gaf_img, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0),
                            torch.tensor(sync_seq, dtype=torch.float32).to(device).unsqueeze(0)
                        )
                    else:
                        continue # Wait for all TFs to be ready

                    signal = consensus_engine.analyze_mtf_consensus(
                        mtf_input_sync, 
                        exploration_rate=srv_config.get("exploration_rate", 0.0), 
                        symbol=symbol
                    )
                    
                    if positions:
                        m1_data_pt = mgr.get_data_for_tf(60)
                        current_atr_pt = m1_data_pt['atr'].iloc[-1] if not m1_data_pt.empty and 'atr' in m1_data_pt.columns else 0.0002
                        pip_size_pt = get_pip_size(symbol)
                        min_tp_pips = 10
                        dynamic_target_pips = max(min_tp_pips, (current_atr_pt * 2.0) / pip_size_pt)
                        
                        for p in positions:
                            if p.magic != 123456: continue
                            tick_pt = mt5.symbol_info_tick(symbol)
                            if not tick_pt: continue
                            if p.type == mt5.POSITION_TYPE_BUY:
                                profit_pips = (tick_pt.bid - p.price_open) / pip_size_pt
                            else:
                                profit_pips = (p.price_open - tick_pt.ask) / pip_size_pt
                            
                            if profit_pips >= dynamic_target_pips:
                                logger.info(f"🎯 Dynamic Target Hit: {symbol} Ticket {p.ticket} | +{profit_pips:.1f} pips (Target: {dynamic_target_pips:.1f}). Closing.")
                                trade_exe_logger.info(f"CLOSE_PROFIT_TARGET: {symbol} | Ticket: {p.ticket} | Pips: +{profit_pips:.1f}")
                                await close_all_positions(user_id, symbol, action_type=p.type, notifier=notifier, db=db)
                                
                    if positions and signal.get('is_exhausted'):
                        for p in positions:
                            if p.magic != 123456: continue
                            if (p.type == mt5.POSITION_TYPE_BUY and signal['action'] == 'BUY') or \
                               (p.type == mt5.POSITION_TYPE_SELL and signal['action'] == 'SELL'):
                                logger.info(f"🛑 AI Exhaustion Exit: {symbol} Ticket {p.ticket}. Reason: {signal['exhaustion_reason']}")
                                trade_exe_logger.info(f"CLOSE_EXHAUSTION: {symbol} | Ticket: {p.ticket} | Reason: {signal['exhaustion_reason']}")
                                await close_all_positions(user_id, symbol, action_type=p.type, notifier=notifier, db=db)
                    
                    if signal['action'] != 'HOLD':
                        trade_logger.info(f"SIGNAL: {username} | {symbol} {signal['action']} (Confidence: {signal['confidence']:.2f})")
                        logger.info(f"🚀 Worker {username}: Signal {symbol} {signal['action']}")
                        
                        account = mt5.account_info()
                        current_equity = account.equity if account else 1000.0
                        
                        current_positions = [p for p in positions if p.magic == 123456]
                        if current_positions:
                            opp_type = mt5.POSITION_TYPE_SELL if signal['action'] == 'BUY' else mt5.POSITION_TYPE_BUY
                            opposite_positions = [p for p in current_positions if p.type == opp_type]
                            
                            if opposite_positions:
                                total_opp_profit = sum(p.profit for p in opposite_positions)
                                if total_opp_profit > 0 and signal['confidence'] >= 0.75:
                                    logger.info(f"🔄 Strategic Profit-Harvest: {symbol} Net Profit ${total_opp_profit:.2f} + Strong Signal ({signal['confidence']:.2f}). Flushing.")
                                    await close_all_positions(user_id, symbol, notifier=notifier, db=db)
                                    positions = mt5.positions_get(symbol=symbol) or []
                        
                        m1_data = mgr.get_data_for_tf(60)
                        current_atr = m1_data['atr'].iloc[-1] if not m1_data.empty and 'atr' in m1_data.columns else 0.0002
                        
                        tick = mt5.symbol_info_tick(symbol)
                        entry_price = tick.ask if signal['action'] == 'BUY' else tick.bid
                        
                        symbol_info = mt5.symbol_info(symbol)
                        point = symbol_info.point
                        tick_size = symbol_info.trade_tick_size
                        tick_value = symbol_info.trade_tick_value
                        
                        sl, tp = risk_manager.calculate_sl_tp(
                            symbol, signal['action'], entry_price, 
                            atr=current_atr, point=point, tick_size=tick_size,
                            digits=symbol_info.digits,
                            confidence=signal['confidence']
                        )
                        
                        sl_price_distance = abs(entry_price - sl)
                        
                        lot_size = risk_manager.calculate_lot_size(
                            current_equity, sl_price_distance, confidence=signal['confidence'], 
                            tick_value=tick_value, tick_size=tick_size
                        )
                        
                        account_info = mt5.account_info()
                        free_margin = account_info.margin_free if account_info else 0.0
                        affordable_lots = risk_manager.calculate_max_affordable_lots(symbol, signal['action'], free_margin, mt5_module=mt5)
                        lot_size = min(lot_size, affordable_lots)
                        
                        if lot_size < symbol_info.volume_min:
                            logger.warning(f"⚠️ {username}: Final lot size {lot_size} is below minimum {symbol_info.volume_min}. Skipping trade.")
                            continue
                        
                        worker_state["pending_signals"][symbol] = {
                            "action": signal['action'],
                            "entry": entry_price,
                            "lots": lot_size,
                            "sl": sl,
                            "tp": tp,
                            "atr": current_atr,
                            "time": time.time(),
                            "signal_data": signal
                        }
                        worker_state["last_signal_candle"][symbol] = current_candle['time']
                        logger.info(f"💓 Heartbeat Pending: {symbol} {signal['action']} @ {entry_price}... waiting for confirmation.")
                
                await asyncio.sleep(2.0)
        except Exception as e:
            logger.error(f"Main Loop Error: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5)

    logger.info("🔄 Syncing historical data from Local Database...")
    m1_history = await sync_historical_data(mtf_managers, db, window_size=window_size)
    
    if m1_history:
        logger.info(f"📤 Pushing {len(m1_history)} historical candles to Dashboard (Batch)...")
        await post_to_dashboard(m1_history)
    
    await notifier.send_message(
        "🚀 **QuantSystem Started**\n\nOperation Mode: {}\nSystem is ready and monitoring.".format(srv_config.get("ai_mode", "EXPLORER")),
        reply_markup=notifier.get_main_menu()
    )
    
    users = await db.get_active_users()
    logger.info(f"👥 Found {len(users)} active users in database")
    
    background_tasks = [
        asyncio.create_task(command_worker()),
        asyncio.create_task(db_monitor_worker()),
        asyncio.create_task(retraining_worker()),
        asyncio.create_task(reward_shaping_worker()),
        asyncio.create_task(position_monitor_worker()),
        asyncio.create_task(shared_market_poller())
    ]
    
    for user in users:
        task = asyncio.create_task(run_user_worker(user))
        active_workers[user['id']] = task
        background_tasks.append(task)
    
    logger.info(f"✅ System fully initialized. Running {len(background_tasks)} workers.")
    
    try:
        await asyncio.gather(*background_tasks)
    except Exception as e:
        logger.error(f"Critical system error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_engine())
    except KeyboardInterrupt:
        logger.info("Shutdown.")
