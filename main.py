import asyncio
import httpx
import logging
import json
import torch
import datetime
import os
import MetaTrader5 as mt5
from AI_Brain.preprocessor import SlidingWindowBuffer, GAFTransformer, CandleAggregator, MTFManager
from AI_Brain.models import PatternCNN, TrendLSTM
from AI_Brain.decision_engine import DecisionEngine, RiskManager
from Database.db_handler import DBHandler
from ZMQ_Bridge.telegram_notifier import TelegramNotifier
import traceback
import subprocess
from AI_Brain.llm_advisor import LLMRewardAdvisor
from AI_Brain.training_pipeline import train_rl_mode

# Setup Logging (Hardened)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers if any (to prevent duplicates/locks)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

file_handler = logging.FileHandler("quant_system.log", mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

# Silence verbose third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("Main")
logger.info("="*50)
logger.info("--- SYSTEM STARTUP ---")
logger.info("="*50)

# Shared HTTP Client for Dashboard
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
        pass # Dashboard might be offline

def get_filling_type(symbol):
    """Dynamically detects the supported filling mode for a symbol/broker"""
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return mt5.ORDER_FILLING_IOC
        
    filling_mode = symbol_info.filling_mode
    if filling_mode & 1: # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    elif filling_mode & 2: # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def is_market_open(symbol):
    """
    Checks if the market for a specific symbol is open for full trading.
    Returns: (bool, message)
    """
    # 1. Global Terminal Trade Permission
    terminal = mt5.terminal_info()
    if not terminal or not terminal.trade_allowed:
        return False, "Broker/Terminal trade permission denied (Global)."

    # 2. Symbol-Specific Permission
    info = mt5.symbol_info(symbol)
    if info is None:
        return False, f"Symbol {symbol} info not found."
    
    # trade_mode: 0=Disabled, 1=LongOnly, 2=ShortOnly, 3=CloseOnly, 4=Full
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return False, "Market is completely CLOSED for this symbol."
    elif info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        return False, "Market is CLOSE-ONLY (No new entries allowed)."
    
    # Note: LongOnly or ShortOnly are rare but allowed for specific directions.
    # We primarily look for SYMBOL_TRADE_MODE_FULL for general trading.
    if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        return False, f"Market restricted (Trade Mode: {info.trade_mode})."

    return True, "Market is open."

async def execute_mt5_order(symbol, action, volume, sl=0.0, tp=0.0, notifier=None, comment="QuantAI", signal_data=None, db=None):
    """Executes a trade directly via MetaTrader5 Python API"""
    try:
        # 0. Check if MT5 is actually initialized
        if not mt5.terminal_info():
            msg = "❌ **MT5 Error**: Terminal not connected or initialized."
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None

        # 1. Ensure symbol is selected and visible
        if not mt5.symbol_select(symbol, True):
            msg = f"❌ **MT5 Error**: Symbol {symbol} not found or cannot be selected."
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None

        # 1.5 Market Open Check
        is_open, mkt_msg = is_market_open(symbol)
        if not is_open:
            msg = f"🕰️ **Market Closed**: {symbol}\n📝 {mkt_msg}"
            logger.warning(msg)
            if notifier: await notifier.send_message(msg)
            return None

        # 2. Get latest tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            msg = f"❌ **MT5 Error**: Could not get price for {symbol}."
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None
            
        print(f"DEBUG: Preparing order for {symbol} {action} {volume}")
        order_type = mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action.upper() == "BUY" else tick.bid
        
        # Detect Filling Type dynamically
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
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        
        # 3. Send Order
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"❌ **Order Failed**: {result.comment if result else 'None'} (Code: {result.retcode if result else 'N/A'})"
            logger.error(msg)
            if notifier: await notifier.send_message(msg)
            return None
        else:
            reason = signal_data.get('reason', 'N/A') if signal_data else 'Manual/Unknown'
            msg = f"✅ **Order Executed**\n\n🎯 {action} {volume} {symbol} @ {result.price}\n📝 Reason: {reason}"
            logger.info(msg)
            if notifier: await notifier.send_message(msg)
        
        # 4. Log Trade Entry for RL
        if db and signal_data:
            # Fetch Position Ticket (more reliable than Order Ticket for tracking)
            position_ticket = result.order # Default to order ticket
            await asyncio.sleep(0.5) # Give MT5 a moment to create position
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for p in positions:
                    if p.magic == 123456 and p.volume == float(volume):
                        position_ticket = p.ticket
                        break
            
            await db.log_trade_entry(symbol, action, volume, result.price, signal_data, ticket=position_ticket)
            
        return result
    except Exception as e:
        msg = f"⚠️ **Execution Crash**: {str(e)}"
        logger.error(msg)
        if notifier: await notifier.send_message(msg)
        return None

async def close_all_positions(symbol, action_type=None, notifier=None, db=None):
    """Closes positions and logs exit for RL"""
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for p in positions:
            if action_type is not None and p.type != action_type: continue

            tick = mt5.symbol_info_tick(symbol)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask,
                "magic": 123456,
                "comment": "Close Reversal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": get_filling_type(symbol),
            }
            res = mt5.order_send(request)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                # Log Exit for RL (Fetch Profit)
                total_profit = 0.0
                profit_text = "Checking..."
                
                # Wait for history update to get accurate profit
                await asyncio.sleep(0.5) 
                deals = mt5.history_deals_get(position=p.ticket)
                if deals and len(deals) > 0:
                    exit_deal = deals[-1] 
                    total_profit = sum([d.profit + d.swap + d.commission for d in deals])
                    profit_text = f"${total_profit:.2f}"
                    
                    if db:
                        try:
                            # Log as REVERSAL since this is triggered by signal contradiction
                            await db.close_latest_trade(symbol, exit_deal.price, exit_deal.profit, total_profit, reason="REVERSAL")
                        except Exception as e:
                            logger.error(f"Failed to log RL exit: {e}")

                msg = f"🎯 **Position Closed (REVERSAL)**\n\n📌 {symbol}\n🎫 Ticket: {p.ticket}\n💰 P/L: {profit_text}"
                logger.info(msg)
                if notifier: await notifier.send_message(msg)
            else:
                logger.error(f"Failed to close {p.ticket}: {res.comment}")

async def sync_historical_data(managers_dict, db):
    """Fetches history for all configured symbols and timeframes."""
    logger.info("Syncing 5000 historical candles for all symbols from MT5...")
    m1_history_batch = []
    
    config_path = "Config/mt5_config.json"
    if not os.path.exists(config_path): return []

    with open(config_path, "r") as f:
        config = json.load(f)

    if not mt5.terminal_info():
        logger.error("MT5 not initialized. Sync skipped.")
        return []

    tf_map = {60: mt5.TIMEFRAME_M1, 300: mt5.TIMEFRAME_M5, 3600: mt5.TIMEFRAME_H1}
    symbols = config.get('symbols', ["EURUSD"])
    
    for symbol in symbols:
        if symbol not in managers_dict:
            managers_dict[symbol] = MTFManager(timeframes=[60, 300, 3600], window_size=32)
            
        logger.info(f"Syncing {symbol}...")
        for tf_sec, mt5_tf in tf_map.items():
            # Fetch 1000 candles (Reduced from 5000 to prevent timeouts)
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 1000)
            if rates is not None:
                # Map timeframe seconds to labels
                tf_label = {60: 'M1', 300: 'M5', 3600: 'H1'}.get(tf_sec, 'M1')
                
                # Batch for DB
                db_batch = []
                
                # Optimize loop with enumerate
                for i, rate in enumerate(rates):
                    candle = {
                        'symbol': symbol, 'timeframe': tf_label,
                        'open': float(rate['open']), 'high': float(rate['high']), 
                        'low': float(rate['low']), 'close': float(rate['close']), 
                        'tick_volume': int(rate['tick_volume']), 'time': int(rate['time'])
                    }
                    
                    # Add to memory buffer (only needs recent ones for calculation)
                    # Optimization: Only add last 500 to buffer to save RAM
                    if len(rates) - i <= 500:
                         managers_dict[symbol].buffers[tf_sec].add_candle(candle)
                    
                    # Prepare for DB
                    db_batch.append(candle)

                    # Prepare for Dashboard
                    m1_history_batch.append({
                        "type": "candle", "symbol": symbol, "timeframe": tf_label,
                        "time": int(rate['time']), "open": candle['open'],
                        "high": candle['high'], "low": candle['low'], "close": candle['close']
                    })
                
                logger.info(f"  > Synced {len(db_batch)} candles for {tf_label}")

                # Log Batch to DB
                if db:
                    await db.log_candles_batch(db_batch)
    
    logger.info(f"Sync complete. Total items prepared: {len(m1_history_batch)}")
    return m1_history_batch

async def run_trading_engine():
    """Main Trading Engine with Reconnection Loop and Telegram Control"""
    logger.info("Initializing Hybrid Quant System...")
    
    # Load MT5 Connection early for native execution
    config_path = "Config/mt5_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            mt5_cfg = json.load(f)
        if not mt5.initialize(login=mt5_cfg['login'], server=mt5_cfg['server'], password=mt5_cfg['password']):
            logger.error(f"MT5 Initialization Failed: {mt5.last_error()}")
            return
    else:
        logger.error("MT5 Config missing.")
        return
    
    # 1. Initialize Core Components
    notifier = TelegramNotifier()
    db = DBHandler()
    
    # AI Components
    mtf_managers = {} # sym -> MTFManager
    gaf_transformer = GAFTransformer(image_size=32)
    
    cnn = PatternCNN()
    lstm = TrendLSTM()
    try:
        cnn.load_state_dict(torch.load("AI_Brain/weights/cnn_model.pt"))
        lstm.load_state_dict(torch.load("AI_Brain/weights/lstm_model.pt"))
        logger.info("AI Models Ready.")
    except Exception as e:
        logger.error(f"Critical Error Loading Models: {e}")
        return

    decision_engine = DecisionEngine(cnn, lstm)
    risk_manager = RiskManager()
    
    # Initialize LLM Reward Advisor for Training
    advisor = None
    config_path = "Config/server_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            srv_config = json.load(f)
        advisor = LLMRewardAdvisor(
            srv_config.get("gemini_api_key"), 
            model_name=srv_config.get("gemini_model", "gemini-1.5-flash")
        )

    # Shared State for Background Tasks
    account = mt5.account_info()
    state = {
        "trading_enabled": True,
        "ai_mode": "CONSERVATIVE",
        "exploration_rate": 0.0,
        "symbol": "EURUSD", # Default for manual tests
        "session_stats": {
            "trades": 0, "buys": 0, "sells": 0, 
            "start_time": datetime.datetime.now(),
            "equity": account.equity if account else 10000.0,
            "balance": account.balance if account else 10000.0
        },
        "last_rl_order_count": 0,
        "main_menu": notifier.get_main_menu()
    }

    # Load Server Config
    try:
        with open("Config/server_config.json", "r") as f:
            srv_config = json.load(f)
        state["trading_enabled"] = srv_config.get("trading_enabled", True)
        state["ai_mode"] = srv_config.get("ai_mode", "CONSERVATIVE")
        state["exploration_rate"] = srv_config.get("exploration_rate", 0.0)
    except:
        pass

    async def command_worker():
        """Background task to poll Telegram commands without blocking data flow"""
        while True:
            try:
                cmds = await notifier.check_commands()
                for cmd in cmds:
                    if cmd == "/off":
                        if state["trading_enabled"]:
                            state["trading_enabled"] = False
                            await notifier.send_message("🔴 **Trading Disabled**", reply_markup=state["main_menu"])
                    elif cmd == "/on":
                        if not state["trading_enabled"]:
                            state["trading_enabled"] = True
                            await notifier.send_message("🟢 **Trading Enabled**", reply_markup=state["main_menu"])
                    elif cmd == "/status":
                        status = "ACTIVE" if state["trading_enabled"] else "PAUSED"
                        await notifier.send_message(f"ℹ️ **Status**: {status}\nTrades: {state['session_stats']['trades']}")
                    elif cmd == "/stats":
                        uptime = datetime.datetime.now() - state["session_stats"]['start_time']
                        await notifier.send_message(f"📊 **Stats**\nUptime: {str(uptime).split('.')[0]}\nTrades: {state['session_stats']['trades']}")
                    elif cmd == "/history":
                        signals = await db.get_recent_signals(5)
                        if signals:
                            hist_text = "📜 **History**\n"
                            for s in signals: hist_text += f"• {s['time'].strftime('%H:%M')} | {s['final_signal']}\n"
                            await notifier.send_message(hist_text)
                    elif cmd == "/test_buy":
                        await execute_mt5_order(state['symbol'], "BUY", 0.01, notifier=notifier)
                    elif cmd == "/test_sell":
                        await execute_mt5_order(state['symbol'], "SELL", 0.01, notifier=notifier)
            except Exception as e:
                logger.error(f"Command worker error: {e}")
            await asyncio.sleep(2)

    async def db_monitor_worker():
        """Checks DB health every 5 minutes and alerts if down"""
        while True:
            await asyncio.sleep(300)
            if not await db.is_healthy():
                await notifier.send_message("🚨 **DATABASE ALERT**: Connection lost! RL data collection is paused.")
                logger.error("Database health check failed.")

    async def retraining_worker():
        """Retrains the model every 200 closed orders using the trade logic"""
        # Initialize last count on startup to avoid immediate retraining
        state["last_rl_order_count"] = await db.count_closed_trades()
        logger.info(f"Retraining worker started. initial closed trades: {state['last_rl_order_count']}")
        
        while True:
            await asyncio.sleep(600) # Check every 10 minutes
            
            try:
                current_closed = await db.count_closed_trades()
                new_trades = current_closed - state["last_rl_order_count"]
                
                if new_trades >= 200:
                    logger.info(f"🕒 RL Trigger: {new_trades} new trades detected. Starting RL retraining...")
                    await notifier.send_message(f"🧠 **RL Retraining Started**\n\nLearning from {new_trades} new trade results...")
                    
                    # 1. Fetch Training Data from DB
                    experiences = await db.get_rl_training_data(limit=500) # Get recent 500
                    
                    if experiences:
                        # 2. Run RL Training (Async with LLM Reward Shaping)
                        await train_rl_mode(experiences, advisor=advisor, db=db)
                        
                        # 3. Reload weights and update state
                        cnn.load_state_dict(torch.load("AI_Brain/weights/cnn_model.pt", map_location='cpu'))
                        lstm.load_state_dict(torch.load("AI_Brain/weights/lstm_model.pt", map_location='cpu'))
                        state["last_rl_order_count"] = current_closed
                        
                        logger.info("♻️ System updated with new RL weights.")
                        
                        # 4. Optional: Run backtest and notify
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
                            await notifier.send_message("🧠 **RL Learning Complete**: บอทเรียนรู้จากผลกำไรขาดทุนล่าสุดเรียบร้อยครับ!")
            except Exception as e:
                logger.error(f"⚠️ Retraining Exception: {e}")

    async def position_monitor_worker():
        """Monitor phantom positions closed by SL/TP in MT5 and sync with DB"""
        logger.info("Position monitor worker started.")
        while True:
            try:
                # 1. Get all OPEN trades from DB
                open_trades = await db.pool.fetch("SELECT id, symbol, ticket FROM trade_logs WHERE status = 'OPEN'")
                
                if open_trades:
                    # 2. Get active positions from MT5
                    active_positions = mt5.positions_get()
                    active_tickets = [p.ticket for p in active_positions] if active_positions else []
                    
                    for trade in open_trades:
                        db_id = trade['id']
                        symbol = trade['symbol']
                        ticket = trade['ticket']
                        
                        # 3. If DB trade is NOT in MT5 active positions, it was closed (SL, TP, or manual)
                        if ticket not in active_tickets:
                            logger.info(f"🔍 Detected closed position for ticket {ticket} ({symbol}). Syncing exit...")
                            
                            # 4. Fetch history to get details
                            # Look back 24 hours to find the deal
                            from_date = datetime.datetime.now() - datetime.timedelta(days=1)
                            deals = mt5.history_deals_get(from_date, datetime.datetime.now(), position=ticket)
                            
                            if deals and len(deals) >= 2: # At least Entry and Exit deal
                                entry_deal = deals[0]
                                exit_deal = deals[-1]
                                
                                # Detect Reason from MT5
                                # mt5.DEAL_REASON_CLIENT (3) = Manual
                                # mt5.DEAL_REASON_SL (4) = Stop Loss
                                # mt5.DEAL_REASON_TP (5) = Take Profit
                                # mt5.DEAL_REASON_EXPERT (1) = EA/Bot
                                reason_map = {
                                    1: "EA/BOT",
                                    3: "MANUAL",
                                    4: "SL",
                                    5: "TP",
                                    16: "SO" # Stop Out
                                }
                                raw_reason = getattr(exit_deal, 'reason', 0)
                                reason_str = reason_map.get(raw_reason, f"OTHER ({raw_reason})")

                                total_profit = sum([d.profit + d.swap + d.commission for d in deals])
                                logger.info(f"✅ Syncing trade exit: Ticket {ticket} | Reason: {reason_str} | Profit: ${total_profit:.2f}")
                                
                                await db.log_trade_exit(db_id, exit_deal.price, exit_deal.profit, total_profit, reason=reason_str)
                                
                                # Notify Telegram
                                msg = f"🎯 **Position Closed ({reason_str})**\n\n📌 {symbol}\n🎫 Ticket: {ticket}\n💰 P/L: ${total_profit:.2f}"
                                await notifier.send_message(msg)
                            else:
                                # Fallback: Maybe it was closed just now, wait and retry next loop
                                logger.warning(f"Could not find deal history for ticket {ticket} yet.")

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            
            await asyncio.sleep(10) # Check every 10 seconds

    # Connect to DB first
    await db.connect()

    # Start Background Tasks
    asyncio.create_task(command_worker())
    asyncio.create_task(db_monitor_worker())
    asyncio.create_task(retraining_worker())
    asyncio.create_task(position_monitor_worker())

    # Heartbeat task to ensure logger and engine are alive
    async def heartbeat():
        while True:
            logger.info("💓 HEARTBEAT: Trading engine is running.")
            await asyncio.sleep(600) # Every 10 minutes
    asyncio.create_task(heartbeat())
    await db.connect()
    
    # Sync History + Recent Signals for Dashboard
    history_batch = await sync_historical_data(mtf_managers, db)
    
    # Add recent signals from DB to history batch
    recent_signals = await db.get_recent_signals(10)
    for s in recent_signals:
        history_batch.append({
            "type": "signal", "symbol": s['symbol'], 
            "action": s['final_signal'], "confidence": s['cnn_confidence'],
            "time": int(s['time'].timestamp())
        })
        
    if history_batch:
        await post_to_dashboard(history_batch)

    await notifier.send_message(
        "🚀 **Server Started**\nBuffers warmed up. System ready to trade immediately.", 
        reply_markup=state["main_menu"]
    )

    # 2. Main Signal Loop (Native MT5 Polling)
    # This replaces the ZMQ bridge to ensure 100% reliability without needing an EA
    symbols = mt5_cfg.get('symbols', ["EURUSD"])
    logger.info(f"Starting Native Polling for symbols: {symbols}")
    
    while True:
        try:
            for symbol in symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None: continue
                
                # Standardize tick data for aggregators
                tick_data = {
                    'symbol': symbol,
                    'bid': float(tick.bid), 
                    'time': int(tick.time_msc) 
                }
                
                if symbol not in mtf_managers:
                    mtf_managers[symbol] = MTFManager(timeframes=[60, 300, 3600], window_size=32)
                
                mgr = mtf_managers[symbol]
                closed_tfs = mgr.add_tick(tick_data)
                
                # Update Dashboard on every tick (M1 current candle)
                current_m1 = mgr.aggregators[60].get_current_candle()
                if current_m1:
                    asyncio.create_task(post_to_dashboard({
                        "type": "candle", "symbol": symbol,
                        "time": int(current_m1['time']), "open": current_m1['open'],
                        "high": current_m1['high'], "low": current_m1['low'],
                        "close": current_m1['close']
                    }))

                if 60 in closed_tfs:
                    # Log Closed M1 Candle for RL
                    m1_data = mgr.aggregators[60].get_last_closed_candle()
                    if m1_data:
                        asyncio.create_task(db.log_candle(symbol, "M1", m1_data))

                    # 0. Market Open Check (Skip analysis if closed)
                    is_open, _ = is_market_open(symbol)
                    if not is_open:
                        # Log once per closure to avoid spam
                        if int(m1_data['time']) % 3600 == 0: # Every hour
                            logger.info(f"🕰️ Market for {symbol} is closed. Skipping AI analysis.")
                        continue

                    logger.info(f"Analyzing {symbol} (Closed M1)...")
                    if mgr.is_tf_ready(60):
                        mtf_inputs = {}
                        for tf in [60, 300, 3600]:
                            if mgr.is_tf_ready(tf):
                                df = mgr.get_data_for_tf(tf)
                                gaf_img = gaf_transformer.transform(df['close'].values.astype(float))
                                features = df[['open', 'high', 'low', 'close', 'tick_volume']].values.astype(float)
                                min_v, max_v = features.min(axis=0), features.max(axis=0)
                                rng = (max_v - min_v)
                                rng[rng==0] = 1.0
                                norm_feat = (features - min_v) / rng
                                
                                mtf_inputs[tf] = (
                                    torch.tensor(gaf_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                    torch.tensor(norm_feat, dtype=torch.float32).unsqueeze(0)
                                )

                        signal = decision_engine.analyze_mtf(
                            mtf_inputs, 
                            ai_mode=state["ai_mode"], 
                            exploration_rate=state["exploration_rate"],
                            symbol=symbol
                        )
                        
                        if signal['action'] != 'HOLD' or signal.get('analyst_metadata'):
                            meta = signal.get('analyst_metadata', {})
                            
                            # Update Dashboard with Signal (Direct from AI)
                            asyncio.create_task(post_to_dashboard({
                                "type": "signal", 
                                "symbol": symbol, 
                                "action": signal['action'], 
                                "confidence": signal['confidence'],
                                "time": int(current_m1['time']),
                                "pattern": meta.get('pattern'),
                                "future_outlook": meta.get('future_outlook'),
                                "price": tick.bid,
                                "analysis": f"🎯 Signal: {meta.get('pattern', 'Unknown')} | Trend: {meta.get('future_outlook', 'Neutral')}"
                            }))
                            
                            if signal['action'] != 'HOLD':
                                logger.info(f"LIVE SIGNAL: {symbol} -> {signal['action']}")
                                # await db.log_signal(symbol, signal) # Replaced by trade entry logging later
                                
                                state["session_stats"]['trades'] += 1
                                if signal['action'] == 'BUY': state["session_stats"]['buys'] += 1
                                elif signal['action'] == 'SELL': state["session_stats"]['sells'] += 1

                                if state["trading_enabled"]:
                                    # 1. Inform user processing has started
                                    init_msg = f"🔍 **Signal Detected**: {signal['action']} {symbol}\n📊 Confidence: {int(signal['confidence']*100)}%"
                                    await notifier.send_message(init_msg)
                                    logger.info(f"Processing execution for {symbol} {signal['action']}")
                                    
                                    # Fetch current balance for lot calculation
                                    account = mt5.account_info()
                                    current_equity = account.equity if account else 10000.0
                                    
                                    # Close Reversal Logic
                                    target_to_close = mt5.POSITION_TYPE_SELL if signal['action'] == 'BUY' else mt5.POSITION_TYPE_BUY
                                    await close_all_positions(symbol, action_type=target_to_close, notifier=notifier, db=db)
                
                                    # Calculate SL/TP and Entry
                                    entry_price = tick.ask if signal['action'] == 'BUY' else tick.bid
                                    
                                    # Ensure 20 pips is at least > StopLevel
                                    sym_info = mt5.symbol_info(symbol)
                                    stop_level = sym_info.trade_stops_level if sym_info else 0
                                    point = sym_info.point if sym_info else 0.00001
                                    tick_size = sym_info.trade_tick_size if sym_info else 0.00001
                                    
                                    pip_in_points = 10.0
                                    min_dist_points = stop_level + 2
                                    sl_pips = 20
                                    if (sl_pips * pip_in_points) < min_dist_points:
                                        sl_pips = (min_dist_points / pip_in_points) + 1
                                        print(f"DEBUG: Adjusted SL pips to {sl_pips:.1f} due to StopLevel {stop_level}")

                                    lot_size = risk_manager.calculate_lot_size(current_equity, sl_pips, confidence=signal['confidence'])
                                    sl, tp = risk_manager.calculate_sl_tp(
                                        symbol, signal['action'], entry_price, 
                                        stop_loss_pips=sl_pips, point=point, tick_size=tick_size
                                    )
                                    
                                    logger.info(f"📊 Risk Calc: Lot={lot_size:.2f} | Entry={entry_price:.5f} | SL={sl:.5f} | TP={tp:.5f}")
                                    
                                    # Execute with full logging
                                    logger.info(f"🚀 Sending order to MT5: {symbol} {signal['action']} volume={lot_size}")
                                    result = await execute_mt5_order(symbol, signal['action'], lot_size, sl=sl, tp=tp, notifier=notifier, signal_data=signal, db=db)
                                    
                                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                        logger.info(f"✅ Trade Successful: Ticket {result.order}")
                                    else:
                                        err_code = result.retcode if result else "NO_RESULT"
                                        err_msg = result.comment if result else "Execution failed silently"
                                        logger.error(f"❌ Trade Failed: {symbol} {signal['action']} | Code: {err_code} | Msg: {err_msg}")
                                else:
                                    full_msg = f"👀 **OBSERVATION MODE**\n\n{llm_reasoning}"
                                    await notifier.send_message(full_msg)

            await asyncio.sleep(0.01) # Poll every 10ms
        except Exception as e:
            logger.error(f"Main Loop Error: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_engine())
    except KeyboardInterrupt:
        logger.info("Shutdown.")
