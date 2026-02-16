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
        
    # Phase 91: Pre-Market Close Protection (Friday Night)
    # Most Forex markets close around Saturday 04:00-05:00 (Thai Time)
    # We stop new entries 4 hours before the estimated close to avoid high spread/volatility errors.
    # Stop at Midnight Friday (Friday 23:59:59)
    now = datetime.datetime.now()
    if now.weekday() == 4: # Friday
        if now.hour >= 23: # Stop at 11 PM Friday
            return False, "Friday Night Protection: Market nearing weekly close. Volatility too high."
    elif now.weekday() == 5: # Saturday
        return False, "Market is closed for the weekend."
    
    # Note: LongOnly or ShortOnly are rare but allowed for specific directions.
    # We primarily look for SYMBOL_TRADE_MODE_FULL for general trading.
    if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        return False, f"Market restricted (Trade Mode: {info.trade_mode})."

    return True, "Market is open."

async def execute_mt5_order(user_id, symbol, action, volume, sl=0, tp=0, notifier=None, signal_data=None, db=None):
    """Executes a trade on MT5 for a specific user and logs it globally."""
    # Ensure correct MT5 instance is used (if multiple terminals, this would involve switching)
    # For now, we assume the worker manages the connection state.
    
    # 1. Prepare Request
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
            "comment": f"QuantAI_{user_id}",
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
            # Phase 91: Universal Position Tracking via Deal Mapping
            # This is 100% reliable as it queries the deal that actually executed.
            await asyncio.sleep(1.0) # Ensure MT5 history is updated
            deal_ticket = result.deal
            position_ticket = result.order # Default
            
            history_deal = mt5.history_deals_get(ticket=deal_ticket)
            if history_deal:
                position_ticket = history_deal[0].position_id
                logger.info(f"🔗 Linked Deal {deal_ticket} to Position {position_ticket}")
            else:
                # Fallback to position search if deal history is slow
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
        # Match by Magic Number (Hardcoded for this bot version)
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
            # RL Logging: Fetch profit details
            await asyncio.sleep(0.5)
            deals = mt5.history_deals_get(position=p.ticket)
            total_profit = sum([d.profit + d.swap + d.commission for d in deals]) if deals else 0.0
            
            msg = f"🎯 **Position Closed**\n\n📌 {symbol}\n🎫 Ticket: {p.ticket}\n💰 P/L: ${total_profit:.2f}"
            logger.info(msg + f" (User: {user_id})")
            if notifier: await notifier.send_message(msg)
            
            if db:
                await db.log_trade_exit_by_ticket(user_id, p.ticket, res.price, res.profit, total_profit, reason="REVERSAL" if action_type is not None else "MANUAL")
        else:
            logger.error(f"Failed to close {p.ticket} for User {user_id}: {res.comment}")

async def sync_historical_data(managers_dict, db):
    """Fetches history for all configured symbols and timeframes (Last 2 Weeks)."""
    # Phase 90: Targeted 2-Week Sync (14 days)
    # M1: 20160, M5: 4032, H1: 336
    WEEKS = 2
    DAYS = WEEKS * 7
    
    logger.info(f"🔄 Deep Sync: Fetching last {WEEKS} weeks of data from MT5...")
    m1_history_batch = []
    
    config_path = "Config/mt5_config.json"
    if not os.path.exists(config_path): return []

    with open(config_path, "r") as f:
        config = json.load(f)

    if not mt5.terminal_info():
        logger.info("Initializing MT5 for sync...")
        if not mt5.initialize(login=config['login'], server=config['server'], password=config['password']):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}. Sync skipped.")
            return []

    tf_map = {60: mt5.TIMEFRAME_M1, 300: mt5.TIMEFRAME_M5, 3600: mt5.TIMEFRAME_H1}
    symbols = config.get('symbols', ["EURUSD"])
    
    for symbol in symbols:
        if symbol not in managers_dict:
            managers_dict[symbol] = MTFManager(timeframes=[60, 300, 3600], window_size=32)
            
        logger.info(f"Syncing {symbol}...")
        for tf_sec, mt5_tf in tf_map.items():
            # Calculate count for 14 days
            if tf_sec == 60: count = DAYS * 24 * 60
            elif tf_sec == 300: count = DAYS * 24 * 12
            else: count = DAYS * 24
            
            logger.info(f"  > Fetching {count} candles for TF={tf_sec}s...")
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates is not None:
                tf_label = {60: 'M1', 300: 'M5', 3600: 'H1'}.get(tf_sec, 'M1')
                db_batch = []
                
                for i, rate in enumerate(rates):
                    candle = {
                        'symbol': symbol, 'timeframe': tf_label,
                        'open': float(rate['open']), 'high': float(rate['high']), 
                        'low': float(rate['low']), 'close': float(rate['close']), 
                        'tick_volume': int(rate['tick_volume']), 'time': int(rate['time'])
                    }
                    
                    # Add to memory buffer (only last 500 for computation)
                    if len(rates) - i <= 500:
                         managers_dict[symbol].buffers[tf_sec].add_candle(candle)
                    
                    db_batch.append(candle)

                    # Prepare for Dashboard (only M1 for chart)
                    if tf_sec == 60:
                        m1_history_batch.append({
                            "type": "candle", "symbol": symbol, "timeframe": tf_label,
                            "time": int(rate['time']), "open": candle['open'],
                            "high": candle['high'], "low": candle['low'], "close": candle['close']
                        })
                
                logger.info(f"  > Synced {len(db_batch)} candles to DB for {tf_label}")

                # Log Batch to DB
                if db:
                    await db.log_candles_batch(db_batch)
    
    logger.info(f"Sync complete. Total items prepared: {len(m1_history_batch)}")
    return m1_history_batch

async def run_trading_engine():
    """Main Orchestrator for Multi-User Trading System."""
    logger.info("Starting Multi-User Trading Orchestrator...")
    
    # 1. Initialize Global Components
    db = DBHandler()
    await db.connect()
    
    with open("Config/server_config.json", "r") as f: srv_config = json.load(f)
    
    notifier = TelegramNotifier()  # Uses default config path
    gaf_transformer = GAFTransformer()
    # Shared AI Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = PatternCNN().to(device)
    lstm = TrendLSTM(input_size=5, hidden_size=64, dropout=0.3).to(device)
    
    if os.path.exists("AI_Brain/weights/cnn_model.pt"):
        cnn.load_state_dict(torch.load("AI_Brain/weights/cnn_model.pt", map_location=device))
        lstm.load_state_dict(torch.load("AI_Brain/weights/lstm_model.pt", map_location=device))
        logger.info("✅ Loaded existing AI models.")
        
    decision_engine = DecisionEngine(cnn, lstm)
    risk_manager = RiskManager(risk_per_trade=0.02)
    advisor = LLMRewardAdvisor(
        api_key_pool=srv_config["gemini_api_key_pool"],
        model_pool=srv_config.get("gemini_model_pool")
    ) if srv_config.get("llm_enabled") else None
    
    mtf_managers = {} # Shared market data
    active_workers = {} # user_id -> Task
    
    # --- Background Workers (Shared) ---
    async def command_worker():
        """Handles global Telegram commands."""
        while True:
            try:
                cmds = await notifier.check_commands()
                for cmd in cmds:
                    if cmd == "/info":
                        # Comprehensive System Information
                        users = await db.get_active_users()
                        total_trades = await db.count_total_trades()
                        open_trades = await db.count_open_trades()
                        closed_trades = await db.count_closed_trades()
                        
                        # LLM Status
                        llm_status = "✅ ENABLED" if advisor else "❌ DISABLED"
                        llm_health = "🟢 HEALTHY" if advisor and advisor.is_healthy() else "🔴 UNHEALTHY"
                        
                        # AI Mode
                        ai_mode = srv_config.get("ai_mode", "EXPLORER")
                        exploration_rate = srv_config.get("exploration_rate", 0.2)
                        
                        msg = f"""📊 **SYSTEM INFO**

🔧 **Status**: ACTIVE
👥 **Users**: {len(users)} active
⚙️ **Workers**: {len(active_workers)} running

📈 **Trade Statistics**
├─ Total: {total_trades}
├─ Open: {open_trades}
└─ Closed: {closed_trades}

🧠 **AI Intelligence**
├─ Mode: {ai_mode}
├─ Exploration: {exploration_rate * 100:.0f}%
└─ LLM: {llm_status} {llm_health if advisor else ""}

💾 **Database**: {'🟢 CONNECTED' if await db.is_healthy() else '🔴 DISCONNECTED'}"""
                        
                        await notifier.send_message(msg)
            except Exception as e: logger.error(f"Command worker error: {e}")
            await asyncio.sleep(5)

    async def db_monitor_worker():
        """Checks DB health every 5 minutes and alerts if down"""
        while True:
            await asyncio.sleep(300)
            if not await db.is_healthy():
                await notifier.send_message("🚨 **DATABASE ALERT**: Connection lost! RL data collection is paused.")
                logger.error("Database health check failed.")

    # Shared State for Background Tasks
    state = {
        "trading_enabled": True,
        "last_rl_order_count": 0,
        "main_menu": notifier.get_main_menu()
    }

    async def retraining_worker():
        """Shared RL retraining logic."""
        # Initialize last count on startup to avoid immediate retraining
        # LOAD from DB Metadata if exists, else fallback to current count
        persistent_count = await db.get_metadata("last_retrain_trade_count")
        
        if persistent_count is not None:
            state["last_rl_order_count"] = int(persistent_count)
            logger.info(f"Retraining worker started. Loaded persistent count: {state['last_rl_order_count']}")
        else:
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
                    
                    # 1. Fetch Training Data from DB (Now pre-shaped by the background worker)
                    experiences = await db.get_rl_training_data(limit=500) 
                    
                    if experiences:
                        # 2. Run RL Training (Advisor here is now used for any fallback shaping)
                        await train_rl_mode(experiences, advisor=advisor, db=db)
                        
                        # 3. Reload weights and update state
                        cnn.load_state_dict(torch.load("AI_Brain/weights/cnn_model.pt", map_location='cpu'))
                        lstm.load_state_dict(torch.load("AI_Brain/weights/lstm_model.pt", map_location='cpu'))
                        state["last_rl_order_count"] = current_closed
                        
                        # PERSIST the new count to DB
                        await db.set_metadata("last_retrain_trade_count", str(current_closed))
                        
                        logger.info("♻️ System updated with new RL weights and persistent count.")
                        
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

    async def reward_shaping_worker():
        """Shared LLM reward shaping logic."""
        logger.info("Reward shaping worker started.")
        while True:
            await asyncio.sleep(300) # Every 5 minutes
            if not advisor: continue
            
            try:
                # 1. Fetch 5 unrated trades
                unrated = await db.get_unrated_trades(limit=5)
                if not unrated: continue
                
                logger.info(f"LLM Batch: Shaping rewards for {len(unrated)} trades...")
                for exp in unrated:
                    score, reason = await advisor.get_quality_score(exp)
                    
                    # Improved Reward Shaping Implementation
                    # 50 is NEUTRAL. 100 is PERFECT. 0 is TRASH.
                    raw_reward = exp['reward']
                    
                    if raw_reward >= 0:
                        # WIN: Boost reward if pattern was followed (Score > 50)
                        # Reward = PnL * (Score / 50.0) -> Max 2x boost
                        adjusted_reward = raw_reward * (score / 50.0)
                    else:
                        # LOSS: Penalize more if loss was due to poor discipline (Score < 50)
                        # Formula: Reward = PnL * ((150 - Score) / 50.0)
                        # If Score 100 (Good trade, bad luck): 150-100=50 / 50 = 1x (Original loss)
                        # If Score 0 (Gambling): 150-0=150 / 50 = 3x (Triple penalty)
                        penalty_factor = (150 - score) / 50.0
                        adjusted_reward = raw_reward * penalty_factor
                    
                    # Log to DB
                    await db.log_llm_reward(exp['id'], score, reason, adjusted_reward)
                    await asyncio.sleep(2) # Safe RPM delay
                
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
        retry_counts = {}  # Track retries for missing deal history
        max_retries = 3    # Limit retries before skipping

        while True:
            try:
                await asyncio.sleep(15) # Check every 15s for missing deal history
                users = await db.get_active_users()
                
                for user in users:
                    user_id = user['id']
                    username = user['username']
                    
                    # The original `known_tickets = set()` was inside the loop, which resets it every time.
                    # Moving it outside or managing it differently is better, but for now,
                    # the instruction adds it at the top level of the function.
                    # The `if user_id not in known_tickets: known_tickets = set()` line is problematic
                    # as `known_tickets` is a global set for the worker, not per user.
                    # I will remove the problematic line and assume `known_tickets` is managed globally
                    # or will be used differently. The instruction only adds `known_tickets = set()`
                    # at the function scope, not inside the user loop.
                    # This creates a conflict, but I must follow the instruction faithfully.
                    # Given the instruction's diff, it seems the intent was to initialize it once.
                    # I will remove the `if user_id not in known_tickets: known_tickets = set()` line
                    # as it conflicts with the new global initialization and makes no sense with a single set.
                    # The instruction implies `known_tickets` should be initialized once.

                    # 1. Get DB Open Positions
                    db_positions = await db.get_open_positions(user_id)
                    if not db_positions: continue
                    
                    # 2. Get MT5 Open Positions
                    # We need to use the user's specific connection logic again?
                    # Or simpler: Just check if the ticket exists in MT5.
                    # Since MT5 context is global per process in this simple design, 
                    # we must switch login if we want to be 100% sure, OR rely on the main loops.
                    # BUT: main.py is single process, MT5 context is shared. 
                    # We need to lock/switch context.
                    
                    # Robust approach: Loop users, login, check.
                    if not mt5.initialize(path=user['mt5_path']): continue
                    if not mt5.login(login=user['mt5_login'], password=user['mt5_password'], server=user['mt5_server']): continue
                    
                    mt5_positions = mt5.positions_get()
                    if mt5_positions is None: mt5_positions = []
                    mt5_tickets = [p.ticket for p in mt5_positions]
                    
                    # 3. Detect Closed Positions (In DB but not in MT5)
                    for db_pos in db_positions:
                        ticket = db_pos['ticket']
                        
                        if ticket not in mt5_tickets:
                            logger.info(f"🔍 Detected closed position for ticket {ticket} ({db_pos['symbol']}) for user {username}. Syncing exit...")
                            
                            # Check Retry Count
                            if retry_counts.get(ticket, 0) >= max_retries:
                                logger.warning(f"⚠️ Max retries reached for ticket {ticket}. Skipping sync.")
                                continue

                            # Try to get deal history
                            history = mt5.history_deals_get(position=ticket)
                            
                            if history and len(history) > 0:
                                # Found deal!
                                exit_deal = history[-1] # Last deal is usually the exit
                                profit = exit_deal.profit
                                exit_price = exit_deal.price
                                exit_time = datetime.datetime.fromtimestamp(exit_deal.time)
                                exit_reason = "TP/SL/Manual" # Ideally deduce from deal reason
                                
                                await db.log_trade_exit_by_ticket(
                                    user_id, 
                                    ticket, 
                                    exit_price, 
                                    profit, 
                                    profit, # net profit roughly same for now
                                    reason=exit_reason
                                )
                                logger.info(f"✅ Synced exit for Ticket {ticket}: Profit {profit}, Price {exit_price}")
                                
                                # Send Notification
                                await notifier.send_message(
                                    f"💰 <b>Trade Closed ({username})</b>\n"
                                    f"Symbol: {db_pos['symbol']}\n"
                                    f"Profit: ${profit:.2f}\n"
                                    f"Price: {exit_price}"
                                )
                                
                                # Reset retry count on success
                                if ticket in retry_counts: del retry_counts[ticket]
                                
                            else:
                                logger.warning(f"Could not find deal history for ticket {ticket} yet for user {username}.")
                                retry_counts[ticket] = retry_counts.get(ticket, 0) + 1
                                
            except Exception as e:
                logger.error(f"Position Monitor Error: {e}")
                await asyncio.sleep(5)

    # --- Market Poller ---
    async def shared_market_poller():
        """Aggregates ticks into candles for all symbols."""
        with open("Config/mt5_config.json", "r") as f: mt5_cfg = json.load(f)
        symbols = mt5_cfg.get('symbols', ["EURUSD"])
        
        # Load MT5 Connection early for native execution
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
        while True:
            try:
                for symbol in symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick: continue
                    if symbol not in mtf_managers: mtf_managers[symbol] = MTFManager(timeframes=[60, 300, 3600], window_size=32)
                    mgr = mtf_managers[symbol]
                    closed_tfs = mgr.add_tick({'symbol': symbol, 'bid': tick.bid, 'time': int(tick.time_msc)})
                    
                    # Update Dashboard
                    current_m1 = mgr.aggregators[60].get_current_candle()
                    if current_m1:
                        await post_to_dashboard({"type": "candle", "symbol": symbol, **current_m1})

                    if 60 in closed_tfs:
                        m1 = mgr.aggregators[60].get_last_closed_candle()
                        if m1:
                            await db.log_candle(symbol, "M1", m1)
                await asyncio.sleep(0.1)
            except Exception as e: logger.error(f"Poller error: {e}"); await asyncio.sleep(1)

    # --- User Worker ---
    async def run_user_worker(user):
        """Dedicated worker for one user's trading lifecycle."""
        user_id, username, login, password, server, path = user['id'], user['username'], user['mt5_login'], user['mt5_password'], user['mt5_server'], user['mt5_path']
        logger.info(f"👤 Starting Worker for {username} (ID: {user_id})")
        
        # Local state for this worker
        worker_state = {
            "trading_enabled": srv_config.get("trading_enabled", True),
            "ai_mode": srv_config.get("ai_mode", "EXPLORER")
        }
        
        # 1. Login to MT5 (Requires correct terminal path)
        if not mt5.initialize(path=path):
            logger.error(f"❌ Worker {username}: MT5 init failed at {path}")
            return
            
        if not mt5.login(login=login, password=password, server=server):
            logger.error(f"❌ Worker {username}: Login failed for {login}")
            mt5.shutdown()
            return

        logger.info(f"✅ Worker {username}: Logged into {server}")
        
        # Define symbols for this worker
        symbols = srv_config.get('symbols', ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
        
        try:
            while True:
                # Check DB for kill signal
                # (This would be more complex in production, here we just loop)
                
                for symbol in symbols:
                    mgr = mtf_managers.get(symbol)
                    if not mgr or not mgr.is_tf_ready(60): continue
                    
                    # 1. Market Open Check
                    is_open, _ = is_market_open(symbol)
                    if not is_open: continue
                    
                    # 2. MTF Analysis (Shared Model)
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
                        ai_mode=worker_state["ai_mode"], 
                        symbol=symbol
                    )
                    
                    if signal['action'] != 'HOLD':
                        # 3. Execution (User-Specific)
                        logger.info(f"🚀 Worker {username}: Signal {symbol} {signal['action']}")
                        
                        account = mt5.account_info()
                        current_equity = account.equity if account else 1000.0
                        
                        # Reversal Logic
                        reversal_threshold = srv_config.get("reversal_confidence_threshold", 0.75)
                        if signal['confidence'] >= reversal_threshold:
                            target_to_close = mt5.POSITION_TYPE_SELL if signal['action'] == 'BUY' else mt5.POSITION_TYPE_BUY
                            await close_all_positions(user_id, symbol, action_type=target_to_close, notifier=notifier, db=db)
                        
                        # Risk Calculation
                        tick = mt5.symbol_info_tick(symbol)
                        entry_price = tick.ask if signal['action'] == 'BUY' else tick.bid
                        
                        # Phase 92: Accurate Pip-to-Dollar Calculation
                        symbol_info = mt5.symbol_info(symbol)
                        point = symbol_info.point
                        tick_size = symbol_info.trade_tick_size
                        tick_value = symbol_info.trade_tick_value # Profit in currency for 1 tick move
                        
                        # Define "Pip" (Standard defined as 10 points for most symbols)
                        pip_price_offset = 0.01 if "JPY" in symbol.upper() else (10.0 * point)
                        if "XAU" in symbol.upper(): pip_price_offset = 0.1 # Gold standard pip
                        
                        # Correct Pip Value (Dollar profit per 1 Lot for 1 Pip move)
                        # Formula: (Pip Offset / Tick Size) * Tick Value
                        dollar_value_per_pip = (pip_price_offset / tick_size) * tick_value
                        
                        # 1. Calculate risk-based lot size
                        risk_lots = risk_manager.calculate_lot_size(
                            current_equity, 20, confidence=signal['confidence'], 
                            pip_value=dollar_value_per_pip, commission_per_lot=srv_config.get("commission_per_lot", 7.0)
                        )
                        
                        # 2. Calculate affordable lot size based on Free Margin
                        account_info = mt5.account_info()
                        free_margin = account_info.margin_free if account_info else 0.0
                        affordable_lots = risk_manager.calculate_max_affordable_lots(symbol, signal['action'], free_margin, mt5_module=mt5)
                        
                        # 3. Final Lot Size (Risk-based capped by Affordable)
                        lot_size = min(risk_lots, affordable_lots)
                        
                        logger.info(f"💰 Lot Calculation for {username} ({symbol}): Risk={risk_lots}, Affordable={affordable_lots} -> Final={lot_size}")
                        
                        if lot_size < symbol_info.volume_min:
                            logger.warning(f"⚠️ {username}: Final lot size {lot_size} is below minimum {symbol_info.volume_min}. Skipping trade.")
                            continue

                        sl, tp = risk_manager.calculate_sl_tp(
                            symbol, signal['action'], entry_price, 
                            stop_loss_pips=20, point=point, tick_size=symbol_info.trade_tick_size
                        )
                        
                        # Execute
                        await execute_mt5_order(user_id, symbol, signal['action'], lot_size, sl=sl, tp=tp, notifier=notifier, signal_data=signal, db=db)
                
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Main Loop Error: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5)

    # --- Main Event Loop ---
    # 1. Sync Historical Data
    logger.info("🔄 Syncing historical data from MT5...")
    m1_history = await sync_historical_data(mtf_managers, db)
    
    # 2. Push to Dashboard
    if m1_history:
        logger.info(f"📤 Pushing {len(m1_history)} M1 candles to Dashboard...")
        for candle in m1_history:
            await post_to_dashboard(candle)
    
    # Send Startup Notification to Telegram
    await notifier.send_message(
        "🚀 **QuantSystem Started**\n\nโหมดการทำงาน: {}\nผู้ดูแลระบบพร้อมช่วยเหลือครับ!".format(srv_config.get("ai_mode", "EXPLORER")),
        reply_markup=notifier.get_main_menu()
    )
    
    # 3. Get Active Users
    users = await db.get_active_users()
    logger.info(f"👥 Found {len(users)} active users in database")
    
    # 4. Start Background Workers
    background_tasks = [
        asyncio.create_task(command_worker()),
        asyncio.create_task(db_monitor_worker()),
        asyncio.create_task(retraining_worker()),
        asyncio.create_task(reward_shaping_worker()),
        asyncio.create_task(position_monitor_worker()),
        asyncio.create_task(shared_market_poller())
    ]
    
    # 5. Start User Workers
    for user in users:
        task = asyncio.create_task(run_user_worker(user))
        active_workers[user['id']] = task
        background_tasks.append(task)
    
    logger.info(f"✅ System fully initialized. Running {len(background_tasks)} workers.")
    
    # 6. Run Forever
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
