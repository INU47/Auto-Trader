import asyncpg
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger("DBHandler")

class DBHandler:
    def __init__(self, config_path="Config/server_config.json"):
        self.config_path = config_path
        self.pool = None

    async def connect(self):
        try:
            # Load from server_config
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                db_cfg = config.get("db_config", {})
                host = db_cfg.get("host", "127.0.0.1")
                user = db_cfg.get("user", "postgres")
                db_name = db_cfg.get("database", "quant_db")
                port = db_cfg.get("port", 5432)
                
                logger.info(f"Connecting to DB: {host}:{port} | User: {user} | DB: {db_name}")

                self.pool = await asyncpg.create_pool(
                    user=user,
                    password=db_cfg.get("password", "password"),
                    database=db_name,
                    host=host,
                    port=port
                )
                logger.info("Connected to PostgreSQL successfully.")
                await self.initialize_schema()
            else:
                logger.error(f"Config file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"DB Connection Failed: {e}")
            self.pool = None

    async def is_healthy(self):
        """Checks if DB connection is alive and reachable"""
        if not self.pool: return False
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def initialize_schema(self):
        """Automatically creates tables if they don't exist"""
        if not self.pool: return
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                async with self.pool.acquire() as conn:
                    await conn.execute(schema_sql)
                    
                    # Phase 93: Data Migration (Add user_id if missing)
                    await conn.execute("ALTER TABLE trade_logs ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
                    await conn.execute("ALTER TABLE system_metadata ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
                    # Migrate system_metadata keys if necessary (optional here)
                    
                logger.info("Database schema verified/initialized/migrated.")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")

    def _to_datetime(self, timestamp):
        """Converts timestamp (seconds or milliseconds) to datetime."""
        if not timestamp: return datetime.now()
        if isinstance(timestamp, (int, float)):
            # If > 1e11, it's likely milliseconds (e.g., 1700...000)
            if timestamp > 100000000000:
                return datetime.fromtimestamp(timestamp / 1000.0)
            return datetime.fromtimestamp(timestamp)
        return timestamp

    async def log_candle(self, symbol, timeframe, candle_data):
        if not self.pool: return
        try:
            query = """
                INSERT INTO market_candles (time, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (time, symbol, timeframe) DO NOTHING
            """
            ts = self._to_datetime(candle_data['time'])

            async with self.pool.acquire() as conn:
                await conn.execute(query, 
                                   ts, 
                                   symbol, 
                                   timeframe,
                                   float(candle_data['open']), 
                                   float(candle_data['high']),
                                   float(candle_data['low']), 
                                   float(candle_data['close']),
                                   float(candle_data.get('volume', candle_data.get('tick_volume', 0)) if isinstance(candle_data, dict) else 0))
        except Exception as e:
            logger.error(f"Failed to log candle: {e}")

    async def log_candles_batch(self, candles_list):
        """
        Efficiently logs a batch of candles.
        candles_list: List of dicts with keys: symbol, timeframe, time, open, high, low, close, tick_volume
        """
        if not self.pool or not candles_list: return
        try:
            query = """
                INSERT INTO market_candles (time, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (time, symbol, timeframe) DO NOTHING
            """
            
            data_tuples = []
            for c in candles_list:
                ts = self._to_datetime(c['time'])
                
                data_tuples.append((
                    ts,
                    c['symbol'],
                    c['timeframe'],
                    float(c['open']),
                    float(c['high']),
                    float(c['low']),
                    float(c['close']),
                    float(c.get('tick_volume', c.get('volume', 0)))
                ))

            async with self.pool.acquire() as conn:
                await conn.executemany(query, data_tuples)
            logger.info(f"Batch logged {len(data_tuples)} candles.")
        except Exception as e:
            logger.error(f"Failed to log candle batch: {e}")

    async def get_candles(self, symbol, timeframe, days=7):
        """
        Fetches historical candles from the database.
        """
        if not self.pool: return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT time, open, high, low, close, volume as tick_volume
                    FROM market_candles
                    WHERE symbol = $1 AND timeframe = $2 AND time > NOW() - (INTERVAL '1 day' * $3)
                    ORDER BY time ASC
                """, symbol, timeframe, days)
                
                candles = []
                for r in rows:
                    candles.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'time': int(r['time'].timestamp()), # Seconds for dashboard/backtest
                        'open': float(r['open']),
                        'high': float(r['high']),
                        'low': float(r['low']),
                        'close': float(r['close']),
                        'tick_volume': int(r['tick_volume'])
                    })
                return candles
        except Exception as e:
            logger.error(f"Failed to fetch candles from DB: {e}")
            return []

    async def log_trade_entry(self, user_id, symbol, action, lot_size, price, signal_data, ticket=0):
        """
        Logs the OPENING of a trade for a specific user. 
        Returns the database ID of the log entry.
        """
        if not self.pool: return None
        try:
            query = """
                INSERT INTO trade_logs (
                    user_id, open_time, symbol, action, open_price, lot_size,
                    ai_mode, pattern_type, cnn_confidence, lstm_trend_pred, lstm_confidence, status, ticket
                ) VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7, $8, $9, $10, 'OPEN', $11)
                RETURNING id
            """
            
            # Extract AI State
            raw_pattern = signal_data.get('raw_cnn_class', 0)
            raw_trend = signal_data.get('raw_lstm_trend', 0.0)
            raw_conf = signal_data.get('raw_lstm_conf', 0.0)
            ai_mode = signal_data.get('ai_mode', 'CONSERVATIVE')  # Get actual mode
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, 
                                          user_id,
                                          symbol, 
                                          action, 
                                          float(price), 
                                          float(lot_size),
                                          ai_mode,
                                          str(raw_pattern),
                                          float(signal_data.get('confidence', 0.0)),
                                          float(raw_trend),
                                          float(raw_conf),
                                          int(ticket))
                return row['id'] if row else None
        except Exception as e:
            logger.error(f"Failed to log trade entry: {e}")
            return None

    async def log_trade_exit(self, db_id, close_price, profit, net_profit, reason="UNKNOWN"):
        """Updates the trade log with CLOSE details"""
        if not self.pool or not db_id: return
        try:
            query = """
                UPDATE trade_logs 
                SET close_time = NOW(), 
                    close_price = $1, 
                    gross_profit = $2, 
                    net_profit = $3, 
                    status = 'CLOSED',
                    close_reason = $4
                WHERE id = $5
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, float(close_price), float(profit), float(net_profit), reason, db_id)
        except Exception as e:
            logger.error(f"Failed to log trade exit: {e}")

    async def log_trade_exit_by_ticket(self, user_id, ticket, close_price, profit, net_profit, reason="UNKNOWN"):
        """Updates the trade log with CLOSE details using MT5 Ticket and User ID"""
        if not self.pool or not ticket: return
        try:
            query = """
                UPDATE trade_logs 
                SET close_time = NOW(), 
                    close_price = $1, 
                    gross_profit = $2, 
                    net_profit = $3, 
                    status = 'CLOSED',
                    close_reason = $4
                WHERE user_id = $5 AND ticket = $6 AND status = 'OPEN'
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, float(close_price), float(profit), float(net_profit), reason, user_id, int(ticket))
        except Exception as e:
            logger.error(f"Failed to log trade exit by ticket (User: {user_id}): {e}")

    async def close_latest_trade(self, user_id, symbol, close_price, profit, net_profit, reason="UNKNOWN"):
        """Closes the latest OPEN trade for a user/symbol (FIFO logic)"""
        if not self.pool: return
        try:
            # Find the latest OPEN trade for this user and symbol
            find_query = """
                SELECT id FROM trade_logs 
                WHERE user_id = $1 AND symbol = $2 AND status = 'OPEN' 
                ORDER BY open_time DESC LIMIT 1
            """
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(find_query, user_id, symbol)
                if row:
                    await self.log_trade_exit(row['id'], close_price, profit, net_profit, reason=reason)
                else:
                    logger.warning(f"No OPEN trade found to close for {symbol} (User: {user_id})")
        except Exception as e:
            logger.error(f"Failed to close latest trade (User: {user_id}): {e}")

    async def count_total_trades(self, user_id=None):
        """Returns the total number of trades (Global or Per-User)."""
        if not self.pool: return 0
        try:
            async with self.pool.acquire() as conn:
                if user_id:
                    return await conn.fetchval("SELECT COUNT(*) FROM trade_logs WHERE user_id = $1", user_id)
                return await conn.fetchval("SELECT COUNT(*) FROM trade_logs")
        except Exception as e:
            logger.error(f"Failed to count total trades: {e}")
            return 0

    async def get_recent_signals(self, limit=5, offset=0, user_id=None):
        if not self.pool: return []
        try:
            where_clause = "WHERE user_id = $3" if user_id else ""
            query = f"""
                SELECT open_time as time, symbol, action as final_signal, cnn_confidence 
                FROM trade_logs 
                {where_clause}
                ORDER BY open_time DESC LIMIT $1 OFFSET $2
            """
            async with self.pool.acquire() as conn:
                if user_id:
                    rows = await conn.fetch(query, limit, offset, user_id)
                else:
                    rows = await conn.fetch(query, limit, offset)
                return rows
        except Exception as e:
            logger.error(f"Failed to fetch recent signals: {e}")
            return []

    async def count_closed_trades(self, user_id=None):
        if not self.pool: return 0
        try:
            async with self.pool.acquire() as conn:
                if user_id:
                    return await conn.fetchval("SELECT COUNT(*) FROM trade_logs WHERE status = 'CLOSED' AND user_id = $1", user_id)
                return await conn.fetchval("SELECT COUNT(*) FROM trade_logs WHERE status = 'CLOSED'")
        except Exception as e:
            logger.error(f"Failed to count closed trades: {e}")
            return 0

    async def count_open_trades(self, user_id=None):
        if not self.pool: return 0
        try:
            async with self.pool.acquire() as conn:
                if user_id:
                    return await conn.fetchval("SELECT COUNT(*) FROM trade_logs WHERE status = 'OPEN' AND user_id = $1", user_id)
                return await conn.fetchval("SELECT COUNT(*) FROM trade_logs WHERE status = 'OPEN'")
        except Exception as e:
            logger.error(f"Failed to count open trades: {e}")
            return 0

    async def get_open_positions(self, user_id):
        """Returns all OPEN positions for a specific user."""
        if not self.pool: return []
        try:
            query = """
                SELECT id, symbol, ticket, open_price, lot_size, action 
                FROM trade_logs 
                WHERE user_id = $1 AND status = 'OPEN'
            """
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, user_id)
        except Exception as e:
            logger.error(f"Failed to get open positions for user {user_id}: {e}")
            return []

    async def get_rl_training_data(self, limit=1000, window_size=32):
        """
        Fetches CLOSED trades and reconstructs the State (last 32 candles) for each.
        Returns: list of dicts { 'state': candles, 'action': 1 or 2, 'reward': net_profit }
        """
        if not self.pool: return []
        try:
            experiences = []
            async with self.pool.acquire() as conn:
                # 1. Get closed trades with LLM-Adjusted rewards if available
                trades = await conn.fetch("""
                    SELECT t.id, t.symbol, t.action, t.open_price, t.close_price, 
                           t.pattern_type, t.cnn_confidence, t.open_time,
                           COALESCE(l.adjusted_reward, t.net_profit) as final_reward
                    FROM trade_logs t
                    LEFT JOIN llm_training_logs l ON l.trade_id = t.id
                    WHERE t.status = 'CLOSED' 
                    ORDER BY t.open_time DESC LIMIT $1
                """, limit)
                
                for t in trades:
                    # 2. Reconstruct State: Get last 32 candles for ALL timeframes (M1, M5, H1)
                    # This aligns RL training with actual MTF inference
                    mtf_state = {}
                    
                    for tf_label in ['M1', 'M5', 'H1']:
                        candles = await conn.fetch("""
                            SELECT open, high, low, close, volume as tick_volume
                            FROM market_candles
                            WHERE symbol = $1 AND timeframe = $2 AND time < $3
                            ORDER BY time DESC LIMIT $4
                        """, t['symbol'], tf_label, t['open_time'], window_size)
                        
                        if len(candles) == window_size:
                            # Reverse list to get chronological order (Oldest to Newest)
                            mtf_state[tf_label] = [dict(c) for c in reversed(candles)]
                    
                    # Only append if we have all timeframes (to ensure complete state vector)
                    if len(mtf_state) == 3:
                        experiences.append({
                            'id': t['id'],
                            'symbol': t['symbol'],
                            'action': 1 if t['action'] == 'BUY' else 2,
                            'reward': float(t['final_reward'] or 0.0),
                            'open_price': float(t['open_price'] or 0.0),
                            'close_price': float(t['close_price'] or 0.0),
                            'pattern_name': t['pattern_type'],
                            'cnn_confidence': float(t['cnn_confidence'] or 0.0),
                            'state': mtf_state # Dict of lists
                        })
            
            logger.info(f"Reconstructed {len(experiences)} RL experiences from DB.")
            return experiences
        except Exception as e:
            logger.error(f"Failed to fetch RL training data: {e}")
            return []

    async def get_unrated_trades(self, limit=5, window_size=32):
        """
        Fetches CLOSED trades that haven't been scored by LLM yet.
        """
        if not self.pool: return []
        try:
            unrated = []
            async with self.pool.acquire() as conn:
                # Find trades where status='CLOSED' and no entry in llm_training_logs
                query = """
                    SELECT id, symbol, action, open_price, close_price, pattern_type, cnn_confidence, lstm_confidence, open_time, close_time, net_profit 
                    FROM trade_logs t
                    WHERE status = 'CLOSED' 
                      AND NOT EXISTS (SELECT 1 FROM llm_training_logs l WHERE l.trade_id = t.id)
                    ORDER BY open_time ASC LIMIT $1
                """
                trades = await conn.fetch(query, limit)
                
                for t in trades:
                    # Calculate duration in minutes
                    duration_min = 0
                    if t['open_time'] and t['close_time']:
                        diff = t['close_time'] - t['open_time']
                        duration_min = round(diff.total_seconds() / 60.0, 1)

                    # Reconstruct state for the advisor
                    candles = await conn.fetch("""
                        SELECT open, high, low, close, volume as tick_volume
                        FROM market_candles
                        WHERE symbol = $1 AND timeframe = 'M1' AND time < $2
                        ORDER BY time DESC LIMIT $3
                    """, t['symbol'], t['open_time'], window_size)
                    
                    if len(candles) == window_size:
                        state_candles = [dict(c) for c in reversed(candles)]
                        unrated.append({
                            'id': t['id'],
                            'symbol': t['symbol'],
                            'action': t['action'], # Pass original string (BUY/SELL)
                            'reward': float(t['net_profit'] or 0.0),
                            'open_price': float(t['open_price'] or 0.0),
                            'close_price': float(t['close_price'] or 0.0),
                            'open_time': t['open_time'].strftime('%Y-%m-%d %H:%M:%S') if t['open_time'] else "N/A",
                            'close_time': t['close_time'].strftime('%Y-%m-%d %H:%M:%S') if t['close_time'] else "N/A",
                            'duration_minutes': duration_min,
                            'pattern_name': t['pattern_type'],
                            'cnn_confidence': float(t['cnn_confidence'] or 0.0),
                            'lstm_confidence': float(t['lstm_confidence'] or 0.0),
                            'net_profit': float(t['net_profit'] or 0.0),
                            'state': state_candles
                        })
            return unrated
        except Exception as e:
            logger.error(f"Failed to fetch unrated trades: {e}")
            return []

    async def log_llm_reward(self, trade_id, quality_score, reasoning, adjusted_reward):
        """Logs LLM mentorship feedback for training audit."""
        if not self.pool: return
        try:
            query = """
                INSERT INTO llm_training_logs (trade_id, quality_score, reasoning, adjusted_reward)
                VALUES ($1, $2, $3, $4)
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, trade_id, int(quality_score), reasoning, float(adjusted_reward))
        except Exception as e:
            logger.error(f"Failed to log LLM reward: {e}")



    # User Management (Phase 93)
    
    async def get_or_create_user(self, username, mt5_login, mt5_password, mt5_server, mt5_path=None):
        """Ensures a user exists and returns their user_id."""
        if not self.pool: return None
        try:
            query = """
                INSERT INTO users (username, mt5_login, mt5_password, mt5_server, mt5_path)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (username) DO UPDATE 
                SET mt5_login = EXCLUDED.mt5_login, 
                    mt5_password = EXCLUDED.mt5_password,
                    mt5_server = EXCLUDED.mt5_server,
                    mt5_path = EXCLUDED.mt5_path
                RETURNING id
            """
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, username, int(mt5_login), str(mt5_password), mt5_server, mt5_path)
                return row['id'] if row else None
        except Exception as e:
            logger.error(f"Failed to get/create user {username}: {e}")
            return None

    async def get_active_users(self):
        """Returns all active users for the orchestrator."""
        if not self.pool: return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM users WHERE is_active = TRUE")
                return rows
        except Exception as e:
            logger.error(f"Failed to fetch active users: {e}")
            return []

    async def get_metadata(self, key, user_id=None, default=None):
        """Fetches a persistent property (Global or Per-User)."""
        if not self.pool: return default
        try:
            async with self.pool.acquire() as conn:
                if user_id:
                    val = await conn.fetchval("SELECT value FROM system_metadata WHERE key = $1 AND user_id = $2", key, user_id)
                else:
                    val = await conn.fetchval("SELECT value FROM system_metadata WHERE key = $1 AND user_id IS NULL", key)
                return val if val is not None else default
        except Exception as e:
            logger.error(f"Failed to fetch metadata {key} (User: {user_id}): {e}")
            return default

    async def set_metadata(self, key, value, user_id=None):
        """Stores or updates a persistent property (Global or Per-User)."""
        if not self.pool: return
        try:
            query = """
                INSERT INTO system_metadata (key, value, user_id) VALUES ($1, $2, $3)
                ON CONFLICT (key, user_id) DO UPDATE SET value = EXCLUDED.value
            """
            # Adjust mapping for NULL if user_id is None
            async with self.pool.acquire() as conn:
                await conn.execute(query, key, str(value), user_id)
        except Exception as e:
            logger.error(f"Failed to set metadata {key} (User: {user_id}): {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("DB Connection closed.")
