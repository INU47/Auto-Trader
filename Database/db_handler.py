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
        if not self.pool: return False
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def initialize_schema(self):
        if not self.pool: return
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                async with self.pool.acquire() as conn:
                    await conn.execute(schema_sql)
                    
                    await conn.execute("ALTER TABLE trade_logs ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
                    await conn.execute("ALTER TABLE system_metadata ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
                    
                    migration_sql = """
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'system_metadata_key_user_id_key') THEN 
                            ALTER TABLE system_metadata ADD CONSTRAINT system_metadata_key_user_id_key UNIQUE (key, user_id); 
                        END IF; 
                    END $$;
                    """
                    await conn.execute(migration_sql)
                    
                logger.info("Database schema verified/initialized/migrated.")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")

    def _to_datetime(self, timestamp):
        if not timestamp: return datetime.now()
        if isinstance(timestamp, (int, float)):
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

    async def get_candles(self, symbol, timeframe, days=7, limit=None):
        if not self.pool: return []
        try:
            query = """
                SELECT time, open, high, low, close, volume as tick_volume
                FROM market_candles
                WHERE symbol = $1 AND timeframe = $2
            """
            params = [symbol, timeframe]
            
            if days:
                query += " AND time > NOW() - (INTERVAL '1 day' * $3)"
                params.append(days)
            
            query += " ORDER BY time ASC"
            
            if limit:
                query += f" LIMIT {limit}"

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                candles = []
                for r in rows:
                    candles.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'time': int(r['time'].timestamp()),
                        'open': float(r['open']),
                        'high': float(r['high']),
                        'low': float(r['low']),
                        'close': float(r['close']),
                        'tick_volume': int(r['tick_volume']),
                        'sentiment': 0.0
                    })
                return candles
        except Exception as e:
            logger.error(f"Failed to fetch candles from DB: {e}")
            return []

    async def clear_market_data(self):
        if not self.pool: return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE market_candles;")
            logger.info("🔥 DB Reset: market_candles table truncated.")
        except Exception as e:
            logger.error(f"Failed to truncate market_candles: {e}")

    async def ensure_data_continuity(self, symbol, timeframe, target_candles=100000):
        if not self.pool: return False
        
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT COUNT(*) as count, MIN(time) as start_time, MAX(time) as end_time
                FROM market_candles
                WHERE symbol = $1 AND timeframe = $2
            """, symbol, timeframe)
        
        count = stats['count'] if stats['count'] else 0
        start_ts = int(stats['start_time'].timestamp()) if stats['start_time'] else 0
        end_ts = int(stats['end_time'].timestamp()) if stats['end_time'] else 0
        
        logger.info(f"🔍 Checking continuity for {symbol} {timeframe}: {count}/{target_candles} candles.")

        import MetaTrader5 as mt5
        mt5_tf = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'D1': mt5.TIMEFRAME_D1
        }.get(timeframe, mt5.TIMEFRAME_H1)

        if count < target_candles:
            needed = target_candles - count
            logger.info(f"⏳ Backfilling {needed} candles for {symbol} {timeframe}...")
            
            if start_ts > 0:
                rates = mt5.copy_rates_from(symbol, mt5_tf, start_ts, needed + 1)
            else:
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, target_candles)
            
            if rates is not None and len(rates) > 0:
                batch = []
                for r in rates:
                    batch.append({
                        'symbol': symbol, 'timeframe': timeframe,
                        'time': int(r['time']), 'open': r['open'], 'high': r['high'],
                        'low': r['low'], 'close': r['close'], 'tick_volume': r['tick_volume'],
                        'sentiment': 0.0
                    })
                await self.log_candles_batch(batch)
                logger.info(f"✅ Backfilled {len(batch)} candles from MT5.")

        tf_sec = 60 if timeframe == 'M1' else 300 if timeframe == 'M5' else 3600 if timeframe == 'H1' else 86400
        
        if count > 1:
            expected_count = (end_ts - start_ts) // tf_sec
            if count < expected_count * 0.95:
                logger.warning(f"⚠️ Potential gaps detected in {symbol} {timeframe} ({count}/{expected_count}). Running deep sync...")
                rates = mt5.copy_rates_range(symbol, mt5_tf, start_ts, end_ts)
                if rates is not None:
                    batch = []
                    for r in rates:
                        batch.append({
                            'symbol': symbol, 'timeframe': timeframe,
                            'time': int(r['time']), 'open': r['open'], 'high': r['high'],
                            'low': r['low'], 'close': r['close'], 'tick_volume': r['tick_volume']
                        })
                    await self.log_candles_batch(batch)

        import time as pytime
        current_ts = int(pytime.time())
        if end_ts > 0 and (current_ts - end_ts) > (tf_sec * 1.5):
            logger.info(f"⏩ Forward-filling {symbol} {timeframe} to current price...")
            rates = mt5.copy_rates_range(symbol, mt5_tf, end_ts, current_ts)
            if rates is not None and len(rates) > 0:
                batch = []
                for r in rates:
                    batch.append({
                        'symbol': symbol, 'timeframe': timeframe,
                        'time': int(r['time']), 'open': r['open'], 'high': r['high'],
                        'low': r['low'], 'close': r['close'], 'tick_volume': r['tick_volume']
                    })
                await self.log_candles_batch(batch)
                logger.info(f"✅ Forward-fill complete ({len(batch)} candles).")

        return True

    async def log_trade_entry(self, user_id, symbol, action, lot_size, price, signal_data, ticket=0):
        if not self.pool: return None
        try:
            query = """
                INSERT INTO trade_logs (
                    user_id, open_time, symbol, action, open_price, lot_size,
                    ai_mode, pattern_type, cnn_confidence, lstm_trend_pred, lstm_confidence, status, ticket
                ) VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7, $8, $9, $10, 'OPEN', $11)
                RETURNING id
            """
            
            raw_pattern = signal_data.get('raw_cnn_class', 0)
            raw_trend = signal_data.get('raw_lstm_trend', 0.0)
            raw_conf = signal_data.get('raw_lstm_conf', 0.0)
            ai_mode = signal_data.get('ai_mode', 'CONSERVATIVE')
            
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
        if not self.pool: return
        try:
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
        if not self.pool: return []
        try:
            experiences = []
            async with self.pool.acquire() as conn:
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
                    mtf_state = {}
                    
                    for tf_label in ['M1', 'M5', 'H1']:
                        candles = await conn.fetch("""
                            SELECT open, high, low, close, volume as tick_volume
                            FROM market_candles
                            WHERE symbol = $1 AND timeframe = $2 AND time < $3
                            ORDER BY time DESC LIMIT $4
                        """, t['symbol'], tf_label, t['open_time'], window_size)
                        
                        if len(candles) == window_size:
                            mtf_state[tf_label] = [dict(c) for c in reversed(candles)]
                    
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
                            'state': mtf_state
                        })
            
            logger.info(f"Reconstructed {len(experiences)} RL experiences from DB.")
            return experiences
        except Exception as e:
            logger.error(f"Failed to fetch RL training data: {e}")
            return []

    async def get_unrated_trades(self, limit=5, window_size=32):
        if not self.pool: return []
        try:
            unrated = []
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, symbol, action, open_price, close_price, pattern_type, cnn_confidence, lstm_confidence, open_time, close_time, net_profit 
                    FROM trade_logs t
                    WHERE status = 'CLOSED' 
                      AND NOT EXISTS (SELECT 1 FROM llm_training_logs l WHERE l.trade_id = t.id)
                    ORDER BY open_time ASC LIMIT $1
                """
                trades = await conn.fetch(query, limit)
                
                for t in trades:
                    duration_min = 0
                    if t['open_time'] and t['close_time']:
                        diff = t['close_time'] - t['open_time']
                        duration_min = round(diff.total_seconds() / 60.0, 1)

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
                            'action': t['action'],
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



    async def get_or_create_user(self, username, mt5_login, mt5_password, mt5_server, mt5_path=None):
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
        if not self.pool: return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM users WHERE is_active = TRUE")
                return rows
        except Exception as e:
            logger.error(f"Failed to fetch active users: {e}")
            return []

    async def get_metadata(self, key, user_id=None, default=None):
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
        if not self.pool: return
        try:
            if user_id is None:
                query = """
                    INSERT INTO system_metadata (key, value, user_id) VALUES ($1, $2, NULL)
                    ON CONFLICT (key) WHERE user_id IS NULL 
                    DO UPDATE SET value = EXCLUDED.value
                """
                async with self.pool.acquire() as conn:
                    await conn.execute(query, key, str(value))
            else:
                query = """
                    INSERT INTO system_metadata (key, value, user_id) VALUES ($1, $2, $3)
                    ON CONFLICT (user_id, key) WHERE user_id IS NOT NULL
                    DO UPDATE SET value = EXCLUDED.value
                """
                async with self.pool.acquire() as conn:
                    await conn.execute(query, key, str(value), user_id)
        except Exception as e:
            logger.error(f"Failed to set metadata {key} (User: {user_id}): {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("DB Connection closed.")
