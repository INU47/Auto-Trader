-- Database Schema for Quant Trading System
-- Optimized for Reinforcement Learning (RL)

-- 1. Drop Old Tables (Cleanup)

-- 2. Create Tables

-- Table: users (Stores multiple accounts)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    mt5_login BIGINT NOT NULL,
    mt5_password TEXT NOT NULL, -- Encrypted
    mt5_server TEXT NOT NULL,
    mt5_path TEXT, -- Custom path to MT5 terminal
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table: market_candles (Shared across all users)
CREATE TABLE IF NOT EXISTS market_candles (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL, -- 'M1', 'M5', 'H1'
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Table: trade_logs (Full Trade Lifecycle with User Isolation)
CREATE TABLE IF NOT EXISTS trade_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    ticket BIGINT, -- MT5 Ticket ID
    symbol TEXT NOT NULL,
    action TEXT NOT NULL, -- 'BUY' or 'SELL'
    
    -- Entry Details
    open_time TIMESTAMPTZ NOT NULL,
    open_price DOUBLE PRECISION NOT NULL,
    lot_size DOUBLE PRECISION NOT NULL,
    
    -- Exit Details (Nullable until closed)
    close_time TIMESTAMPTZ,
    close_price DOUBLE PRECISION,
    gross_profit DOUBLE PRECISION, -- Raw Profit from MT5
    net_profit DOUBLE PRECISION,   -- After Swap/Comm
    
    -- AI State at Entry (The "Observation" & "Policy" for RL)
    ai_mode TEXT,
    pattern_type TEXT, -- CNN Class Name
    cnn_confidence DOUBLE PRECISION,
    lstm_trend_pred DOUBLE PRECISION,
    lstm_confidence DOUBLE PRECISION,
    
    -- Status & Analytics
    status TEXT DEFAULT 'OPEN', -- 'OPEN', 'CLOSED'
    close_reason TEXT
);

-- Table: llm_training_logs (LLM Mentor Feedback for RL Training)
CREATE TABLE IF NOT EXISTS llm_training_logs (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES trade_logs(id) ON DELETE CASCADE,
    quality_score INTEGER,
    reasoning TEXT,
    adjusted_reward DOUBLE PRECISION,
    is_exploration BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
    
-- Table: system_metadata (Persistent Key-Value Store)
CREATE TABLE IF NOT EXISTS system_metadata (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, -- NULL for global config
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    UNIQUE (user_id, key)
);



-- 3. Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_candles_time ON market_candles (time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_curr ON trade_logs (symbol, status);
