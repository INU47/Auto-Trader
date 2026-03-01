import asyncio
import logging
import json
import os
import MetaTrader5 as mt5
from Database.db_handler import DBHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataSyncInitializer")

async def run_initial_sync():
    db = DBHandler()
    await db.connect()
    
    config_path = "Config/mt5_config.json"
    if not os.path.exists(config_path):
        logger.error("MT5 config not found!")
        return
        
    with open(config_path, 'r') as f:
        mt5_config = json.load(f)
        
    symbols = mt5_config.get('symbols', ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    timeframes = ['H1', 'M5', 'M1']
    target = 100000

    logger.info("Initializing MT5 for bulk sync...")
    if not mt5.initialize(login=mt5_config['login'], server=mt5_config['server'], password=mt5_config['password']):
        logger.error("MT5 Init Failed!")
        return

    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"⏳ Syncing {symbol} {tf} (Target: {target})...")
            success = await db.ensure_data_continuity(symbol, tf, target_candles=target)
            if success:
                logger.info(f"✅ {symbol} {tf} Sync Complete.")
            else:
                logger.error(f"❌ {symbol} {tf} Sync Failed.")

    mt5.shutdown()
    await db.close()
    logger.info("🎉 Global Data Sync Complete!")

if __name__ == "__main__":
    asyncio.run(run_initial_sync())
