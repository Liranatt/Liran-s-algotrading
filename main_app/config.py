# main_app/config.py

import os

# --- FILE STRUCTURE CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data_files')
LOG_DIR = os.path.join(BASE_DIR, 'logs') # For live trading logs
STRATEGY_DIR = os.path.join(BASE_DIR, 'strategies')
BACKTEST_OUTPUT_DIR = os.path.join(BASE_DIR, 'backtests') # New backtest output folder

# --- FILE PATHS ---
SNP500_FILE = os.path.join(DATA_DIR, 'snp500.csv')
POSITIONS_FILE = os.path.join(DATA_DIR, 'positions.csv')
ALLOCATIONS_FILE = os.path.join(DATA_DIR, 'allocations.csv')
MANAGER_LOG_FILE = os.path.join(LOG_DIR, 'manager.log') # Live manager log
# NEW: Path for the cached data file
DATA_CACHE_FILE = os.path.join(DATA_DIR, 'market_data_cache.pkl')


# --- IBKR ACCOUNT & CONNECTION ---
IB_HOST = '127.0.0.1'
IB_PORT = 7497
IB_CLIENT_ID = 1

# --- BACKTESTING PARAMETERS ---
BACKTEST_STARTING_CAPITAL = 2000.00
BACKTEST_DAYS = 3000 # Your new value
EQUITY_CURVE_FILENAME = "backtest_equity_curve.png"

# --- STRATEGY-SPECIFIC PARAMETERS ---
STRATEGY_PARAMS = {
    'roi_bot': {
        'class_name': 'RoiStrategy',
        'log_file_name': 'roi_bot.log', # We only need the filename now
        'sma_window': 7,
        'fall_threshold': 0.20,
        'recovery_threshold': 0.95,
        'buy_threshold': 0.85,
        'stop_loss_threshold': 0.95
    }
}