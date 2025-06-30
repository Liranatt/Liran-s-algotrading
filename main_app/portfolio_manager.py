# main_app/portfolio_manager.py

import pandas as pd
import yfinance as yf
from ib_insync import IB, Stock, MarketOrder
import logging
import importlib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import config


class PortfolioManager:
    """
    The central orchestrator. Manages the master portfolio, fetches data,
    allocates capital to strategies, and executes all trades.
    """

    def __init__(self):
        self.ib = IB()
        self.logger = self._setup_logger(config.MANAGER_LOG_FILE)
        self.logger.info("Initializing Portfolio Manager...")
        self.positions_df = self._load_csv(config.POSITIONS_FILE, ['symbol', 'quantity', 'buy_price', 'strategy'])
        self.allocations_df = self._load_csv(config.ALLOCATIONS_FILE, ['strategy_name', 'allocation_percentage'])

    def _setup_logger(self, log_file_path):
        logger = logging.getLogger(log_file_path)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file_path, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        return logger

    def _load_csv(self, file_path, columns):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=columns)
            df.to_csv(file_path, index=False)
            self.logger.warning(f"Created or reset data file with headers: {file_path}")
            return df

    def _fetch_market_data(self, days):
        if os.path.exists(config.DATA_CACHE_FILE):
            self.logger.info(f"Loading market data from cache: {config.DATA_CACHE_FILE}")
            market_data = pd.read_pickle(config.DATA_CACHE_FILE)
            return market_data

        self.logger.info(f"No cache found. Fetching market data for the last {days} days...")
        symbols_to_fetch = pd.read_csv(config.SNP500_FILE).iloc[:, 0].dropna().tolist()
        if not symbols_to_fetch:
            raise ValueError("No symbols found in snp500.csv.")

        period_str = f"{days}d"
        sp500 = yf.download('^GSPC', period=period_str, interval="1d", auto_adjust=True)
        stocks = yf.download(symbols_to_fetch, period=period_str, interval="1d", auto_adjust=True, group_by='ticker')
        market_data = {'sp500': sp500,
                       'stocks': {sym: stocks[sym] for sym in symbols_to_fetch if not stocks[sym].empty}}
        pd.to_pickle(market_data, config.DATA_CACHE_FILE)
        self.logger.info(f"Market data saved to cache: {config.DATA_CACHE_FILE}")
        return market_data

    # --- Backtesting Methods ---

    def run_backtest(self):
        """
        Runs a full historical simulation, saving all output to a unique,
        timestamped folder in the 'backtests' directory.
        """
        run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join(config.BACKTEST_OUTPUT_DIR, f"run_{run_timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        manager_log_path = os.path.join(output_dir, 'manager.log')
        backtest_logger = self._setup_logger(manager_log_path)

        backtest_logger.info("--- Starting New BACKTEST ---")
        backtest_logger.info(f"Output will be saved to: {output_dir}")

        market_data_full = self._fetch_market_data(days=config.BACKTEST_DAYS)
        sp500_full = market_data_full['sp500']

        sim_cash = config.BACKTEST_STARTING_CAPITAL
        sim_positions = pd.DataFrame(columns=['symbol', 'quantity', 'buy_price', 'strategy'])
        equity_curve = []

        # Instantiate all strategies ONCE before the loop starts
        strategy_instances = self._initialize_strategies(output_dir)

        backtest_logger.info(f"Simulating from {sp500_full.index[0].date()} to {sp500_full.index[-1].date()}")
        for today in sp500_full.index:
            market_data_today = {
                'sp500': sp500_full.loc[:today],
                'stocks': {sym: df.loc[:today] for sym, df in market_data_full['stocks'].items() if
                           not df.loc[:today].empty}
            }

            current_market_value = 0
            if not sim_positions.empty:
                for _, pos in sim_positions.iterrows():
                    try:
                        current_price = market_data_today['stocks'][pos['symbol']]['Close'].iloc[-1]
                        current_market_value += pos['quantity'] * current_price
                    except (KeyError, IndexError):
                        current_market_value += pos['quantity'] * pos['buy_price']

            total_value = sim_cash + current_market_value
            equity_curve.append({'date': today, 'value': total_value})

            # --- THIS IS THE CORRECTED FUNCTION CALL ---
            all_signals = self._generate_all_signals(strategy_instances, market_data_today, total_value, sim_positions)

            if all_signals:
                sim_cash, sim_positions = self._simulate_trades(all_signals, market_data_today, sim_cash, sim_positions,
                                                                backtest_logger)

        backtest_logger.info("Backtest simulation finished. Processing results...")
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        plt.figure(figsize=(14, 7))
        plt.plot(equity_df['value'])
        plt.title(f'Portfolio Equity Curve - {run_timestamp}')
        plt.savefig(os.path.join(output_dir, config.EQUITY_CURVE_FILENAME))
        plt.close()
        final_value = equity_df['value'].iloc[-1]
        backtest_logger.info(f"Final portfolio value: ${final_value:,.2f}")
        backtest_logger.info("--- BACKTEST Complete ---")

    def _initialize_strategies(self, output_dir):
        """Initializes all strategy instances based on the allocations file."""
        strategy_instances = {}
        for _, alloc_row in self.allocations_df.iterrows():
            strategy_name = alloc_row['strategy_name']
            strategy_params = config.STRATEGY_PARAMS.get(strategy_name)
            if strategy_params:
                try:
                    strategy_log_path = os.path.join(output_dir, strategy_params['log_file_name'])
                    strategy_logger = self._setup_logger(strategy_log_path)
                    module_path = f"strategies.{strategy_name}_strategy"
                    StrategyClass = getattr(importlib.import_module(module_path), strategy_params['class_name'])
                    strategy_instances[strategy_name] = StrategyClass(strategy_params, strategy_logger)
                except Exception as e:
                    self.logger.error(f"Failed to initialize strategy '{strategy_name}': {e}", exc_info=True)
        return strategy_instances

    def _generate_all_signals(self, strategy_instances, market_data, portfolio_value, positions):
        """Gathers signals from all initialized strategy instances."""
        all_signals = []
        for strategy_name, instance in strategy_instances.items():
            percentage = \
            self.allocations_df[self.allocations_df['strategy_name'] == strategy_name]['allocation_percentage'].iloc[0]
            capital_budget = portfolio_value * (percentage / 100.0)
            strategy_positions = positions[positions['strategy'] == strategy_name].to_dict('records')

            try:
                signals = instance.generate_signals(market_data, strategy_positions, capital_budget)
                for signal in signals:
                    signal['strategy'] = strategy_name
                all_signals.extend(signals)
            except Exception as e:
                self.logger.error(f"Failed to get signals from strategy '{strategy_name}': {e}", exc_info=True)
        return all_signals

    def _simulate_trades(self, signals, market_data, cash, positions, logger):
        """Simulates the execution of trades for a single day in the backtest."""
        for signal in signals:
            symbol, action, qty, strategy = signal['symbol'], signal['action'], signal['qty'], signal['strategy']
            try:
                price = market_data['stocks'][symbol]['Close'].iloc[-1]
            except (KeyError, IndexError):
                logger.warning(f"Could not find price for {symbol} on this day. Skipping trade.")
                continue

            cost = qty * price
            if action == 'BUY' and cash >= cost:
                cash -= cost
                new_pos = {'symbol': symbol, 'quantity': qty, 'buy_price': price, 'strategy': strategy}
                new_row_df = pd.DataFrame([new_pos])
                positions = pd.concat([positions, new_row_df], ignore_index=True)
                logger.info(f"SIMULATED BUY: {qty} of {symbol} at ${price:.2f} for {strategy}")
            elif action == 'SELL':
                pos_index = positions[(positions['symbol'] == symbol) & (positions['strategy'] == strategy)].index
                if not pos_index.empty:
                    cash += cost
                    positions = positions.drop(pos_index)
                    logger.info(f"SIMULATED SELL: {qty} of {symbol} at ${price:.2f} for {strategy}")
        return cash, positions

    # --- Live Trading Methods (placeholder, not fully implemented in this refactor) ---
    def run_trading_cycle(self):
        """Executes one full trading cycle for LIVE trading."""
        self.logger.info("--- run_trading_cycle is not fully implemented in this version. ---")

    def _execute_live_trades(self, signals):
        """Connects to IB and executes a list of trade signals."""
        pass

    def _update_positions_csv(self, signal, price):
        """Updates the positions.csv file after a successful trade."""
        pass