# strategies/roi_bot_strategy.py

import pandas as pd


class RoiStrategy:
    """
    This strategy detects a market-wide fall, then targets the 20 "smallest"
    companies (based on share price) and buys those that are still priced low.
    """

    def __init__(self, params, logger):
        """
        Initializes the strategy with its specific parameters and a dedicated logger.
        """
        self.params = params
        self.logger = logger
        self.prefall_prices = {}
        self.fall_detected = False
        self.logger.info("ROI Strategy initialized.")

    def generate_signals(self, market_data, portfolio_positions, available_capital):
        """
        The main function that the PortfolioManager will call to get trade signals.
        This version contains corrected reset and buying logic.
        """
        self.logger.info(f"Generating signals with available capital: ${available_capital:.2f}")
        signals = []
        if 'sp500' not in market_data or market_data['sp500'].empty:
            self.logger.warning("S&P 500 data is missing, cannot generate signals.")
            return signals

        sp500_data = market_data['sp500']
        stock_data = market_data['stocks']

        # 1. Sell Logic: Always check for sell signals first.
        sell_signals = []
        for pos in portfolio_positions:
            symbol = pos['symbol']
            prefall_price = self.prefall_prices.get(symbol)
            if prefall_price and symbol in stock_data and not stock_data[symbol].empty:
                current_price = stock_data[symbol]['Close'].iloc[-1]
                if current_price >= self.params['recovery_threshold'] * prefall_price:
                    self.logger.info(f"SELL signal (Take Profit) for {symbol} at ${current_price:.2f}")
                    sell_signals.append({'action': 'SELL', 'symbol': symbol, 'qty': pos['quantity']})
                elif current_price <= self.params['stop_loss_threshold'] * pos['buy_price']:
                    self.logger.info(f"SELL signal (Stop Loss) for {symbol} at ${current_price:.2f}")
                    sell_signals.append({'action': 'SELL', 'symbol': symbol, 'qty': pos['quantity']})

        signals.extend(sell_signals)

        # 2. State Reset Logic: If we just sold ALL positions, reset the strategy state.
        if portfolio_positions and len(sell_signals) == len(portfolio_positions):
            self.logger.info("All strategy positions are being sold. Resetting state to detect the next fall.")
            self.fall_detected = False
            self.prefall_prices = {}

        # 3. Buy Logic
        if not self.fall_detected:
            self._detect_fall(sp500_data, stock_data)

        # If a fall is detected, look for buys, BUT ONLY IF WE DON'T ALREADY HAVE POSITIONS.
        # This ensures we buy a basket of stocks once per fall detection cycle.
        if self.fall_detected and not portfolio_positions:
            buy_candidates = self._find_buy_candidates(stock_data)
            if buy_candidates:
                capital_per_stock = available_capital / len(buy_candidates)
                for symbol in buy_candidates:
                    current_price = stock_data[symbol]['Close'].iloc[-1]
                    if current_price > 0 and capital_per_stock > current_price:
                        quantity = int(capital_per_stock // current_price)
                        if quantity > 0:
                            self.logger.info(f"BUY signal for {quantity} shares of {symbol} at ${current_price:.2f}")
                            signals.append({'action': 'BUY', 'symbol': symbol, 'qty': quantity})
        return signals

    def _detect_fall(self, sp500_data, stock_data):
        """Analyzes S&P 500 for a fall and snapshots prices."""
        rolling_max = sp500_data['Close'].rolling(window=self.params['sma_window']).max()
        pct_drop = (rolling_max - sp500_data['Close']) / rolling_max
        fall_days = pct_drop[pct_drop >= self.params['fall_threshold']]

        if not fall_days.empty:
            breach_date = fall_days.index[0]
            self.logger.warning(f"Market fall detected on {breach_date.date()}. Snapshotting prices.")
            self.fall_detected = True
            for sym, hist in stock_data.items():
                if not hist.empty:
                    price = hist[~hist.index.duplicated(keep='first')]['Close'].asof(breach_date)
                    if pd.notna(price):
                        self.prefall_prices[sym] = price

    # --- THIS METHOD CONTAINS YOUR NEW LOGIC ---
    def _find_buy_candidates(self, stock_data):
        """
        Finds the 20 "smallest" companies (using share price as a proxy)
        and then checks if they are trading below their buy threshold.
        """
        self.logger.info("Finding the 20 smallest companies based on share price...")

        # Step 1: Get the current price for all stocks to determine their "size"
        all_company_prices = []
        for symbol, hist in stock_data.items():
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                if current_price > 0:  # Exclude stocks with no price
                    all_company_prices.append((symbol, current_price))

        if not all_company_prices:
            return []

        # Step 2: Sort by price to find the 20 smallest and get their symbols
        all_company_prices.sort(key=lambda item: item[1])
        smallest_20_symbols = [symbol for symbol, price in all_company_prices[:20]]
        self.logger.info(f"Identified 20 smallest companies: {smallest_20_symbols}")

        # Step 3: From these 20, find the ones that meet our buy criteria
        candidates = []
        # Create a dictionary for quick price lookups
        price_dict = dict(all_company_prices)
        for symbol in smallest_20_symbols:
            prefall_price = self.prefall_prices.get(symbol)
            current_price = price_dict.get(symbol)

            if prefall_price and current_price:
                if current_price <= self.params['buy_threshold'] * prefall_price:
                    candidates.append(symbol)

        if candidates:
            self.logger.info(f"Found {len(candidates)} buy candidates from the smallest 20: {candidates}")
        return candidates