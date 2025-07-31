# strategies/ml_bot_strategy.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class MLStrategy:
    """
    Machine Learning strategy that predicts if closing price will be higher than opening price.
    Integrates with the existing trading platform architecture.
    """

    def __init__(self, params, logger):
        """
        Initialize the ML strategy with parameters and logger.
        """
        self.params = params
        self.logger = logger
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_path = params.get('model_path', 'ml_trading_model.pkl')
        self.min_probability = params.get('min_probability', 0.6)
        self.max_positions = params.get('max_positions', 10)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.02)
        self.take_profit_pct = params.get('take_profit_pct', 0.03)

        self.logger.info("ML Strategy initialized.")

        # Try to load existing model
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            self.logger.warning(f"Model file {self.model_path} not found. Will need to train model first.")

    def _load_model(self):
        """Load the pre-trained ML model."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.logger.info(f"ML model loaded successfully from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None

    def generate_signals(self, market_data, portfolio_positions, available_capital):
        """
        Main function called by PortfolioManager to get trade signals.
        """
        self.logger.info(f"Generating ML signals with available capital: ${available_capital:.2f}")
        signals = []

        if self.model is None:
            self.logger.warning("ML model not loaded. Cannot generate signals.")
            return signals

        # 1. Generate sell signals first
        sell_signals = self._generate_sell_signals(market_data, portfolio_positions)
        signals.extend(sell_signals)

        # 2. Generate buy signals if we have capacity and capital
        if len(portfolio_positions) < self.max_positions and available_capital > 0:
            buy_signals = self._generate_buy_signals(market_data, available_capital, len(portfolio_positions))
            signals.extend(buy_signals)

        self.logger.info(
            f"Generated {len(signals)} total signals ({len([s for s in signals if s['action'] == 'BUY'])} BUY, {len([s for s in signals if s['action'] == 'SELL'])} SELL)")
        return signals

    def _generate_sell_signals(self, market_data, portfolio_positions):
        """Generate sell signals based on stop loss and take profit."""
        sell_signals = []

        for pos in portfolio_positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            buy_price = pos['buy_price']

            if symbol not in market_data['stocks'] or market_data['stocks'][symbol].empty:
                continue

            try:
                current_price = market_data['stocks'][symbol]['Close'].iloc[-1]
                return_pct = (current_price - buy_price) / buy_price

                # Stop loss
                if return_pct <= -self.stop_loss_pct:
                    self.logger.info(f"SELL signal (Stop Loss) for {symbol}: {return_pct:.2%}")
                    sell_signals.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'qty': quantity
                    })
                # Take profit
                elif return_pct >= self.take_profit_pct:
                    self.logger.info(f"SELL signal (Take Profit) for {symbol}: {return_pct:.2%}")
                    sell_signals.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'qty': quantity
                    })
                # Check ML prediction for exit
                else:
                    # Get ML prediction for this stock
                    prediction, probability = self._predict_stock(market_data['stocks'][symbol])
                    if prediction is not None and prediction == 0 and probability < 0.4:
                        self.logger.info(f"SELL signal (ML Prediction) for {symbol}: Low probability {probability:.2%}")
                        sell_signals.append({
                            'action': 'SELL',
                            'symbol': symbol,
                            'qty': quantity
                        })

            except Exception as e:
                self.logger.error(f"Error generating sell signal for {symbol}: {e}")

        return sell_signals

    def _generate_buy_signals(self, market_data, available_capital, current_positions):
        """Generate buy signals based on ML predictions."""
        buy_signals = []
        stock_data = market_data['stocks']

        # Get all available stocks and their ML predictions
        predictions = []

        for symbol, hist in stock_data.items():
            if hist.empty or len(hist) < 50:  # Need enough data for features
                continue

            try:
                prediction, probability = self._predict_stock(hist)
                if prediction == 1 and probability >= self.min_probability:
                    current_price = hist['Close'].iloc[-1]
                    predictions.append({
                        'symbol': symbol,
                        'probability': probability,
                        'price': current_price
                    })
            except Exception as e:
                self.logger.debug(f"Error predicting {symbol}: {e}")
                continue

        if not predictions:
            self.logger.info("No stocks meet ML criteria for buying.")
            return buy_signals

        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        # Calculate how many positions we can add
        max_new_positions = self.max_positions - current_positions

        # Calculate capital per position
        if max_new_positions > 0:
            capital_per_position = available_capital / max_new_positions

            for pred in predictions[:max_new_positions]:
                symbol = pred['symbol']
                price = pred['price']
                probability = pred['probability']

                if capital_per_position > price:
                    quantity = int(capital_per_position // price)
                    if quantity > 0:
                        self.logger.info(
                            f"BUY signal for {quantity} shares of {symbol} at ${price:.2f} (prob: {probability:.2%})")
                        buy_signals.append({
                            'action': 'BUY',
                            'symbol': symbol,
                            'qty': quantity
                        })

        return buy_signals

    def _predict_stock(self, stock_hist):
        """Make ML prediction for a single stock."""
        try:
            # Calculate features
            featured_df = self._calculate_features(stock_hist)
            if featured_df is None:
                return None, None

            # Get latest features
            latest_features = featured_df[self.feature_columns].iloc[-1:].dropna()
            if len(latest_features) == 0:
                return None, None

            # Scale and predict
            scaled_features = self.scaler.transform(latest_features)
            prediction = self.model.predict(scaled_features)[0]
            probability = self.model.predict_proba(scaled_features)[0][1]

            return prediction, probability

        except Exception as e:
            self.logger.debug(f"Error in prediction: {e}")
            return None, None

    def _calculate_features(self, df):
        """Calculate technical features for ML model."""
        if len(df) < 50:  # Need enough data
            return None

        data = df.copy()

        try:
            # Basic returns
            data['returns_1d'] = data['Close'].pct_change(1)
            data['returns_5d'] = data['Close'].pct_change(5)
            data['returns_10d'] = data['Close'].pct_change(10)

            # Moving averages
            data['sma_5'] = data['Close'].rolling(5).mean()
            data['sma_10'] = data['Close'].rolling(10).mean()
            data['sma_20'] = data['Close'].rolling(20).mean()

            # Price relative to moving averages
            data['price_vs_sma_5'] = data['Close'] / data['sma_5'] - 1
            data['price_vs_sma_20'] = data['Close'] / data['sma_20'] - 1

            # Volume features
            data['volume_sma'] = data['Volume'].rolling(20).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_sma']
            data['volume_change'] = data['Volume'].pct_change(1)

            # Technical indicators
            data['rsi'] = ta.rsi(data['Close'], length=14)

            # MACD
            macd_data = ta.macd(data['Close'])
            if macd_data is not None and not macd_data.empty:
                data['macd'] = macd_data.iloc[:, 0]
                data['macd_signal'] = macd_data.iloc[:, 1]
                data['macd_hist'] = macd_data.iloc[:, 2]
            else:
                data['macd'] = np.nan
                data['macd_signal'] = np.nan
                data['macd_hist'] = np.nan

            # Bollinger Bands
            bb_data = ta.bbands(data['Close'])
            if bb_data is not None and not bb_data.empty:
                data['bb_upper'] = bb_data.iloc[:, 0]
                data['bb_middle'] = bb_data.iloc[:, 1]
                data['bb_lower'] = bb_data.iloc[:, 2]
                data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            else:
                data['bb_position'] = np.nan

            # Volatility
            data['volatility'] = data['returns_1d'].rolling(20).std()

            # High-Low spread
            data['hl_spread'] = (data['High'] - data['Low']) / data['Close']

            return data

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return None

    def train_model(self, market_data, retrain=False):
        """
        Train the ML model using historical data.
        This method can be called separately to train/retrain the model.
        """
        if self.model is not None and not retrain:
            self.logger.info("Model already loaded. Use retrain=True to force retraining.")
            return

        self.logger.info("Starting ML model training...")

        # Prepare training data
        X, y = self._prepare_training_data(market_data)

        if len(X) == 0:
            self.logger.error("No training data available.")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))

        self.logger.info(f"Training accuracy: {train_acc:.4f}")
        self.logger.info(f"Testing accuracy: {test_acc:.4f}")

        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")

    def _prepare_training_data(self, market_data):
        """Prepare training dataset from market data."""
        all_data = []

        for symbol, hist in market_data['stocks'].items():
            if len(hist) < 100:  # Need enough data
                continue

            # Calculate features
            featured_df = self._calculate_features(hist)
            if featured_df is None:
                continue

            # Create target variable
            featured_df['target'] = (featured_df['Close'] > featured_df['Open']).astype(int)

            # Add to dataset
            all_data.append(featured_df)

        if not all_data:
            return pd.DataFrame(), pd.Series()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Define feature columns
        self.feature_columns = [
            'returns_1d', 'returns_5d', 'returns_10d',
            'price_vs_sma_5', 'price_vs_sma_20',
            'volume_ratio', 'volume_change',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'volatility', 'hl_spread'
        ]

        # Filter to available features and remove NaN
        available_features = [col for col in self.feature_columns if col in combined_df.columns]
        self.feature_columns = available_features

        model_data = combined_df[self.feature_columns + ['target']].dropna()

        self.logger.info(f"Training dataset: {len(model_data)} samples, {len(self.feature_columns)} features")

        return model_data[self.feature_columns], model_data['target']