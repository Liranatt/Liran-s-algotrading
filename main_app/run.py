# main_app/run.py

# --- NEW: Add project root to Python's path ---
# This allows the program to find the 'strategies' module, which is in the parent directory.
import sys
import os

# Get the path of the directory containing this script ('main_app')
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory ('my_trading_platform')
project_root = os.path.dirname(current_dir)
# Add the parent directory to Python's list of places to look for modules
sys.path.append(project_root)
# --- END OF NEW CODE ---


# Now we can import the other modules
import argparse
from main_app.portfolio_manager import PortfolioManager


def main():
    """
    Main entry point to the trading platform.
    Use command-line arguments to select the run mode.
    """
    # Set up the argument parser to read command-line inputs.
    parser = argparse.ArgumentParser(description="A multi-strategy trading platform.")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['live', 'backtest'],
        help="Select the operational mode: 'live' for real trading, 'backtest' for simulation."
    )
    args = parser.parse_args()

    print(f"--- Initializing Trading Platform in {args.mode.upper()} mode ---")

    manager = PortfolioManager()

    if args.mode == 'live':
        manager.run_trading_cycle()
    elif args.mode == 'backtest':
        manager.run_backtest()

    print(f"--- Platform Run Complete ---")


if __name__ == "__main__":
    main()
