from multi_asset_framework.utils import compute_indicators
from multi_asset_framework.strategies.portfolio_allocator import allocate_portfolio
from multi_asset_framework.oanda_api import OandaV20API  # Ensure we use Oanda V20 API


def execute_multi_asset_trades(trade_signals, capital=10000):
    """
    Executes trades for multiple assets using allocated capital.
    Includes SL/TP using Chandelier Exit strategy.
    """
    api = OandaV20API()
    allocation = allocate_portfolio(trade_signals, capital)

    for asset, amount in allocation.items():
        if amount == 0:
            print(f"Skipping {asset}, no trade signal.")
            continue

        df = api.fetch_candles(asset)
        df.rename(columns={"close": "Close", "high": "High", "low": "Low"}, inplace=True)
        df = compute_indicators(df)

        side = "BUY" if amount > 0 else "SELL"
        api.place_trade(asset, side, abs(amount), df=df)

    print("✅ Multi-Asset Trades Executed.")

