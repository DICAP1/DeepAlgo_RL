import numpy as np
import pandas as pd


def allocate_portfolio(trade_signals, capital=10000):
    """
    Allocates portfolio weights across multiple assets dynamically.

    Parameters:
    - trade_signals (dict): { "EUR_USD": "BUY", "XAU_USD": "SELL", ... }
    - capital (float): Total capital available for trading.

    Returns:
    - allocation (dict): { "EUR_USD": 2500, "XAU_USD": -2500, ... }
    """
    num_assets = len(trade_signals)
    if num_assets == 0:
        return {}

    # Equal allocation per asset
    allocation_per_asset = capital / num_assets

    allocation = {}
    for asset, signal in trade_signals.items():
        if signal == "BUY":
            allocation[asset] = allocation_per_asset
        elif signal == "SELL":
            allocation[asset] = -allocation_per_asset
        else:
            allocation[asset] = 0  # No trade

    return allocation
