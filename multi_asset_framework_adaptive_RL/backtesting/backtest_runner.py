import pandas as pd
from multi_asset_framework.strategies.ppo_signal_generator import generate_trade_signal
from multi_asset_framework.strategies.rule_based import apply_rule_based_strategy
from multi_asset_framework.utils import compute_indicators


def walk_forward_backtest(df, assets, train_window=50, test_window=10):
    """
    Walk-Forward Backtest: Iteratively trains and tests the strategy over rolling windows.

    Parameters:
    - df (pd.DataFrame): Market data with OHLC prices.
    - assets (list): List of instruments to backtest.
    - train_window (int): Number of bars used for training.
    - test_window (int): Number of bars used for testing.

    Returns:
    - results_df (pd.DataFrame): DataFrame with trade signals and simulated results.
    """

    all_results = []  # Store results for all assets

    for asset in assets:
        asset_df = df[df["ticker"] == asset].copy()  # Filter for the asset
        asset_df = compute_indicators(asset_df)

        asset_df["PPO_Signal"] = "NO SIGNAL"
        asset_df["Rule_Based_Signal"] = "NO SIGNAL"

        num_rows = len(asset_df)

        for start in range(0, num_rows - train_window - test_window, test_window):
            train_data = asset_df.iloc[start: start + train_window]  # Training window
            test_data = asset_df.iloc[start + train_window: start + train_window + test_window]  # Test window

            # Apply PPO model & Rule-based strategy only on test window
            for i in test_data.index:
                asset_df.at[i, "PPO_Signal"] = generate_trade_signal(asset_df.loc[:i])
                asset_df.at[i, "Rule_Based_Signal"] = apply_rule_based_strategy(asset_df.loc[:i])

        all_results.append(asset_df)

    # Combine results for all assets
    results_df = pd.concat(all_results)
    return results_df


# === Example Usage ===
if __name__ == "__main__":
    import oanda_api  # Ensure API is correctly set up

    assets = ["EUR_USD", "XAU_USD", "SPX500_USD"]

    # Fetch market data from OANDA
    df = oanda_api.fetch_market_data(assets, count=500, granularity="M5")

    # Run backtest
    results = walk_forward_backtest(df, assets)

    # Save to CSV for analysis
    results.to_csv("backtest_results.csv", index=False)
    print("✅ Walk-forward backtest completed. Results saved.")
