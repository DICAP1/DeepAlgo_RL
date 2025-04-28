import v20
import pandas as pd
import time
...
time.sleep(1)  # Sleep 1 second between requests


class OandaV20API:
    def __init__(self):
        self.client = v20.Context(
            hostname="api-fxpractice.oanda.com",
            token="f7eff581944bb0b5efb4cac08003be9d-feea72696d43fb03101ddaa84eea2148"
        )
        self.account_id = "101-004-1683826-005"

    def fetch_candles(self, instrument, count=100, granularity="M5"):
        """
        Fetch historical candles from OANDA.
        """
        try:
            response = self.client.instrument.candles(
                instrument=instrument,
                count=count,
                granularity=granularity
            )

            # 🔹 Debugging: Print raw response
            print("DEBUG: Full OANDA API Response:", response)

            # 🔹 Extract response body correctly
            if hasattr(response, "body") and isinstance(response.body, dict):
                candles_data = response.body.get("candles", [])
            else:
                print("❌ ERROR: Unexpected response format:", response)
                return pd.DataFrame()  # Return empty DataFrame if response is invalid

            # 🔹 Print first candle to debug structure
            if candles_data:
                print("DEBUG: First Candle Data:", candles_data[0])

            # 🔹 Convert Candlestick objects to a usable format
            cleaned_candles = []
            for c in candles_data:
                if hasattr(c, "mid") and hasattr(c.mid, "c") and c.complete:
                    cleaned_candles.append({
                        "High": float(c.mid.h) if hasattr(c.mid, "h") else float(c.mid.c),  # Use Close if High missing
                        "Low": float(c.mid.l) if hasattr(c.mid, "l") else float(c.mid.c),  # Use Close if Low missing
                        "Close": float(c.mid.c)  # ✅ Corrected reference
                    })

            # 🔹 Convert to DataFrame
            df = pd.DataFrame(cleaned_candles)

            if df.empty:
                print("⚠️ WARNING: No valid candle data returned from OANDA API")
            else:
                print(f"✅ DEBUG: Successfully retrieved {len(df)} candles.")

            return df

        except Exception as e:  # ✅ Catch all exceptions safely
            print(f"❌ ERROR: OANDA API request failed: {e}")
            return pd.DataFrame()

    def fetch_training_data(self, instrument, total_candles=5000, granularity="M5"):
        """
        Fetch a large number of historical candles for training.
        """
        all_data = []
        count_per_request = 500
        loops = total_candles // count_per_request

        latest_time = None  # ✅ Start with no 'to' parameter (most recent data)

        for i in range(loops):
            print(f"🔄 Fetching batch {i + 1}/{loops}...")

            response = self.client.instrument.candles(
                instrument=instrument,
                count=count_per_request,
                granularity=granularity,
                to=latest_time
            )

            # ✅ FIXED: Safely extract response data
            if hasattr(response, "body") and isinstance(response.body, dict):
                candles_data = response.body.get("candles", [])
            else:
                print("❌ ERROR: Unexpected response format:", response)
                break

            cleaned_candles = []
            for c in candles_data:
                if hasattr(c, "mid") and hasattr(c.mid, "c") and c.complete:
                    cleaned_candles.append({
                        "Time": c.time,
                        "High": float(c.mid.h) if hasattr(c.mid, "h") else float(c.mid.c),
                        "Low": float(c.mid.l) if hasattr(c.mid, "l") else float(c.mid.c),
                        "Close": float(c.mid.c)
                    })

            if not cleaned_candles:
                print("⚠️ Skipping empty or incomplete batch.")
                continue

            df_chunk = pd.DataFrame(cleaned_candles)
            all_data.append(df_chunk)

            # ✅ Update the latest_time to the oldest candle from this batch to go back in time
            latest_time = candles_data[0].time

        full_df = pd.concat(all_data).drop_duplicates().sort_values("Time").reset_index(drop=True)
        full_df = full_df[["Time", "High", "Low", "Close"]]
        print(f"✅ Collected {len(full_df)} candles for {instrument}")
        return full_df

    def place_trade(self, instrument, side, amount, df=None,
                    atr_period=22, chandelier_multiplier=3.0, reward_risk_ratio=2.0):
        """
        Place a market trade on OANDA with Stop-Loss (SL) using Chandelier Exit
        and Take-Profit (TP) based on reward-risk ratio.

        Parameters:
        - instrument (str): OANDA instrument, e.g., "EUR_USD"
        - side (str): "BUY" or "SELL"
        - amount (int): Number of units to trade
        - df (DataFrame): Candle data with indicators
        - atr_period (int): Period to calculate Chandelier Exit
        - chandelier_multiplier (float): Multiplier for ATR-based SL
        - reward_risk_ratio (float): TP = SL distance * ratio
        """

        try:
            units = amount if side.upper() == "BUY" else -amount
            stop_loss_price = None
            take_profit_price = None

            if df is not None and len(df) >= atr_period:
                recent = df[-atr_period:]
                atr = recent["ATR"].iloc[-1]
                last_close = df["Close"].iloc[-1]

                if side.upper() == "BUY":
                    highest_high = recent["High"].max()
                    stop_loss_price = highest_high - chandelier_multiplier * atr
                    sl_distance = last_close - stop_loss_price
                    take_profit_price = last_close + reward_risk_ratio * sl_distance
                else:  # SELL
                    lowest_low = recent["Low"].min()
                    stop_loss_price = lowest_low + chandelier_multiplier * atr
                    sl_distance = stop_loss_price - last_close
                    take_profit_price = last_close - reward_risk_ratio * sl_distance

                print(f"📉 SL: {stop_loss_price:.5f}, 🎯 TP: {take_profit_price:.5f} for {side} {instrument}")

            # Build the order
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }

            if stop_loss_price:
                order_data["order"]["stopLossOnFill"] = {
                    "price": f"{stop_loss_price:.5f}"
                }

            if take_profit_price:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": f"{take_profit_price:.5f}"
                }

            response = self.client.order.market(self.account_id, body=order_data)
            print(f"✅ Trade executed: {side} {amount} units of {instrument}")
            return response

        except Exception as e:
            print(f"❌ ERROR: Trade execution failed: {e}")
            return None
