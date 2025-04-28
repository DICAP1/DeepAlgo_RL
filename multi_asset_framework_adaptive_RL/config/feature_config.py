# Central list of features used in PPO training and inference
FEATURE_COLUMNS = [
    "Close",
    "EMA_50",
    "EMA_200",
    "RSI",
    "RSI_6",           # NEW
    "ATR",
    "MACD_Histogram",
    "MACD_Line",
    "MACD_Signal",
    "ADX_20",          # NEW
    "PLUS_DI_20",      # NEW
    "MINUS_DI_20",     # NEW
    "BB_Upper",
    "BB_Middle",
    "BB_Lower"
]

def get_feature_list(expected_size=None):
    """
    Returns the list of features used for the PPO model.
    Pads the list with dummy feature names if expected_size is greater than the base length.
    """
    features = FEATURE_COLUMNS.copy()
    if expected_size and len(features) < expected_size:
        for i in range(len(features), expected_size):
            features.append(f"Feature_{i+1}")
    return features

