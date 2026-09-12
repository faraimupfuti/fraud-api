"""
Shared feature engineering for the Zimswitch-style fraud detection project.
Used by both train_model.py (batch, historical data) and streamlit_app.py
(single what-if transactions typed in by a user).
"""

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "amount", "hour", "is_odd_hour", "day_of_week",
    "seconds_since_prev", "km_from_prev_txn", "implied_speed_kmh",
    "amount_zscore", "txn_count_1h", "txn_count_24h",
]
CATEGORICAL_FEATURES = ["channel", "mcc", "currency"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CITIES = {
    "Harare": (-17.8292, 31.0522), "Bulawayo": (-20.1500, 28.5833),
    "Mutare": (-18.9707, 32.6709), "Gweru": (-19.4500, 29.8167),
    "Kwekwe": (-18.9281, 29.8149), "Masvingo": (-20.0637, 30.8277),
    "Chinhoyi": (-17.3667, 30.2000), "Victoria Falls": (-17.9243, 25.8567),
}
MCC_LABELS = {
    "5411": "Groceries", "5541": "Fuel", "5812": "Restaurant", "5912": "Pharmacy",
    "6011": "ATM / cash withdrawal", "5651": "Clothing", "4900": "Utilities",
    "5999": "Retail — misc", "6051": "Money transfer", "7011": "Hotel",
    "5311": "Department store", "4814": "Telecom / airtime",
}


def build_checklist(amount, hour, day_name, minutes_since_prev, km_apart,
                     txn_count_1h, txn_count_24h, typical_amount):
    """Build the human-readable checklist shown during analysis, computed
    live from whatever transaction values are supplied. Shared by the
    FastAPI service and any client so the explanation always matches the
    number the model actually saw."""
    hours_gap = max(minutes_since_prev / 60, 1 / 60)
    implied_speed = min(km_apart / hours_gap, 2000)
    zscore = (amount - typical_amount) / max(typical_amount * 0.4, 1)
    is_odd_hour = 1 <= hour <= 4

    return [
        {
            "label": "Travel speed between transactions",
            "value": f"{km_apart:,.0f} km apart, {minutes_since_prev} min apart -> {implied_speed:,.0f} km/h implied",
            "flag": implied_speed > 150,
            "detail": ("No real form of transport reaches this speed - this card can't genuinely be "
                       "in both places.") if implied_speed > 150 else
                      "Consistent with normal travel time, or no travel at all.",
        },
        {
            "label": "Recent transaction frequency",
            "value": f"{txn_count_1h} transaction(s) in the last hour, {txn_count_24h} in the last 24 hours",
            "flag": txn_count_1h >= 4,
            "detail": ("Rapid repeated use in a short window - a common pattern when someone is "
                       "testing whether a stolen card still works.") if txn_count_1h >= 4 else
                      "Normal transaction frequency for this card.",
        },
        {
            "label": "Amount vs. this card's typical spend",
            "value": f"${amount:,.2f} vs. a usual ${typical_amount:,.2f} ({zscore:+.1f} std devs)",
            "flag": abs(zscore) > 3,
            "detail": "Far outside how this card is normally used - worth a second look."
                      if abs(zscore) > 3 else "Within this card's normal spending range.",
        },
        {
            "label": "Time of transaction",
            "value": f"{hour:02d}:00 on {day_name}",
            "flag": is_odd_hour and amount > typical_amount * 1.5,
            "detail": ("A larger-than-usual transaction during overnight hours (1am-4am) is an "
                       "uncommon pattern.") if (is_odd_hour and amount > typical_amount * 1.5) else
                      "Normal timing for a transaction.",
        },
    ]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(df):
    """Batch feature engineering over historical transaction history,
    computed causally per card (only using each card's prior transactions)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["is_odd_hour"] = df["hour"].between(1, 4).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    grp = df.groupby("card_id")

    df["prev_timestamp"] = grp["timestamp"].shift(1)
    df["seconds_since_prev"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    df["seconds_since_prev"] = df["seconds_since_prev"].fillna(999999)

    df["prev_lat"] = grp["latitude"].shift(1)
    df["prev_lon"] = grp["longitude"].shift(1)
    df["km_from_prev_txn"] = haversine_km(
        df["latitude"], df["longitude"], df["prev_lat"], df["prev_lon"]
    ).fillna(0)
    hours_gap = (df["seconds_since_prev"] / 3600).replace(0, 1 / 3600)
    df["implied_speed_kmh"] = (df["km_from_prev_txn"] / hours_gap).clip(upper=2000)

    shifted = grp["amount"].shift(1)
    df["_shifted_amount"] = shifted
    df["_has_prior"] = df["_shifted_amount"].notna().astype(int)
    df["_shifted_filled"] = df["_shifted_amount"].fillna(0)
    df["_shifted_filled_sq"] = df["_shifted_filled"] ** 2

    g2 = df.groupby("card_id")
    cum_count = g2["_has_prior"].cumsum()
    cum_sum = g2["_shifted_filled"].cumsum()
    cum_sumsq = g2["_shifted_filled_sq"].cumsum()

    df["running_avg_amount"] = (cum_sum / cum_count.replace(0, np.nan))
    running_var = (cum_sumsq / cum_count.replace(0, np.nan)) - df["running_avg_amount"] ** 2
    df["running_std_amount"] = np.sqrt(running_var.clip(lower=0))
    df["running_avg_amount"] = df["running_avg_amount"].fillna(df["amount"].median())
    df["running_std_amount"] = df["running_std_amount"].fillna(0).replace(0, 1)
    df["amount_zscore"] = (df["amount"] - df["running_avg_amount"]) / df["running_std_amount"]
    df = df.drop(columns=["_has_prior", "_shifted_filled", "_shifted_filled_sq"])

    def rolling_count(sub, window):
        sub = sub.set_index("timestamp")
        return sub["amount"].rolling(window, closed="left").count().values

    df["txn_count_1h"] = 0
    df["txn_count_24h"] = 0
    for card_id, sub in df.groupby("card_id"):
        idx = sub.index
        df.loc[idx, "txn_count_1h"] = rolling_count(sub, "1h")
        df.loc[idx, "txn_count_24h"] = rolling_count(sub, "24h")
    df["txn_count_1h"] = df["txn_count_1h"].fillna(0)
    df["txn_count_24h"] = df["txn_count_24h"].fillna(0)

    return df


def build_single_row(amount, hour, day_of_week, minutes_since_prev, km_from_prev,
                      txn_count_1h, txn_count_24h, channel, mcc, currency,
                      avg_amount, std_amount):
    """Build a single-row feature DataFrame for a what-if transaction typed
    in by a user (used by the Streamlit app), matching the same derived
    features the batch pipeline computes."""
    seconds_since_prev = max(minutes_since_prev, 1 / 60) * 60
    is_odd_hour = 1 if 1 <= hour <= 4 else 0
    hours_gap = max(seconds_since_prev / 3600, 1 / 3600)
    implied_speed_kmh = min(km_from_prev / hours_gap, 2000)
    amount_zscore = (amount - avg_amount) / max(std_amount, 1)

    row = {
        "amount": amount, "hour": hour, "is_odd_hour": is_odd_hour,
        "day_of_week": day_of_week, "seconds_since_prev": seconds_since_prev,
        "km_from_prev_txn": km_from_prev, "implied_speed_kmh": implied_speed_kmh,
        "amount_zscore": amount_zscore, "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h, "channel": channel, "mcc": mcc,
        "currency": currency,
    }
    return pd.DataFrame([row])[ALL_FEATURES]
