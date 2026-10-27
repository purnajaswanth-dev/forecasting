import numpy as np


def create_sequences(data, sequence_length=12):

    features = [
        "SolarGeneration",
        "ApparentTemperature",
        "AirTemperature",
        "DewPointTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
    ]

    X = []
    y = []

    # Create sequences separately for each solar site
    grouped = data.groupby(["CampusKey", "SiteKey"])

    for _, group in grouped:

        group = group.sort_values("Timestamp")

        values = group[features].values
        targets = group["SolarGeneration"].values

        if len(group) <= sequence_length:
            continue

        for i in range(len(group) - sequence_length):

            X.append(
                values[i:i + sequence_length]
            )

            y.append(
                targets[i + sequence_length]
            )

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)