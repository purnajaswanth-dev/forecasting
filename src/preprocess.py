from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import joblib


def preprocess_data(data):

    data = data.copy()

    weather_cols = [
        "ApparentTemperature",
        "AirTemperature",
        "DewPointTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
    ]

    # Fill missing weather values
    imputer = SimpleImputer(strategy="median")
    data[weather_cols] = imputer.fit_transform(data[weather_cols])

    # Fill missing solar generation
    data["SolarGeneration"] = data["SolarGeneration"].fillna(0)

    # Normalize
    scaler = MinMaxScaler()

    scale_cols = weather_cols + ["SolarGeneration"]

    data[scale_cols] = scaler.fit_transform(data[scale_cols])

    return data, scaler