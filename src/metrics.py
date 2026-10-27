import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_metrics(actual, predicted):

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    # MAE
    mae = mean_absolute_error(actual, predicted)

    # RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Ignore zero and near-zero actual values for MAPE
    mask = actual > 1e-6

    if np.sum(mask) > 0:
        mape = np.mean(
            np.abs((actual[mask] - predicted[mask]) / actual[mask])
        ) * 100
    else:
        mape = float("nan")

    # R² Score
    r2 = r2_score(actual, predicted)

    return mae, rmse, mape, r2