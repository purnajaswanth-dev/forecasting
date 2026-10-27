import joblib
from config import PROCESSED_DATA


def save_processed_data(
    train,
    val,
    test,
    clusters,
    scaler
):

    train[0].dump(
        PROCESSED_DATA / "train_sequences.npy"
    )

    train[1].dump(
        PROCESSED_DATA / "train_labels.npy"
    )

    val[0].dump(
        PROCESSED_DATA / "val_sequences.npy"
    )

    val[1].dump(
        PROCESSED_DATA / "val_labels.npy"
    )

    test[0].dump(
        PROCESSED_DATA / "test_sequences.npy"
    )

    test[1].dump(
        PROCESSED_DATA / "test_labels.npy"
    )

    clusters.to_csv(
        PROCESSED_DATA / "clusters.csv",
        index=False
    )

    joblib.dump(
        scaler,
        PROCESSED_DATA / "scaler.pkl"
    )