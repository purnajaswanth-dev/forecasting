import torch
from torch.utils.data import DataLoader

from config import *
from src.metrics import calculate_metrics


def evaluate(model, test_dataset):

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    checkpoint = CHECKPOINT_DIR / "best_model.pth"

    checkpoint_data = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint_data["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(DEVICE)

            prediction = model(X)

            predictions.extend(
                prediction.cpu().numpy()
            )

            targets.extend(
                y.numpy()
            )

    mae, rmse, mape, r2 = calculate_metrics(
        targets,
        predictions
    )

    print()
    print("=" * 40)
    print("Test Results")
    print("=" * 40)
    print(f"MAE      : {mae:.5f}")
    print(f"RMSE     : {rmse:.5f}")
    print(f"MAPE     : {mape:.2f}%")
    print(f"R² Score : {r2:.5f}")
    print("=" * 40)

    return targets, predictions