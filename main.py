from torch.utils.data import DataLoader

from config import *

from src.load_data import load_data
from src.clean_data import clean_data
from src.merge_data import merge_data
from src.preprocess import preprocess_data
from src.split_data import split_data
from src.sequence import create_sequences

from src.dataset import SolarDataset
from src.model import SolarForecastModel
from src.train import Trainer
from evaluate import evaluate
from src.visualize import plot_loss, plot_predictions
import joblib


def main():

    print("=" * 50)
    print("Weather-Based Solar Power Forecasting")
    print("=" * 50)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------
    # Load Data
    # -----------------------------
    solar, weather, sites, monthly = load_data()

    # -----------------------------
    # Clean Data
    # -----------------------------
    solar, weather, sites, monthly = clean_data(
        solar,
        weather,
        sites,
        monthly
    )

    # -----------------------------
    # Merge Data
    # -----------------------------
    data = merge_data(
        solar,
        weather,
        sites
    )

    # -----------------------------
    # Preprocess
    # -----------------------------
    data, scaler = preprocess_data(data)

    

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        CHECKPOINT_DIR / "scaler.pkl"
    )

    # -----------------------------
    # Train / Validation / Test Split
    # -----------------------------
    train_df, val_df, test_df = split_data(data)

    # -----------------------------
    # Create Sequences
    # -----------------------------
    X_train, y_train = create_sequences(
        train_df,
        SEQUENCE_LENGTH
    )

    X_val, y_val = create_sequences(
        val_df,
        SEQUENCE_LENGTH
    )

    X_test, y_test = create_sequences(
        test_df,
        SEQUENCE_LENGTH
    )

    # -----------------------------
    # Dataset
    # -----------------------------
    train_dataset = SolarDataset(
        X_train,
        y_train
    )

    val_dataset = SolarDataset(
        X_val,
        y_val
    )

    test_dataset = SolarDataset(
        X_test,
        y_test
    )

    # -----------------------------
    # DataLoaders
    # -----------------------------
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = SolarForecastModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=OUTPUT_SIZE
    )

    print(model)
    print(f"\nUsing Device: {DEVICE}")

    # -----------------------------
    # Train
    # -----------------------------
    # trainer = Trainer(
    #     model,
    #     train_loader,
    #     val_loader
    # )

    # history = trainer.train()

    # -----------------------------
    # Evaluate
    # -----------------------------
    actual, predicted = evaluate(
        model,
        test_dataset
    )

    # -----------------------------
    # Create Plots
    # -----------------------------
    # plot_loss(
    #     history,
    #     PLOTS_DIR / "loss.png"
    # )

    plot_predictions(
        actual,
        predicted,
        PLOTS_DIR / "prediction.png"
    )

    print("\nTraining Complete!")
    print("Best model saved to:", CHECKPOINT_DIR)
    print("Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()