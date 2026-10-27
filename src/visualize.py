import matplotlib.pyplot as plt


def plot_loss(history, save_path):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training History")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_predictions(
        actual,
        predicted,
        save_path,
        num_points=300
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        actual[:num_points],
        label="Actual"
    )

    plt.plot(
        predicted[:num_points],
        label="Predicted"
    )

    plt.xlabel("Sample")

    plt.ylabel("Solar Generation")

    plt.title("Prediction vs Actual")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()