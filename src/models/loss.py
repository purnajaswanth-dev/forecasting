import torch
import torch.nn as nn


class ForecastLoss(nn.Module):
    """
    Combined MSE + MAE loss used for training.
    """

    def __init__(
        self,
        mse_weight=1.0,
        mae_weight=1.0
    ):
        super().__init__()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(
        self,
        prediction,
        target
    ):
        mse_loss = self.mse(
            prediction,
            target
        )

        mae_loss = self.mae(
            prediction,
            target
        )

        total_loss = (
            self.mse_weight * mse_loss
            +
            self.mae_weight * mae_loss
        )

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "mae": mae_loss
        }