import torch
import torch.nn as nn

from config import *


class Trainer:

    def __init__(

            self,

            model,

            train_loader,

            val_loader

    ):

        self.device = DEVICE

        self.model = model.to(self.device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=LEARNING_RATE

        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            self.optimizer,

            mode="min",

            factor=0.5,

            patience=5

        )

        CHECKPOINT_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        self.best_loss = float("inf")

        self.early_stop_counter = 0

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for X, y in self.train_loader:

            X = X.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            prediction = self.model(X)

            loss = self.criterion(

                prediction,

                y

            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                max_norm=1.0

            )

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):

        self.model.eval()

        total_loss = 0

        predictions = []

        targets = []

        with torch.no_grad():

            for X, y in self.val_loader:

                X = X.to(self.device)

                y = y.to(self.device)

                prediction = self.model(X)

                loss = self.criterion(

                    prediction,

                    y

                )

                total_loss += loss.item()

                predictions.extend(

                    prediction.cpu().numpy()

                )

                targets.extend(

                    y.cpu().numpy()

                )

        return (

            total_loss / len(self.val_loader),

            predictions,

            targets

        )

    def save_checkpoint(self):

        torch.save(

            {

                "model_state_dict": self.model.state_dict(),

                "optimizer_state_dict": self.optimizer.state_dict(),

                "best_loss": self.best_loss

            },

            CHECKPOINT_DIR / "best_model.pth"

        )

    def train(self):

        history = {

            "train_loss": [],

            "val_loss": []

        }

        for epoch in range(EPOCHS):

            train_loss = self.train_epoch()

            val_loss, _, _ = self.validate()

            history["train_loss"].append(

                train_loss

            )

            history["val_loss"].append(

                val_loss

            )

            self.scheduler.step(val_loss)

            if val_loss < self.best_loss - MIN_DELTA:

                self.best_loss = val_loss

                self.early_stop_counter = 0

                self.save_checkpoint()

            else:

                self.early_stop_counter += 1

            print(

                f"Epoch {epoch+1}/{EPOCHS}"

            )

            print(

                f"Train Loss : {train_loss:.6f}"

            )

            print(

                f"Validation Loss : {val_loss:.6f}"

            )

            print(

                f"Learning Rate : {self.optimizer.param_groups[0]['lr']:.7f}"

            )

            print("-" * 50)

            if self.early_stop_counter >= PATIENCE:

                print("\nEarly stopping triggered.")

                break

        return history