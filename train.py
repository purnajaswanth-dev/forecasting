import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config_model import ModelConfig
from src.models import LSTMGCNModel, ForecastLoss


class Trainer:

    def __init__(
        self,
        graph,
        pilot_sites,
        train_dataset,
        val_dataset
    ):

        self.device = ModelConfig.DEVICE

        pilot_indices = pilot_sites["NodeID"].tolist()

        self.model = LSTMGCNModel(
            num_nodes=graph.num_nodes,
            pilot_indices=pilot_indices,
            hidden_dim=ModelConfig.HIDDEN_SIZE
        ).to(self.device)

        

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=False
        )

        self.criterion = ForecastLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=ModelConfig.LEARNING_RATE,
            weight_decay=ModelConfig.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=ModelConfig.STEP_SIZE,
            gamma=ModelConfig.GAMMA
        )

        os.makedirs("saved_models", exist_ok=True)

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0

        for batch in self.train_loader:

            graph = batch["graph"][0]

            edge_index = graph.edge_index.to(self.device)

            edge_weight = None

            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                edge_weight = graph.edge_attr.float().to(self.device)

            x = batch["sequence"].float().to(self.device)
            y = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()

            prediction = self.model(
                x,
                edge_index,
                edge_weight
            )

            loss_dict = self.criterion(
                prediction,
                y
            )

            loss = loss_dict["loss"]

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0

        for batch in self.val_loader:

            graph = batch["graph"][0]
            
            edge_index = graph.edge_index.to(self.device)

            edge_weight = None

            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                edge_weight = graph.edge_attr.float().to(self.device)

            x = batch["sequence"].float().to(self.device)
            y = batch["label"].float().to(self.device)

            prediction = self.model(
                x,
                edge_index,
                edge_weight
            )

            loss = self.criterion(
                prediction,
                y
            )["loss"]

            running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def train(self):

        best_loss = float("inf")

        for epoch in range(ModelConfig.EPOCHS):

            train_loss = self.train_one_epoch()

            val_loss = self.validate()

            self.scheduler.step()

            print(
                f"Epoch {epoch+1:03d}/{ModelConfig.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Validation Loss: {val_loss:.6f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    ModelConfig.BEST_MODEL_PATH
                )

        torch.save(
            self.model.state_dict(),
            ModelConfig.MODEL_PATH
        )

        print("\nTraining Completed.")
        print("Best Validation Loss :", best_loss)