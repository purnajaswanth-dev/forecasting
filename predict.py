import torch

from config_model import ModelConfig
from src.models import LSTMGCNModel


class Predictor:

    def __init__(
        self,
        graph,
        pilot_sites
    ):

        self.device = ModelConfig.DEVICE

        pilot_indices = pilot_sites["NodeID"].tolist()

        self.model = LSTMGCNModel(
            num_nodes=graph.num_nodes,
            pilot_indices=pilot_indices,
            hidden_dim=ModelConfig.HIDDEN_SIZE
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                ModelConfig.BEST_MODEL_PATH,
                map_location=self.device
            )
        )

        self.model.eval()

        self.edge_index = graph.edge_index.to(self.device)

        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            self.edge_weight = graph.edge_attr.float().to(self.device)
        else:
            self.edge_weight = None

    @torch.no_grad()
    def predict(self, x):

        x = x.float().to(self.device)

        prediction = self.model(
            x,
            self.edge_index,
            self.edge_weight
        )

        return prediction.cpu()