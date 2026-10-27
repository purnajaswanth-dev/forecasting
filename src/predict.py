import torch

from config import *




def predict(model, sequence):

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    with torch.no_grad():

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        sequence = sequence.to(DEVICE)

        prediction = model(sequence)

    return prediction.item()