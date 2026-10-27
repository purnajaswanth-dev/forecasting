import torch


class ModelConfig:

    # ==========================
    # Dataset
    # ==========================

    SEQUENCE_LENGTH = 24

    BATCH_SIZE = 64

    NUM_WORKERS = 0

    SHUFFLE = True


    # ==========================
    # Model
    # ==========================

    INPUT_SIZE = 1

    HIDDEN_SIZE = 64

    NUM_LAYERS = 2

    DROPOUT = 0.2

    MLP_HIDDEN = 32


    # ==========================
    # Training
    # ==========================

    EPOCHS = 50

    LEARNING_RATE = 1e-3

    WEIGHT_DECAY = 1e-5


    # ==========================
    # Optimizer
    # ==========================

    OPTIMIZER = "Adam"


    # ==========================
    # Scheduler
    # ==========================

    USE_SCHEDULER = True

    STEP_SIZE = 10

    GAMMA = 0.5


    # ==========================
    # Device
    # ==========================

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ==========================
    # Saving
    # ==========================

    MODEL_PATH = "saved_models/lstm_gcn.pth"

    BEST_MODEL_PATH = "saved_models/best_model.pth"


    # ==========================
    # Random Seed
    # ==========================

    RANDOM_SEED = 42