# ☀️ Weather-Based Solar Power Forecasting

A deep learning framework for accurate solar energy generation forecasting leveraging multivariate meteorological data, Long Short-Term Memory (LSTM) networks, and Spatio-Temporal Graph Convolutional Networks (LSTM-GCN).

---

## 📌 Project Overview

Solar power generation depends heavily on meteorological conditions such as temperature, humidity, and wind characteristics. This project implements two complementary forecasting paradigms:

1. **Multivariate Temporal Forecasting (Deep LSTM)**: Models time-series sequences of ambient weather variables (`ApparentTemperature`, `AirTemperature`, `DewPointTemperature`, `RelativeHumidity`, `WindSpeed`, `WindDirection`) to forecast future solar generation.
2. **Spatio-Temporal Graph Neural Network (LSTM-GCN)**: Clusters solar PV sites spatially, encodes temporal dynamics across representative "pilot" sites via LSTM encoders, propagates topological dependencies over a graph with Graph Convolutional Networks (GCN via PyTorch Geometric), and predicts power outputs across all PV units via an MLP head.

---

## 🏛️ Architecture

```
[ Raw Solar & Weather Data ]
             │
             ▼
     [ Preprocessing ] ─── Imputation (Median) + Normalization (MinMaxScaler)
             │
             ├──► [ Temporal LSTM Pipeline ]
             │           │
             │           ▼
             │    [ 2-Layer LSTM ] ──► [ BatchNorm + Dropout ] ──► [ MLP Regressor ]
             │                                                           │
             │                                                           ▼
             │                                               [ Power Forecast (Single Site) ]
             │
             └──► [ Spatio-Temporal Graph Pipeline ]
                         │
                         ▼
                  [ K-Means Spatial Clustering ] ──► Identify Pilot PV Sites
                         │
                         ▼
                  [ LSTM Temporal Encoder ] (Pilot Embeddings)
                         │
                         ▼
                  [ GCN Spatial Propagation ] (Graph Topology)
                         │
                         ▼
                  [ Multi-Unit MLP Regressor ] ──► [ Multi-Site Generation Forecast ]
```

---

## 📁 Repository Structure

```plaintext
forecasting/
├── config.py              # Configuration for dataset paths, preprocessing, & LSTM model
├── config_model.py        # Configuration for LSTM-GCN graph neural network training
├── main.py                # Main end-to-end execution pipeline (load -> preprocess -> train -> evaluate)
├── train.py               # Standalone training script for Spatio-Temporal LSTM-GCN
├── evaluate.py            # Evaluation module computing MAE, RMSE, MAPE, and R²
├── predict.py             # Inference module for generating predictions on new sequences
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules for data, weights, and caches
├── .gitattributes         # Line ending and binary file definitions
├── LICENSE                # MIT License
├── data/
│   ├── raw/               # Raw input CSV datasets (ignored in git)
│   ├── processed/         # Normalized sequences, labels, scaler, and clusters
│   └── sequences/         # Generated batch sequence files
├── notebooks/             # Exploratory data analysis (EDA) notebooks
├── outputs/
│   ├── checkpoints/       # Saved PyTorch model checkpoints & scaler
│   ├── plots/             # Training loss curves and actual vs. predicted plots
│   └── predictions/       # Saved evaluation predictions
├── saved_models/          # Model weights for LSTM-GCN
└── src/
    ├── __init__.py        # Package initialization
    ├── clean_data.py      # Timestamp parsing and deduplication
    ├── clustering.py      # K-Means clustering on solar site geographical coordinates
    ├── dataset.py         # PyTorch Dataset implementation for solar sequences
    ├── load_data.py       # Raw CSV data loading utilities
    ├── merge_data.py      # Merges generation data with weather readings and site metadata
    ├── metrics.py         # MAE, RMSE, MAPE, and R² evaluation metrics
    ├── model.py           # PyTorch Deep LSTM solar forecasting model
    ├── preprocess.py      # Imputation and MinMax feature scaling
    ├── save_data.py       # Serializes processed numpy sequences and cluster metadata
    ├── sequence.py        # Sliding-window temporal sequence generator
    ├── split_data.py      # Chronological Train / Val / Test splitting
    ├── utils.py           # Reproducibility seeds and device helpers
    ├── visualize.py       # Loss curve and prediction visualization plots
    └── models/
        ├── __init__.py
        ├── gcn.py         # 2-layer Graph Convolutional Network encoder
        ├── loss.py        # Combined MSE + MAE composite loss function
        ├── lstm.py        # Pilot PV unit temporal LSTM encoder
        ├── lstm_gcn.py    # Integrated Spatio-Temporal LSTM-GCN architecture
        └── mlp.py         # Multi-Layer Perceptron predictor head
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd forecasting
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch & CUDA:**
> If you have an NVIDIA GPU, install the CUDA-compatible PyTorch wheel matching your system from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 📊 Data Setup

Place your raw datasets inside the `data/raw/` directory:

| Filename | Description |
| :--- | :--- |
| `Solar_Energy_Generation.csv` | Solar output timeseries (`CampusKey`, `SiteKey`, `Timestamp`, `SolarGeneration`) |
| `Weather_Data_reordered_all.csv` | Weather readings (`Timestamp`, `AirTemperature`, `RelativeHumidity`, etc.) |
| `Solar_Site_Details.csv` | Solar installation metadata including site coordinates (`lat`, `Lon`) |
| `Monthly_Summary_Solar.csv` | Aggregated monthly generation data |

---

## 🚀 Usage

### 1. Run the Full Forecasting Pipeline
Executes data loading, cleaning, merging, feature scaling, sequence creation, model evaluation, and plot generation:
```bash
python main.py
```

### 2. Train the Spatio-Temporal (LSTM-GCN) Model
Trains the Graph Convolutional network with pilot PV unit embeddings:
```bash
python train.py
```

### 3. Standalone Model Evaluation
Evaluates the trained model against the test dataset and prints error metrics:
```bash
python evaluate.py
```

---

## 📈 Evaluation Metrics

The framework evaluates predictions against ground truth using:

- **Mean Absolute Error (MAE)**: Measures average absolute prediction error magnitude.
- **Root Mean Squared Error (RMSE)**: Penalizes larger forecasting deviations.
- **Mean Absolute Percentage Error (MAPE)**: Relative percentage error (evaluated on positive values).
- **Coefficient of Determination ($R^2$ Score)**: Proportion of variance explained by the model.

---

## 🛠️ Configuration

Hyperparameters and paths can be customized in:
- `config.py`: Controls sequence length (default: 24), hidden units (128), dropout (0.3), batch size (256), learning rate, and training patience.
- `config_model.py`: Controls graph architecture dimensions, pilot unit selection parameters, optimizer settings, and learning rate schedules.

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
