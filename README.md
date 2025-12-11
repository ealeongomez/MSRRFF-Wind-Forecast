# MSRRFF-Wind-Forecast

**Multi-Scale Random Fourier Features for Wind Forecasting**

A research project combining Random Fourier Features (RFF) with Recurrent Neural Networks (RNN) for sustainable wind forecasting in La Guajira, Colombia.

---

## 📋 Project Overview

This project implements hybrid deep learning architectures that combine:
- **Random Fourier Features (RFF)**: Spectral approximations of kernels based on Bochner's theorem
- **Recurrent Neural Networks**: LSTM, GRU, and vanilla RNN variants
- **Bayesian Optimization**: Hyperparameter tuning with Optuna

The goal is to develop accurate and efficient wind forecasting models for sustainable energy applications.

---

## 🗂️ Project Structure

```
MSRRFF-Wind-Forecast/
│
├── data/                    # Wind data
│   ├── raw/                # Original datasets
│   └── processed/          # Preprocessed data
│
├── src/                    # Source code
│   ├── data.py            # Data loading and preprocessing
│   ├── layers.py          # RFF layer implementations
│   ├── models.py          # Model architectures
│   ├── training.py        # Training pipeline
│   └── utils.py           # Metrics, plotting, utilities
│
├── notebooks/              # Jupyter notebooks
│   └── experiments.ipynb  # Experimentation notebook
│
├── results/                # Experimental results
│   ├── models/            # Saved model checkpoints
│   ├── plots/             # Generated visualizations
│   └── metrics/           # Performance metrics
│
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MSRRFF-Wind-Forecast.git
cd MSRRFF-Wind-Forecast
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.data import load_wind_data
from src.models import RFF_GRU
from src.training import train_model
from src.utils import evaluate_model, plot_results

# Load data
X_train, y_train, X_test, y_test = load_wind_data('data/processed/wind_data.csv')

# Create model
model = RFF_GRU(
    input_size=10,
    hidden_size=64,
    rff_dim=100,
    kernel='gaussian',
    horizon=24
)

# Train
history = train_model(model, X_train, y_train, epochs=100)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")

# Visualize
plot_results(history, save_path='results/plots/training_curve.png')
```

---

## 🧪 Research Components

### Random Fourier Features (RFF)

Implementation of spectral feature maps based on **Bochner's theorem**:

- **Gaussian RFF**: Approximates Gaussian (RBF) kernel
- **Laplacian RFF**: Approximates Laplacian kernel  
- **Multiscale RFF**: Combines multiple kernel bandwidths

### Hybrid Architectures

Compositional models following the pattern:

```
Input → RFF Mapping → RNN/LSTM/GRU → MLP → Output
```

### Hyperparameter Optimization

Bayesian optimization using Optuna:
- Automated search space exploration
- Early pruning with HyperbandPruner
- Multi-objective optimization (RMSE, MAE, R², KMSE)

---

## 📊 Experiments

Run experiments from notebooks or scripts:

```bash
# Interactive experimentation
jupyter notebook notebooks/experiments.ipynb

# Batch training (when implemented)
python src/training.py --config config.yaml
```

---

## 📈 Results

Results are automatically saved to the `results/` directory:
- **Models**: `results/models/*.pth`
- **Plots**: `results/plots/*.png`
- **Metrics**: `results/metrics/*.csv`

---

## 📚 Theoretical Background

This project is based on:

1. **Bochner's Theorem**: Characterization of shift-invariant kernels via Fourier analysis
2. **Random Fourier Features** (Rahimi & Recht, 2007): Explicit feature maps for kernel approximation
3. **Recurrent Neural Networks**: Sequence modeling for time series forecasting

---

## 🔬 Research Context

This project is part of a **Doctoral Research** program focused on:
- Sustainable wind energy forecasting
- Spectral methods in deep learning
- Hybrid kernel-based neural architectures

**Location**: La Guajira, Colombia  
**Application**: Renewable energy prediction for grid stability

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{msrrff2025,
  title={Multi-Scale Random Fourier Features for Wind Forecasting},
  author={Your Name},
  year={2025},
  institution={Your University}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

---

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@university.edu]
- **Institution**: [Your University]

---

**Last Updated**: November 2025

