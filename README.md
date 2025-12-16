# VIP Project - Electricity Price Trading Dashboard

A comprehensive, modular quantitative trading system for electricity markets featuring machine learning models (LSTM/GRU/Random Forest), rule-based strategies, comprehensive backtesting, and an interactive Streamlit dashboard with professional visualizations.

## 🚀 Features

### Machine Learning Models

* **LSTM (Long Short-Term Memory)**: Deep learning model for time series forecasting with sequence learning capabilities
* **GRU (Gated Recurrent Unit)**: Efficient RNN architecture with comparable performance to LSTM, faster training
* **Random Forest**: Traditional ML model for tabular data with feature importance analysis
* **Optuna Hyperparameter Optimization**: Automatic hyperparameter tuning for GRU models to find optimal configurations

### Trading Strategies

* **Percentile Channel Breakout**: Buy/sell signals based on price percentiles within rolling windows
* **Break of Structure (BOS)**: Trend-following strategy identifying structural breaks in price movements

### Professional Visualizations

#### Classification Visualizations
* **Prediction Probability Distribution by Classification Result**: Histogram showing TP/FP/TN/FN distributions
* **Calibration Curve (Reliability Diagram)**: Model calibration assessment
* **Predictions Timeline**: Actual labels, predicted probabilities, and binary predictions over time
* **ROC Curve**: With baseline comparisons (Random, No Skill)
* **Precision-Recall Curve**: With baseline reference line
* **Threshold Sweep Analysis**: Precision, Recall, F1 across different thresholds
* **Confusion Matrix**: Interactive heatmap visualization
* **Metric Comparison Tables**: Model vs baselines (Majority Class, Random)

#### Regression Visualizations
* **Scatter Plots**: Predicted vs Actual with perfect prediction line
* **Time Series Plots**: Actual vs Predicted over time
* **Residuals vs Time**: Temporal pattern detection
* **Rolling RMSE**: Error trends over time windows
* **Q-Q Plot**: Normality check for residuals
* **Error by Magnitude Analysis**: Mean absolute error by actual value magnitude
* **Residual Distribution**: Histogram with normality indicators
* **Rolling Mean of Residuals**: Trend detection in errors
* **Metric Comparison Tables**: Model vs baselines (Zero, Mean)

### Evaluation Metrics

#### Prediction Metrics
* **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
* **Regression**: MAE, RMSE, MSE, MAPE, R²

#### Trading Metrics
* **ROI**: Return on Investment (cumulative)
* **Win Rate**: Percentage of profitable trades
* **Sharpe Ratio**: Risk-adjusted return measure
* **Total Trades**: Number of trading signals generated

#### Baseline Comparisons
* **Classification**: Compare against Majority Class and Random baselines
* **Regression**: Compare against Zero prediction and Mean baselines

## 📋 Requirements

* Python 3.8+
* See `requirements.txt` for full dependency list

### Key Dependencies
* **Streamlit** >= 1.28.0: Interactive web dashboard
* **PyTorch** >= 2.0.0: Deep learning framework (for LSTM/GRU)
* **scikit-learn** >= 1.3.0: Machine learning utilities
* **Plotly** >= 5.17.0: Interactive visualizations
* **Optuna** >= 3.0.0: Hyperparameter optimization
* **pandas** >= 1.5.0: Data manipulation
* **numpy** >= 1.24.0: Numerical computing
* **scipy** >= 1.10.0: Statistical analysis

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/igopalakrishna/VIP-project.git
cd VIP-project
```

### 2. Create Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset

Ensure the dataset file `Data_cleaned_Dataset.csv` is in the `datasets/` directory. The dataset should contain:
* Date column (Trade Date)
* Price data
* Volume data
* Additional features (natural gas prices, load data, temperature, etc.)

## 🎯 Quick Start

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### Using the Dashboard

#### Navigation
1. **Dashboard**: Overview and data statistics
2. **Percentile Strategy**: Rule-based trading strategy with configurable parameters
3. **Break of Structure**: Trend-following strategy
4. **ML Models**: Train and evaluate machine learning models
5. **Documentation**: Project documentation and usage guide

#### Training ML Models

1. Navigate to **ML Models** section
2. **Select Task Type**:
   - **Direction Prediction (Classification)**: Predict up/down movement
   - **Price Prediction (Regression)**: Predict next-period price/return
3. **Choose Model**:
   - **LSTM**: Deep learning with long-term memory
   - **GRU**: Efficient RNN with Optuna optimization
   - **Random Forest**: Traditional ML for tabular data
4. **Configure Hyperparameters**:
   - Sequence Length (for RNN models)
   - Train/Val/Test split ratios
   - Enable Optuna optimization (GRU only)
5. **Train and Evaluate**: Click "Train and Evaluate Model"
6. **View Results**:
   - Prediction metrics
   - Trading metrics
   - Training history plots
   - Comprehensive visualizations

#### Running Trading Strategies

1. Navigate to **Percentile Strategy** or **Break of Structure**
2. Configure strategy parameters (window size, percentiles, etc.)
3. Select backtest period (start/end dates)
4. Click **Run Strategy**
5. View results:
   - Portfolio balance chart
   - Trading metrics
   - Trade signals visualization

## 📁 Project Structure

```
VIP-project/
├── app.py                          # Streamlit dashboard (main UI)
├── config.py                       # Global configuration settings
├── data_pipeline.py                # Data loading and preprocessing
├── metrics.py                      # Evaluation metrics (prediction & trading)
├── strategies.py                   # Trading strategies implementation
├── training_utils.py               # Training utilities (standardized training loop)
├── logging_config.py              # Logging configuration
├── requirements.txt                # Python dependencies
├── run_all_models.py              # Batch training script
│
├── datasets/
│   └── Data_cleaned_Dataset.csv   # Main dataset (2001-2022)
│
├── models/
│   ├── __init__.py                 # Model registry
│   ├── model_lstm.py               # LSTM model implementation
│   ├── model_gru.py                # GRU model with Optuna optimization
│   ├── model_rf.py                 # Random Forest model
│   ├── model_template_deep_learning.py    # Template for DL models
│   └── model_template_traditional_ml.py   # Template for traditional ML
│
├── notebooks/
│   ├── gru_forecasting_visualizations_pytorch_colab.ipynb
│   ├── gru_forecasting_visualizations_pytorch.ipynb
│   └── gru_forecasting_visualizations.ipynb
│
└── saved_models/                   # Trained model checkpoints (auto-generated)
```

## 🔧 Configuration

Edit `config.py` to customize:

### Task Configuration
```python
TASK_TYPE = "classification"  # or "regression"
```

### Data Configuration
```python
DATA_PATH = "datasets/Data_cleaned_Dataset.csv"
SEQUENCE_LENGTH = 14  # For RNN models
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SCALER_TYPE = "standard"  # or "minmax"
```

### Model Configuration
```python
GRU_CONFIG = {
    "layer1_units": 64,
    "layer2_units": 32,
    "dropout_rate": 0.3,
    "dense_units": 32,
}

LSTM_CONFIG = { ... }
RF_CONFIG = { ... }
```

## 📊 Dataset Information

### Data Source
* **File**: `Data_cleaned_Dataset.csv`
* **Time Period**: 2001-01-02 to 2022-12-31
* **Records**: ~8,034 daily records

### Features
* Price data (electricity prices)
* Volume data
* Natural gas prices
* Load data
* Temperature data
* Engineered features (moving averages, technical indicators)

### Preprocessing
* Automatic handling of missing values
* Zero price detection and handling
* Feature scaling (StandardScaler or MinMaxScaler)
* Time-based train/val/test splitting (no data leakage)

## 🎓 Key Features Explained

### Optuna Hyperparameter Optimization

For GRU models, Optuna automatically searches for optimal hyperparameters:

**Search Space**:
* **Layer Units**: 
  - Layer 1: 32-256 units
  - Layer 2: 16-128 units
* **Dropout Rate**: 0.1 to 0.6
* **Learning Rate**: Log-uniform from 1e-5 to 1e-2
* **Batch Size**: [16, 32, 64, 128]
* **Dense Units**: 16-64 units

**Optimization Process**:
1. Creates Optuna study with specified number of trials
2. Tests different hyperparameter combinations
3. Evaluates on validation set
4. Returns best configuration
5. Trains final model with best hyperparameters

**Usage**:
- Enable "Hyperparameter Optimization (Optuna)" checkbox
- Set number of trials (default: 20, more = better but slower)
- Set timeout in minutes (0 = no timeout)
- Results displayed after optimization completes

### Professional Visualizations

All visualizations are:
* **Interactive**: Built with Plotly for zooming, panning, hovering
* **Publication-ready**: Suitable for academic papers and professional reports
* **Comprehensive**: Cover all aspects of model performance
* **Baseline Comparisons**: Always compare against simple baselines

### Metric Calculation

#### Classification Metrics
* **Accuracy**: Overall correctness
* **Precision**: True positives / (True positives + False positives)
* **Recall**: True positives / (True positives + False negatives)
* **F1-Score**: Harmonic mean of precision and recall
* **ROC-AUC**: Area under ROC curve
* **PR-AUC**: Area under Precision-Recall curve

#### Regression Metrics
* **MAE**: Mean Absolute Error
* **RMSE**: Root Mean Squared Error
* **MSE**: Mean Squared Error
* **MAPE**: Mean Absolute Percentage Error
* **R²**: Coefficient of determination

#### Trading Metrics
* **ROI**: Cumulative return on investment
* **Win Rate**: Percentage of profitable trades
* **Sharpe Ratio**: (Mean return - Risk-free rate) / Standard deviation of returns
* **Total Trades**: Number of trading signals

## 🐛 Troubleshooting

### Common Issues

#### "Optuna is not installed"
```bash
pip install "optuna>=3.0.0,<4.0.0"
```

#### "PyTorch Not Installed"
```bash
pip install torch
```

#### "FileNotFoundError: Could not find Data_cleaned_Dataset.csv"
- Ensure the dataset file is in `datasets/` directory
- Check the path in `config.py`

#### "BrokenPipeError"
- This is handled automatically with `safe_print()` function
- No action needed

#### Metrics showing 0.0000
- Check prediction diagnostics in the UI
- Verify model is making predictions (not all zeros/ones)
- Check class distribution in training data

### Performance Tips

1. **For Faster Training**:
   - Reduce sequence length
   - Use smaller batch sizes
   - Disable Optuna optimization
   - Use Random Forest for quick results

2. **For Better Accuracy**:
   - Increase sequence length (14-30)
   - Enable Optuna optimization
   - Use more training epochs
   - Increase model complexity

3. **For Large Datasets**:
   - Use GPU if available (PyTorch will auto-detect)
   - Increase batch size
   - Use data parallelism

## 📈 Usage Examples

### Example 1: Training GRU with Optuna

```python
# In Streamlit UI:
1. Navigate to "ML Models"
2. Select "GRU" model
3. Enable "Hyperparameter Optimization (Optuna)"
4. Set trials: 30
5. Set timeout: 60 minutes
6. Click "Train and Evaluate Model"
```

### Example 2: Running Percentile Strategy

```python
# In Streamlit UI:
1. Navigate to "Percentile Strategy"
2. Set window size: 14 days
3. Lower percentile: 20 (buy signal)
4. Upper percentile: 80 (sell signal)
5. Select date range: 2020-01-01 to 2022-12-31
6. Click "Run Strategy"
```

### Example 3: Batch Training All Models

```bash
python run_all_models.py
```

This will train all models sequentially and save results.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit your changes**:
   ```bash
   git commit -m 'Add some feature'
   ```
6. **Push to the branch**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

### Development Guidelines

* Follow PEP 8 style guide
* Add docstrings to all functions
* Update README.md for new features
* Test with both classification and regression tasks
* Ensure visualizations work correctly

## 📝 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

* **Streamlit**: Interactive web dashboard framework
* **PyTorch**: Deep learning framework
* **Optuna**: Hyperparameter optimization library
* **Plotly**: Interactive visualization library
* **scikit-learn**: Machine learning utilities

## 📧 Contact & Support

* **GitHub Issues**: [Open an issue](https://github.com/igopalakrishna/VIP-project/issues)
* **Repository**: [https://github.com/igopalakrishna/VIP-project](https://github.com/igopalakrishna/VIP-project)

## 🔄 Recent Updates

### Version 1.0 (Current)

* ✅ Added Optuna hyperparameter optimization for GRU models
* ✅ Added comprehensive professional visualizations (20+ charts)
* ✅ Fixed classification metrics calculation bug (Precision/Recall/F1)
* ✅ Added metric comparison tables with baselines
* ✅ Improved error handling and diagnostics
* ✅ Added Q-Q plots and residual analysis for regression
* ✅ Enhanced UI with better organization and tabs
* ✅ Added calibration curves and threshold sweep analysis
* ✅ Fixed BrokenPipeError with safe_print() function
* ✅ Added prediction diagnostics panel
* ✅ Improved baseline metric calculations (macro averaging)

### Previous Versions

* Initial release with LSTM, GRU, Random Forest models
* Basic trading strategies (Percentile, Break of Structure)
* Streamlit dashboard with basic visualizations

## 📚 Documentation

### Model Architecture

#### GRU Model
```
Input (sequence_length, n_features)
  ↓
GRU Layer 1 (64 units) + Dropout (0.3)
  ↓
GRU Layer 2 (32 units) + Dropout (0.3)
  ↓
Dense Hidden Layer (32 units, ReLU)
  ↓
Output Layer (1 unit)
```

#### LSTM Model
Similar architecture to GRU but with LSTM cells instead.

#### Random Forest
* Default: 100 estimators
* Max depth: 10
* Min samples split: 5

### Data Pipeline

1. **Load Data**: Read CSV file
2. **Preprocess**: Handle missing values, zero prices
3. **Feature Engineering**: Create technical indicators
4. **Scale Features**: StandardScaler or MinMaxScaler
5. **Create Sequences**: For RNN models (sliding window)
6. **Split Data**: Train/Val/Test (time-based, no shuffling)

### Training Process

1. **Initialize Model**: Create model with specified architecture
2. **Compile**: Set optimizer, loss function, metrics
3. **Train**: Standardized training loop with early stopping
4. **Validate**: Monitor validation metrics
5. **Predict**: Generate predictions on test set
6. **Evaluate**: Calculate prediction and trading metrics

## 🎯 Future Enhancements

* [ ] Add more ML models (XGBoost, Transformer models)
* [ ] Add ensemble methods
* [ ] Real-time data integration
* [ ] Portfolio optimization strategies
* [ ] Risk management features
* [ ] Export results to CSV/PDF
* [ ] Model comparison dashboard
* [ ] Feature importance visualization
* [ ] SHAP values for model interpretability

---

**Note**: Make sure you have the dataset file (`Data_cleaned_Dataset.csv`) in the `datasets/` directory before running the application.

**Happy Trading! 📈**

