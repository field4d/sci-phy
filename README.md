# sci-phy
# Plant Health Analysis and Stress Detection

This repository is dedicated to **analyzing and understanding plant health** through advanced data science techniques. By leveraging time series data, we aim to classify plant stress conditions, predict physiological trends, and detect anomalies that may indicate underlying issues.

## Dataset

The dataset includes comprehensive measurements of plant physiological parameters collected from 500 plants over several days. These parameters help monitor plant health under various environmental and experimental conditions.

- **Key Columns:**
  - `timestamp`: Date and time of the measurement.
  - `plant_ID`: Unique identifier for each plant.
  - `condition`: Indicates normal (`W`) or stressed conditions.
  - Additional features: `temp` (temperature), `weight`, `RH` (Relative Humidity), `vpd` (Vapor Pressure Deficit), and more.

## Data Acquisition and Preprocessing

The first steps involve acquiring raw data, organizing it, and preprocessing to ensure data quality.

1. **Data Acquisition:**
   - Fetching raw data from the SPAC API.
   - Secure API authorization using environment variables.

2. **Data Organization:**
   - Structuring metadata and experimental parameters.
   - Handling multi-plant experiments and splitting plant IDs when necessary.

3. **Data Preprocessing:**
   - Cleaning and standardizing metadata.
   - Adjusting dates for consistency.
   - Validating data integrity (handling missing values, anomalies, overlaps).
   - Converting data into Parquet format for efficient storage and processing.

## Methods

### Stress Classification

Classifying transpiration time series into different stress categories using both traditional machine learning and deep learning approaches:

1. **Traditional Machine Learning Workflow:**
   - **Feature Extraction:** Deriving key features from raw data.
   - **Models:** Random Forest, XGBoost for robust classification.

2. **Deep Learning Workflow:**
   - **Change Point Detection:** Identifying shifts in plant physiological responses.
   - **Models:** LSTM (Long Short-Term Memory), TCN (Temporal Convolutional Networks), CNN (Convolutional Neural Networks), and Sliding Window + RNN approaches for sequence modeling.

### Time Series Forecasting

Forecasting future plant physiological responses to anticipate potential stress conditions:

- **Nixtla Libraries:** `statsforecast`, `neuralforecast`, `TimeGPT` for statistical and neural forecasting.
- **Darts Library:** Unified interface supporting ARIMA, deep learning models, and hybrid approaches.

### Anomaly Detection

Detecting anomalies that indicate irregularities in plant behavior, which could signify stress or data collection issues:

- **Machine Learning:** Isolation Forest for unsupervised anomaly detection.
- **Deep Learning:** Autoencoders to detect deviations from learned normal patterns.

## Visualization

The repository includes visualization tools to support data exploration and insights:

- **Time Series Visualization:** Tracking trends in plant physiological data.
- **Stress Pattern Detection:** Highlighting periods of stress and identifying change points.
- **Anomaly Visualization:** Pinpointing outliers and abnormal patterns in plant growth and transpiration.



## Repository Structure
```
lab-data-project/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── form_data.csv      # Metadata info to get data using API
│   ├── raw/               # Original, immutable data dumps
│   └── processed/         # Cleaned and preprocessed data
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── get_data_form.py        # Code to collect metadata info from user
│   │   ├── get_data.py             # Code to download, load, and manage data
│   │   ├── data_preprocessing.py   # Data cleaning, transformation, feature engineering
│   │   └── data_validation.py      # Error testing & data validation scripts
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── exploratory_analysis.py # Code for EDA, visualization, statistical summaries
│   ├── models/
│   │   ├── __init__.py
│   │   ├── forecasting/
│   │   │   ├── __init__.py
│   │   │   ├── train_forecasting.py   # Training scripts for forecasting models
│   │   │   ├── evaluate_forecasting.py # Evaluation for forecasting models
│   │   │   └── forecasting_utils.py   # Helper functions for forecasting
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── train_classification.py # Training scripts for classification models
│   │   │   ├── evaluate_classification.py # Evaluation for classification models
│   │   │   └── classification_utils.py   # Helper functions for classification
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Centralized logging system
│       └── error_handler.py        # Error handling mechanisms
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_data_preprocessing.py
│   └── test_models.py
└── docs/
    └── setup_guide.md              # Documentation for setup, contributing, etc.
```


## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Transpiration-Stress-Classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API credentials using `.env` file:
   ```env
   AUTHORIZATION=your_api_token_here
   ```
4. Run the data pipeline:
   ```bash
   python src/data/get_data.py
   ```
