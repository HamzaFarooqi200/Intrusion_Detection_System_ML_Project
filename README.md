# Network Intrusion Detection using Machine Learning

## Overview

This project implements a Machine Learning-based Network Intrusion
Detection System (NIDS) using the KDD Cup 1999 dataset. The goal is to
classify network connections as normal or malicious by training machine
learning models on network traffic features.

This project demonstrates a complete machine learning workflow,
including: - Data preprocessing - Feature engineering - Model training -
Model evaluation - Testing pipeline

Such systems help improve network security by detecting potential cyber
attacks automatically.

------------------------------------------------------------------------

## Dataset

The project uses the KDD Cup 1999 dataset, a widely used benchmark
dataset for intrusion detection research.

The dataset contains network connection records labeled as:

-   Normal traffic
-   Attack traffic

Attack categories include multiple types of network intrusions such as:

-   DoS (Denial of Service)
-   Probe attacks
-   Remote to Local (R2L)
-   User to Root (U2R)

Dataset files should be placed in the `data` directory.

-------------------------------------------------------------------
## Project Structure

```
project-root/
│
├── data/                     # Dataset files (not included in repository)
│
├── src/                      # Source code
│   ├── preprocess.py         # Data preprocessing and feature preparation
│   ├── train_model.py        # Model training script
│   └── evaluate.py           # Model evaluation and metrics
│
├── tests/                    # Unit tests
│   └── test_model.py         # Tests for ML pipeline
│
├── notebooks/                # Jupyter notebooks for experimentation
│
├── requirements.txt          # Python dependencies
│
└── README.md                 # Project documentation
```

### Folder Description

- **data/** – Contains dataset files used for training and evaluation (not uploaded to GitHub due to size).
- **src/** – Main source code for preprocessing, training, and evaluating the machine learning model.
- **tests/** – Automated tests to verify that the ML pipeline works correctly.
- **notebooks/** – Jupyter notebooks used for experiments and exploratory data analysis.
- **requirements.txt** – Lists all Python libraries required to run the project.
- **README.md** – Documentation explaining the project, setup, and usage.
------------------------------------------------------------------------

## Installation

Clone the repository:

git clone
https://github.com/yourusername/network-intrusion-detection.git cd
network-intrusion-detection

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

## Usage

### 1. Preprocess Data

Run the preprocessing script:

python src/preprocess.py

This step performs: - Data cleaning - Feature encoding - Feature scaling

------------------------------------------------------------------------

### Run Training

To train the intrusion detection model:

```bash
python src/train.py --data_path data/kddcup.data_10_percent.gz
```

The training pipeline will:

- Load and preprocess the dataset
- Encode categorical features
- Scale features using **MinMaxScaler**
- Split the data into training, validation, and test sets
- Train multiple models
- Select the best-performing model
- Evaluate the model on the test set

The trained model is saved as:

```
best_model.pkl
```

------------------------------------------------------------------------
## Models Used

The training pipeline compares multiple machine learning models:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

The best model is selected based on **validation F1-score**.

-----------------------------------------------------------------------
### 3. Evaluate the Model

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

python src/evaluate.py

The evaluation includes: - Accuracy - Precision - Recall - Confusion
matrix

------------------------------------------------------------------------

## Testing

This repository includes a basic test suite to verify that the core
components work correctly.

### Test Structure

tests/ 
└── test_model.py

The tests validate: - Data loading - Model pipeline - Prediction
functionality

### Running Tests

python -m unittest discover tests

or

pytest tests

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Scikit-learn
-   Pandas
-   NumPy
-   Matplotlib
-   Jupyter Notebook

------------------------------------------------------------------------

## Future Improvements

Possible improvements to this project include:

-   Deep learning models for intrusion detection
-   Real-time network monitoring
-   Feature selection optimization
-   Hyperparameter tuning
-   Deployment as a web service

----------------------------------------------------------------------
