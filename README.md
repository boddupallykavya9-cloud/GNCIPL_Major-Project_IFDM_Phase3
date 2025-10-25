# Insurance Fraud Detection - Feature Engineering, Model Training & Deployment

## Overview
This repository contains my contribution to the Insurance Fraud Detection group project, focusing on feature engineering, model training, evaluation, and deployment through a Streamlit web application.

## Project Structure
├── data/
│   ├── nhic-1.csv                          # Raw input dataset
│   └── insurance_fraud_prob_full.csv       # Output dataset with fraud probabilities
├── notebooks/
│   └── fraud_detection_modelling.ipynb     # Complete EDA, feature engineering & modeling workflow
├── models/
│   └── final_model.pkl                     # Trained model (pickle file)
├── streamlit_app.py                         # Interactive web app for predictions
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation

## My Contribution

### 1. Feature Engineering
- Date conversion and temporal feature creation (`claim_delay`)
- Categorical encoding (sex, region, smoker)
- Outlier detection and treatment using Z-score method
- Feature scaling with StandardScaler
- Handling missing values

### 2. Model Training & Evaluation
- Trained and compared three algorithms:
  - Random Forest Classifier
  - XGBoost Classifier
  - Logistic Regression
- Evaluated using Accuracy, Precision, Recall, and F1 Score
- Selected best model based on F1 Score

### 3. Deployment
- Saved trained model using pickle
- Built interactive Streamlit application for real-time predictions
- Generated fraud probability scores for all records
- Exported enriched dataset for dashboard and reporting teams

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
# Clone the repository
git clone https://github.com/yourusername/insurance-fraud-detection.git
cd insurance-fraud-detection
# Install dependencies
pip install -r requirements.txt
## Usage

### Run Jupyter Notebook
jupyter notebook notebooks/fraud_detection_modelling.ipynb
### Run Streamlit App
streamlit run streamlit_app.py
The app will open in your browser at http://localhost:8501

## Dataset
- *Input*: nhic-1.csv - Raw insurance claim data
- *Output*: insurance_fraud_prob_full.csv - Processed data with fraud probability scores

### Features
- Demographics: age, sex, region, children
- Medical: BMI, smoker status
- Financial: bill_amount, claimed_amount, amount_paid
- Temporal: duration, year_billing, claim_delay

## Models Evaluated
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |
| Logistic Regression | - | - | - | - |

Note: Fill in actual metrics from your results

## Key Insights
- No missing values in dataset
- Outliers detected and removed (66 records with Z-score > 3)
- Strong correlation between bill_amount, claimed_amount, and amount_paid
- Smokers have significantly higher medical costs
- Regional variations in claim patterns

## Technologies Used
- *Python 3.x*
- *Pandas* - Data manipulation
- *NumPy* - Numerical operations
- *Scikit-learn* - Machine learning models and preprocessing
- *XGBoost* - Gradient boosting
- *Matplotlib/Seaborn* - Data visualization
- *Streamlit* - Web app deployment

## Future Work
- Hyperparameter tuning for improved performance
- Additional feature engineering (interaction features)
- Integration with real-time data pipelines
- Model monitoring and retraining workflows

## Team Collaboration
This work serves as the foundation for:
- *Dashboard Team*: Uses insurance_fraud_prob_full.csv for visualization
- *Reporting Team*: Uses notebook and results for documentation

## Files

*   **`data/nhic-1.csv`**: Original dataset.
*   **`data/insurance_fraud_prob_full.csv`**: Processed data with added fraud probability predictions.
*   **`notebooks/fraud_detection_modelling.ipynb`**: Jupyter notebook detailing the data processing, model training, and evaluation.
*   **`models/final_model.pkl`**: Pickled file containing the trained model and scaler object.
*   **`streamlit_app.py`**: Python script for the Streamlit web application.
*   **`requirements.txt`**: List of Python dependencies required for the project.
*   **`README.md`**: Project documentation.
