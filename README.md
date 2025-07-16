# Customer Churn Prediction Dashboard

This project predicts whether a telecom customer will churn (leave the service) based on historical data. The solution includes **data preprocessing**, **model training**, **evaluation**, and **explanability** via SHAP - all accessible through a user-friendly **Streamlit Dasboard**.

---

## Project Overview
- **Dataset**: Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn), containing the data of 7043 customers and 21 features associated with each customer
- **Goal**: Predict customer churn and identify key drivers behind it
- **End to End pipeline**:
  - Data Cleaning and Feature Engineering
  - Model Training (4 models are contested: Logistic Regression, Decision Tree Classifier, XGBoost, Artificial Neural Network)
  - Model Evaluation (F1, ROC AUC, Confusion Matrix, Classification Report)
  - Model Explainability using SHAP
  - Interactive Web Dashboard with Streamlit
 
---

## Why Logistic Regression?
Only Logistic Regression is used as the final model and deployed onto the Dashboard. This is because after training the 4 models specified above (in Jupyter Environment), Logistic Regression performed **best overall** on ROC AUC and F1 score, most notably scoring higher on Churn prediction - which is the goal of the project. Other models predict Not Churn decently well, however the metrics fall off when predicting Churn.

--- 

## Project Structure
Customer-Churn-Predictor/
- app/
  - app.py: Streamlit app
- src/
  - data_loader.py: Data loading & preprocessing
  - model.py: Model training functions
  - evaluate.py: Evaluation metrics & plotting
  - shap_utils.py: SHAP explainability
  - plots/: Saved SHAP plots (png/html)
- main.py: Local testing script (not Streamlit)
- requirements.txt: Python dependencies
- README.md: Project overview

---

## Key Visuals in the Dashboard
- **Confusion Matrix, Classification Report**
- **ROC Curve**
- **SHAP Summary Plot (Bar and Bee Swarm)**
- **SHAP Force Plot (1 and 100 samples)**
- **Decision Plot and Waterfall Plot**
- **Logistic Coefficients Bar Chart**

---
## Model Explainability
Using SHAP (SHapley Additive exPlanations), the model predictions are explained clearly:
- Red: pushes direction towards Churn
- Blue: pushes direction towards Staying
- Interactive plots show how features like 'tenure', 'TotalCharges' influence the outcome, by different degree

---
## Testing and Validation
Before deploying the app, all components were:
- Built in a Jupyter Notebook (.ipynb)
- Modularised into Python scripts inside /src
- Tested using main.py
- Integrated and visualised using Streamlit (app.py)

---
## Future Work
- Add model selection dropdown in dashboard
- Allow user to input their own customer data, and the deployed model can predict whether the customer would churn
- Deploy on Streamlit Cloud
- Simulate ROI

---

## How to run
### Local
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
3. Run the app
   ```
   python -m streamlit run app/app.py
