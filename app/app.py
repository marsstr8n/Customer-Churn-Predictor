import streamlit as st
import pandas as pd
from src.data_loader import load_data, stratified_split, preprocessing_pipeline
from src.model import train_logistic_model
from src.evaluate import ModelEvaluator
from src.shap_utils import shap_explainer
import numpy as np
import os # to create folders (plots/)
import matplotlib.pyplot as plt

# streamlit setup and title
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.markdown("**By Marcus Tran**")
st.markdown("## Project Overview")
st.markdown("""
            I trained a Logistic Regression model on the 'Telco Customer Churn' dataset to predict whether a customer will churn (leave the telecom service).
            
            After training the dataset using Logistic Regression, Decision Tree Classifier, XGBoost and Artificial Neural Network, Logistic Regression gave the best result in terms of AUC score, precision, recall, f1-score.
            
            To make this interpretable, I used SHAP (SHapley Additive exPlanations), which helps to understand which features most influence each customer's likelihood to churn.
            """)

# === Prepare data
@st.cache_data
def prepare_data():
    # load data
    telco_churn = load_data() # load the customer data

    # split data
    train, test = stratified_split(telco_churn) # since this function returns two df - train and test catch them

    # separate feature and target
    X_train = train.drop("Churn", axis=1)
    X_test = test.drop("Churn", axis=1)

    y_train = train["Churn"].map({"No": 0, "Yes": 1})
    y_test = test["Churn"].map({"No": 0, "Yes": 1})

    # instantiate the preprocessing pipeline, and fit it
    pipeline = preprocessing_pipeline(telco_churn)
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    # ==== convert X_train and X_test back to DataFrame ====
    # get the feature names 
    cat_cols = telco_churn.select_dtypes(include=['object']).columns.drop("Churn") # only categorical data here
    num_cols = telco_churn.select_dtypes(include=['int64', 'float64']).columns

    # get the feature names of the one-hot encoded columns
    ohe = pipeline.named_transformers_["cat"].named_steps["onehot"]
    ohe_cols = ohe.get_feature_names_out(cat_cols)

    # full column names
    all_feature_names = list(ohe_cols) + list(num_cols)


    # Wrap back into DataFrames with correct index
    X_train_df = pd.DataFrame(X_train_prepared, columns=all_feature_names, index=train.index)
    X_test_df = pd.DataFrame(X_test_prepared, columns=all_feature_names, index=test.index)

    return X_train_df, X_test_df, y_train, y_test, pipeline, train, test

with st.spinner("Loading and preprocessing data..."):
    X_train_df, X_test_df, y_train, y_test, pipeline, train_df, test_df = prepare_data()
    

# === Train model button ===
if st.button("Train Logistic Regression model"):
    with st.spinner("Training..."):
        clf = train_logistic_model(X_train_df, y_train)
        coefficients = pd.DataFrame({
            'Feature': X_train_df.columns,
            'Coefficient': clf.coef_[0]
        })
        coefficients['OddsRatio'] = coefficients['Coefficient'].apply(lambda x: round(np.exp(x), 3))

    st.success("Model trained")
    
    # === Evaluation ===
    st.markdown("## Model Evaluation Metrics")
    evaluator = ModelEvaluator(X_test_df, y_test, target_names=["No Churn", "Churn"], mode="streamlit")
    
    import io, sys
    output = io.StringIO()
    sys.stdout = output
    evaluator.evaluate(clf)
    sys.stdout = sys.__stdout__
    st.code(output.getvalue())
    
    # get feature importance
    
    # === SHAP ===
    st.markdown("## SHAP feature explanation")
    
    with st.spinner("Getting SHAP plots..."):
        os.makedirs("plots", exist_ok=True)
        shap_explainer(clf, X_train_df, X_test_df)
        
    # display feature importance plot
    st.markdown("### Feature Importance")
    st.markdown("""
                This chart ranks the features (like 'tenure' or 'TotalCharges') by how strongly they influence the model’s predictions across all customers.
                
                Features such as 'tenure', 'TotalCharges_numeric', 'Contract_Two year' have a strong influence on the model prediction. However, we do not know the direction of influence since this displays mean absolute SHAP value""")
    st.image("plots/bar_summary.png", caption="The importance of each feature - ranked from highest to lowest")

        
    # display the direction of the feature
    st.markdown("## Logistic Regression Coefficients")
    st.markdown("""
    - A **positive coefficient** means the feature increases the likelihood of churn.
    - A **negative coefficient** means the feature reduces the likelihood of churn.
    - The **Odds Ratio** shows how much more (or less) likely the customer is to churn, if this feature increases by one unit.
    
    ### Explanations:
    - The longer a customer stays with the company (tenure), the less likely they are to churn
    - High TotalCharges risks a customer churning
    - Specific services also play a role (e.g. OnlineSecurity_No), customers who opted out of online security tend to churn more
    - Long contracts such as 2 years could lead to customers staying for longer with the telecom provider
    
    Insight: The company should pay special attention to newer customers, long-term contracts should be offered to ensure they stay with the service for longer
    """)

    st.dataframe(coefficients, use_container_width=True)
    
    # bar chart of coefficients
    # Set a threshold to filter out low-impact coefficients
    threshold = 0.05  
    # Filter coefficients with absolute value above threshold
    filtered = coefficients[np.abs(coefficients['Coefficient']) >= threshold]
    filtered_sorted = filtered.sort_values('Coefficient')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(
        filtered_sorted['Feature'], 
        filtered_sorted['Coefficient'], 
        color=np.where(filtered_sorted['Coefficient'] > 0, 'red', 'blue')
    )
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title("Logistic Regression Coefficients (Filtered)")
    ax.set_xlabel("Coefficient Value (Red: Churn, Blue: Not Churn)")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.pyplot(fig)
        
        
    # display SHAP force plot HTMLs
    st.markdown("### SHAP Force Plot for One Sample")
    st.markdown("""
    This Force Plot shows which features pushed this customer's predicted churn probability higher or lower.

    - **Red bars** indicate features that push the prediction **toward churning**.
    - **Blue bars** indicate features that push the prediction **toward staying** (not churning).

    In this example, features like **'tenure'**, **'Contract_Two year'**, and **'TotalCharges'** strongly influence the prediction.

    - The **base value** is the model’s expected value (average log-odds across all customers).
    - The **f(x)** value is the final prediction in log-odds for this specific customer.

    Since **f(x) = -3.96**, which corresponds to a very low probability of churn, the model predicts that this customer is **very likely to stay**.
    """)
    with open("plots/force_plot_1.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=300)
    
    st.markdown("### SHAP Force Plot for 100 Samples")
    st.markdown("""
    This Force Plot shows the aggregated force plot for the first 100 customers.

    Red pushes are consistent with short-tenure customers for example.
    """)

    with open("plots/force_plot_100.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=400)
        
    # display waterfall plot
    st.markdown("### Waterfall Plot - for One Sample")
    st.markdown("""
    Another visual of how each feature contributes positively or negatively to a churn prediction

    It shows the biggest churn drivers for one customer at a glance
    """)

    st.image("plots/waterfall_plot.png", caption="Waterfall Plot for One Customer")

    # display bee swarm plot
    st.markdown("### Bee Swarm Plot")
    st.markdown("""
    Beeswarm plot reveals not just the relative importance of features, but the actual relationships with the churn outcome.

    Continuous features are shown as a range, while discrete features are shown as multiple lines.
    Again, from the plot, high 'tenure' value would lead to low SHAP value, which would result in a customer not churn, while the case is reverse with 'TotalCharges'
    """)

    st.image("plots/bee_swarm.png", caption="Bee Swarm Plot")


    # Summary
    st.markdown("""
                ## Summary
                ###Top Churn Predictors:
                - Low Tenure
                - High Total Charge
                - Lack of value-added services such as online security
                - Use of electronic check payments
                - Short term plan like month-to-month
                
                ### Actionable Suggestions:
                - Target new customers early with retention strategies
                - Bundle services (security, tech support) into promotions
                - Explore why electronic check users tend to churn
                """)