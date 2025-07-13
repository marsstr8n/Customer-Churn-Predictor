from src.data_loader import load_data, stratified_split, preprocessing_pipeline
import pandas as pd
from src.evaluate import ModelEvaluator
from src.model import train_logistic_model

# load data
telco_churn = load_data() # load the customer data

# split data
train, test = stratified_split(telco_churn) # since this function returns two df - train and test catch them

# separate feature and target
X_train = train.drop("Churn", axis=1)
y_train = train['Churn'].copy()

X_test = test.drop("Churn", axis=1)
y_test = test['Churn'].copy()

# instantiate the preprocessing pipeline, and fit it
pipeline = preprocessing_pipeline(telco_churn)
X_train_prepared = pipeline.fit_transform(X_train)
X_test_prepared = pipeline.transform(X_test)

y_train = train["Churn"].map({"No": 0, "Yes": 1})
y_test = test["Churn"].map({"No": 0, "Yes": 1})

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

# train the model
clf = train_logistic_model(X_train_df, y_train)

# Model evaluator
evaluator = ModelEvaluator(X_test_df, y_test, target_names=["No Churn", "Churn"])
evaluator.evaluate(clf)

