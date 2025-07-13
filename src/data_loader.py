import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data(path="data/telco_customer_churn.csv"):
    # read the file from the specified path
    df = pd.read_csv(path)
    
    # convert TotalCharges to numeric, coerce invalids to NaN
    df["TotalCharges_numeric"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # drop rows with missing TotalCharges (11 rows with blank strings)
    df = df[df["TotalCharges_numeric"].notna()].copy()
    
    # drop CustomerID - not predictive
    df.drop(columns=["customerID", "TotalCharges"], inplace=True) # drop the original TotalCharges as well to not mess up encoding
    
    return df

def stratified_split(df, target="Churn",test_size=0.2, random_state=42):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in splitter.split(df, df[target]):
        train = df.iloc[train_index].copy() # select train row
        test = df.iloc[test_index].copy() # select test row
    
    return train, test

def preprocessing_pipeline(df):
    cat_cols = df.select_dtypes(include=['object']).columns.drop("Churn") # only categorical data here
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    num_pipeline = Pipeline([
        ("scaler", MinMaxScaler())
    ])
    
    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline, cat_cols),
        ("num", num_pipeline, num_cols),
    ])
    
    return full_pipeline