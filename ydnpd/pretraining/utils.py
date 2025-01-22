import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, classification_report

SMALL_CONSTANT = 1e-12

def _preprocess_data(df, target_column='y', unique_value_threshold=50, config=None):
    feature_columns = [col for col in df.columns if col != target_column]

    y = df[target_column].values
    X = df[feature_columns]

    # make the categories in order, 0 to k-1
    for column in X.columns:
        X.loc[:, column] = pd.Categorical(X[column]).codes 

    if config is None:
        config = {}
        for col in feature_columns:
            if df[col].nunique() <= unique_value_threshold:
                print(f"column {col} is categorical")
                config[col] = {'type': 'categorical'}
            else:
                print(f"column {col} is continuous")
                config[col] = {'type': 'continuous'}

    cat_features = [col for col in feature_columns if config[col]['type'] == 'categorical']
    cont_features = [col for col in feature_columns if config[col]['type'] == 'continuous']

    print((set(cat_features) | set(cont_features)))
    print(X.columns)
    missing = (set(cat_features) | set(cont_features)) - set(X.columns)
    if missing:
        raise KeyError(f"the following features are missing after transformation: {missing}")

    X_cat = X[cat_features].copy()
    X_cont = X[cont_features].copy()

    for col in cat_features:
        print(X_cat[col].unique())

    # adjusted cardinalities calculation
    cat_cardinalities = [X_cat[col].max()+1 for col in cat_features]

    X_cont = (X_cont - X_cont.mean()) / (X_cont.std() + SMALL_CONSTANT)

    (X_cat_train, 
     X_cat_valid, 
     X_cont_train, 
     X_cont_valid, 
     y_train, 
     y_valid) = train_test_split(X_cat, X_cont, y, test_size=0.2, random_state=0)

    X_cat_train_tensor = torch.tensor(X_cat_train.to_numpy(), dtype=torch.long)
    X_cat_valid_tensor = torch.tensor(X_cat_valid.to_numpy(), dtype=torch.long)

    X_cont_train_tensor = torch.tensor(X_cont_train.to_numpy(), dtype=torch.float32)
    X_cont_valid_tensor = torch.tensor(X_cont_valid.to_numpy(), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)

    print(f"cat training features shape: {X_cat_train.shape}")
    print(f"cont training features shape: {X_cont_train.shape}")
    print(f"cat val features shape: {X_cat_valid.shape}")
    print(f"cont val features shape: {X_cont_valid.shape}")
    print(f"training targets shape: {y_train.shape}")
    print(f"val targets shape: {y_valid.shape}")
    print(f"cat feature cards: {cat_cardinalities}")
    print(f"config: {config}")

    return (
        X_cat_train_tensor,
        X_cont_train_tensor,
        X_cat_valid_tensor,
        X_cont_valid_tensor,
        y_train_tensor,
        y_valid_tensor,
        cat_cardinalities,
        config
    )

def preprocess_acs_for_classification(df, random_state=42):
    acs_column_set = set(["SEX", "MSP", "RAC1P", "OWN_RENT", "PINCP_DECILE", "EDU", "HOUSING_TYPE"])
    assert set(df.columns) == acs_column_set
    
    # we will subset based on the two most common OWN_RENT categories
    df = df[df['OWN_RENT'] != 0].copy() 

    # set 'OWN_RENT' column as 'y' and remove it from data
    df.loc[:, 'y'] = df['OWN_RENT'] 
    df = df.drop('OWN_RENT', axis=1)

    # map y (which is 1, 2) to 0, 1
    df.loc[:, 'y'] = df['y'].to_numpy() - 1 
    
    # stratify the data
    df_pos = df[df['y'] == 1]
    df_neg = df[df['y'] == 0]

    min_size = min(len(df_pos), len(df_neg))

    df_pos = df_pos.sample(n=min_size, random_state=random_state)
    df_neg = df_neg.sample(n=min_size, random_state=random_state)

    df = pd.concat([df_pos, df_neg])
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def preprocess_acs_for_ft_transformer(df, config=None, random_state=42):
    df = preprocess_acs_for_classification(df, random_state=random_state)

    return _preprocess_data(df, target_column='y', unique_value_threshold=100, config=config)

def print_model_performance(classifier, X_cat_test, X_cont_test, y_test):
    y_pred_prob = classifier.predict_proba(X_cat_test, X_cont_test)[:, 1]
    
    # roc curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    # compute an f-score for the optimal threshold
    fscore = (2 * precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
    ix = np.argmax(fscore)
    optimal_threshold = thresholds[ix]
    
    # pred using the optimal threshold
    y_pred = classifier.predict(X_cat_test, X_cont_test, binary=True, threshold=optimal_threshold)
    
    # calc auc
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # print results
    print()
    print(f"AUC: {auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(classification_report(y_test, y_pred))
    print()



