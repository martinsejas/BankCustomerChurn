from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
def preprocess_numeric_data(numeric_cols, xtrain, xtest, xval):
    scaler = StandardScaler()
    xtrain_scl1 = scaler.fit_transform(xtrain[numeric_cols])
    xval_scl1 = scaler.transform(xval[numeric_cols])
    xtest_scl1 = scaler.transform(xtest[numeric_cols])
    joblib.dump(scaler, "../models/save_scaler")
    return xtrain_scl1, xtest_scl1, xval_scl1

def preprocess_categorical_data(categorical_cols, xtrain, xtest, xval):
    encoder = OneHotEncoder()
    xtrain_scl2 = encoder.fit_transform(xtrain[categorical_cols])
    xval_scl2 = encoder.transform(xval[categorical_cols])
    xtest_scl2 = encoder.transform(xtest[categorical_cols])
    joblib.dump(encoder, "../models/save_encoder")
    return xtrain_scl2, xtest_scl2, xval_scl2

def combine_processed_data(num_x, cat_x):
    xtrain_scl = np.hstack([num_x[0], cat_x[0].todense()])
    xval_scl = np.hstack([num_x[2], cat_x[2].todense()])
    xtest_scl = np.hstack([num_x[1], cat_x[1].todense()])

    return xtrain_scl, xtest_scl, xval_scl