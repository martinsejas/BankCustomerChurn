import os
import warnings
import sys

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score
from sklearn.metrics import classification_report

fpath = os.path.dirname(os.path.abspath("__file__"+ "/../"))
sys.path.append(os.path.join(fpath,"features"))
import src.features.build_features
#from features.build_features import *

def split_train_test(X, y, size=0.1, seed=42):
    xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=size, random_state=seed, stratify=y)
    xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=size / (1 - size), random_state=seed,
                                                    stratify=ytrain)

    return xtrain, xtest, xval, ytrain, ytest, yval

def eval_metrics(yval, ypred):
    cm = confusion_matrix(yval, ypred)
    preci = precision_score(yval, ypred)
    acc = accuracy_score(yval, ypred)
    f1_sc = f1_score(yval, ypred)
    cl_report = classification_report(y_true=yval, y_pred=ypred)

    return cm, preci, acc, f1_sc, cl_report


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    filepath = r"../../data/raw/Bank Customer Churn Prediction.csv"
    try:
        df = pd.read_csv(filepath)

        X = df.iloc[:, 1:-1]
        Y = df.iloc[:, -1]

        xtrain, xtest, xval, ytrain, ytest, yval = split_train_test(X, Y, 0.1)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_X = X.select_dtypes(include=numerics)
        numeric_cols = list(numeric_X.columns)

        features = X.columns
        categorical_cols = list(set(features) - set(numeric_cols))

        processed_num_x = build_features.preprocess_numeric_data(numeric_cols, xtrain, xtest, xval)
        processed_cat_x = build_features.preprocess_categorical_data(categorical_cols, xtrain, xtest, xval)

        xtrain_scl, xtest_scl, xval_scl = build_features.combine_processed_data(processed_num_x, processed_cat_x)

        reg = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001

        with mlflow.start_run():
            poly = PolynomialFeatures(degree=3, include_bias=True)
            x_poly = poly.fit_transform(np.asarray(xtrain_scl))

            poly_reg_model = LogisticRegression(class_weight={0: 1, 1: 1}, C=reg)
            poly_reg_model.fit(x_poly, ytrain)

            ypred_poly = poly_reg_model.predict(poly.transform(np.asarray(xval_scl)))

            cm, preci, acc, f1_sc, cl_report = eval_metrics(yval, ypred_poly)

            print("Polynomial Features + Logistic regression model (C={:f}):".format(reg))
            print("  Accuracy: %s" % acc)
            print("  Precision: %s" % preci)
            print("  f1-score: %s" % f1_sc)
            print("  Confusion Matrix:\n %s" % cm)

            mlflow.log_param("C", reg)
            mlflow.log_param("accuracy", acc)
            mlflow.log_metric("preicison", preci)
            mlflow.log_metric("f1_score", f1_sc)
            #mlflow.log_metric("confusion_metrics", cm)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(poly_reg_model, "model", registered_model_name="Polynomial+Logistic")
            else:
                mlflow.sklearn.log_model(poly_reg_model, "model")

    except Exception as e:
        print("Exception occcured: ",e)






