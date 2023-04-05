import mlflow.sklearn
import os, sys, numpy, joblib
import pandas as pd


load_model = mlflow.sklearn.load_model("model_logistic")

test_df = pd.read_csv(r"../../project-outputs/test_dataset.csv")
test_df = test_df.drop(columns=test_df.columns[0])

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_X = test_df.select_dtypes(include=numerics)
numeric_cols = list(numeric_X.columns)

features = test_df.columns
categorical_cols = list(set(features) - set(numeric_cols))

print(numeric_cols)
print(categorical_cols)

sc = joblib.load("save_scaler")
enc = joblib.load("save_encoder")
poly = joblib.load("polynomial.joblib")

processed_num_x = sc.transform(test_df[numeric_cols])
processed_cat_x = enc.transform(test_df[categorical_cols])

xtest_scl = numpy.hstack([processed_num_x, processed_cat_x.todense()])
poly_test = poly.transform(numpy.asarray(xtest_scl))

y_pred = load_model.predict(numpy.asarray(poly_test))

print(y_pred[:10], type(y_pred))
df = pd.DataFrame({"predictions": y_pred})
df.to_csv("../../project-outputs/predict_output.csv")

