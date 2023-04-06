Bank Customer Churn
==============================

Aim of this project is to mimic a real business case of calculating the Churn rate for a fictional bank accross three different European countries: France, Germany and Spain. Here we implement MLOps and made an industralized model through mlflow, as well as scripts that will automate prediction jobs. Additionally from our model we were able to extract the most impactful features to define customer strategy in a business context. 

We encourage you to read the report found here as well as this readme to understand project functionality. 


## Project Deliverables

- The **main** report can be found under the /reports/ folder called *Bank_Customer_Churn_MLFlow_Project_Reports.pdf*
- For project outputs: 
    - **Predictions on Test Data Set**: Can be found at the /project-outputs/ folder, the test data set and the prediction csv. 
    - **MLFlow Outputs**: Can be found at /src/models/mlruns, you can see the runs in /project-outputs/runs.csv, as well as figures under /reports/figures
    - **SHAP Outputs**: You can find a report explaining the findings under /reports/ as *Exploring Shapley Values of Bank Customer Churn* as a pdf, and the notebook with the code under /notebooks/ named *1.1-exploring-shapley-values.ipynb* 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original dataset in csv format.
    │
    ├── notebooks          <- Jupyter notebooks. Here you can find the model exploration and Shapley values notebook.
    │
    ├── project-outputs    <- MlFlow Output (runs.csv), test dataset (test_dataset.csv) and the test set predictions.
    │
    ├── reports            <- All reports pertaining to the project scope, including the main one.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py <- Pre-processes our data for model
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├
    │   │   ├── mlruns           <- Folder containing all the runs in mlflow
    │   │   ├── model_logistic   <- Folder containing the saved MLflow model for predictions
    │   │   ├── predict_model.py <- Script for predicting the model
    │   │   └── train_model.py   <- Main script for training model in mlflow
    │   │   └── MLproject        <- Main entry point for MLflow commands
    │   │   └── save_encoder     <- Serialized one hot encoder for pre-processing prediction jobs
    │   │   └── save_scaler      <- Serialized standard scaler for pre-processing prediction jobs
    │   │   └── polynomial       <- Serialized Feature polynomial transformation object for pre-processing jobs
    |________________________________________________________________________________________________________________
    



