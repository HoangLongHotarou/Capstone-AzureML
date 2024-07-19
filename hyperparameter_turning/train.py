from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset

def clean_data(data):
    x_df = data.to_pandas_dataframe().dropna()
    x = x_df.drop(['DEATH_EVENT'],axis=1)
    y = x_df['DEATH_EVENT']
    return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=200, help="max depth of random forest")
    parser.add_argument('--max_depth', type=int, default=2, help="max depth of random forest")
    parser.add_argument('--min_samples_leaf', type=int, default=5, help="min sample leaf of random forest")
    parser.add_argument('--criterion', type=str, default="entropy", help="criterion of random forest")
    parser.add_argument("--input-data", type=str)

    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("max_depth:", np.int(args.max_depth))
    run.log("min_samples_leaf:", np.int(args.min_samples_leaf))
    run.log("criterion:", str(args.criterion))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    
    dataset = Dataset.get_by_id(ws, id=args.input_data)

    x, y = clean_data(dataset)

    # TODO: Split data into train and test sets.
    ### YOUR CODE HERE ###a
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.3,random_state=1)
    model = RandomForestClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion
    ).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    os.makedirs('outputs',exist_ok = True)
    joblib.dump(model,'outputs/model.pkl')
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()