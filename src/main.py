import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import DMatrix, XGBRegressor

from src.enums.accuracy_metric import AccuracyMetric
from src.pipelines.brist1d_blood_glucose_prediction_dt_pipeline import BrisT1DBloodGlucosePredictionDTPipeline

from src.models.xgb_regressor import XGBRegressorWrapper
from src.hyperparameter_optimizers.accurate_grid_optimizer import AccurateGridOptimizer
from src.hyperparameter_optimizers.balanced_grid_optimizer import BalancedGridOptimizer
from src.hyperparameter_optimizers.fast_bayesian_optimizer import FastBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer
from src.pipelines.dt_pipeline import save_data_model

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.trainer import save_model


def multiply_data(df):
    """
    Reduce data from the previous 6 hours of each row to 3 hours.
    Use the remaining 3 hours to create another row, effectively doubling the data.
    :param df:
    :return:
    """
    rows = []
    # for each row in the dataset
    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # row_dict is one row of the original dataset, we need to extract 2 rows out of it.

        # Create first 3 hours row by removing 3-6 columns (keep from 0:00 to 2:55)
        row1 = {k: v for k, v in row_dict.items() if k.split('-')[-1].startswith(('0', '1', '2')) or ':' not in k}
        # Prediction target is the same
        row1['bg+1:00'] = row_dict['bg+1:00']
        rows.append(row1)

        # The prediction target for row will be bg-2:00 (1 hour in the future for bg-3:00)
        # But sometimes is NaN, we'll get the first non NaN value
        row2_target = 0
        for col in ['bg-2:00', 'bg-2:05', 'bg-2:10', 'bg-2:15', 'bg-2:20']:
            if pd.notna(row_dict.get(col)):
                row2_target = row_dict[col]

        # Create last 3 hours row by removing 0-2 columns (keep from 3:00 to 5:55).
        row2 = {k: v for k, v in row_dict.items() if k.split('-')[-1].startswith(('3', '4', '5')) or ':' not in k}
        # Set the prediction target
        row2['bg+1:00'] = row2_target
        # Set the 'time' column to be the time of bg-3:00 collection (3 hours before the current 'time' value)
        hours_ago = pd.to_datetime(row2['time'], format='%H:%M:%S') - pd.Timedelta(hours=3)
        row2['time'] = hours_ago.strftime(r'%H:%M:%S')

        # Adjust column names for the second row to reflect the correct time intervals (shift 3 hours)
        corrected_row2 = {k.replace('3:', '0:').replace('4:', '1:').replace('5:', '2:'): v for k, v in row2.items()}

        rows.append(corrected_row2)

    return pd.DataFrame(rows).sort_values(by=['p_num', 'time']).reset_index(drop=True)


def load_data():
    # Load the data
    file_path = '../resources/brist1d/train.csv'
    data = pd.read_csv(file_path, index_col='id')

    # doubles data by reducing the time series duration from 6 hours to 3 hours
    data = multiply_data(data)

    # Remove rows with missing target, separate target from predictors
    data.dropna(axis=0, subset=['bg+1:00'])
    y = data['bg+1:00']
    X = data.drop(['bg+1:00'], axis=1)
    X = X.drop(['p_num'], axis=1)
    return X, y

print("Loading data...")
X, y = load_data()

# save model file for current dataset on target directory
print("Saving data model...")
save_data_model(X)

# instantiate data pipeline
pipeline = BrisT1DBloodGlucosePredictionDTPipeline(X, True)

# pick a model, and a trainer
model_type = XGBRegressorWrapper()
trainer = AccurateCrossTrainer(pipeline, model_type, AccuracyMetric.RMSE)

# optimizing parameters worsens performance
optimized_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'n_jobs': -1
}

print("Training and evaluating model...")
_, boost_rounds = trainer.validate_model(X, y, log_level=1, params=optimized_params)

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=boost_rounds, params=optimized_params)

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)

# Load the submission data
print("Loading submission data...")
test_data_path = '../resources/brist1d/test.csv'
test_data = pd.read_csv(test_data_path, index_col='id')
test_data.drop(['p_num'], axis=1,  inplace=True)

# drop the last 3 hours of data from each record to align features
features = []
for column in test_data.columns:
    if column.split('-')[-1].startswith(('0', '1', '2')) or ':' not in column:
        features.append(column)
test_data = test_data[features]

print("Processing data...")
processed_test_data = pipeline.transform(test_data)

# make predictions to submit.
print("Predicting target...")
test_preds = complete_model.predict(processed_test_data)

# save predictions in the format used for competition scoring
output = pd.DataFrame({'id': test_data.index,
                       'bg+1:00': test_preds})
output.to_csv('submission.csv', index=False)
