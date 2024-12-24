import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import DMatrix, XGBRegressor

from src.enums.accuracy_metric import AccuracyMetric

from src.models.xgb_regressor import XGBRegressorWrapper
from src.hyperparameter_optimizers.accurate_grid_optimizer import AccurateGridOptimizer
from src.hyperparameter_optimizers.balanced_grid_optimizer import BalancedGridOptimizer
from src.hyperparameter_optimizers.fast_bayesian_optimizer import FastBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer
from src.pipelines.dt_pipeline import save_data_model
from src.pipelines.empty_dt_pipeline import EmptyDTPipeline
from src.preprocessors.blood_glucose_prediction_data_preprocessor import BloodGlucoseDataPreprocessor

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.trainer import save_model

def load_data():
    # Load the data
    file_path = '../resources/brist1d/train.csv'
    data = pd.read_csv(file_path, index_col='id')

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

# instantiate data pipeline and preprocessor
preprocessor = BloodGlucoseDataPreprocessor()
pipeline = EmptyDTPipeline(X)

# preprocess train data
# different from preprocess_data because of the data increase opportunity
preprocessor.preprocess_train_data(X)

# pick a model, and a trainer
model_type = XGBRegressorWrapper()
trainer = AccurateCrossTrainer(pipeline, model_type, AccuracyMetric.RMSE, grouping_columns=['p_num'])

# optimizing parameters worsens performance
optimized_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'n_jobs': -1
}

print("Training and evaluating model...")
_, boost_rounds, _ = trainer.validate_model(X, y, log_level=1, params=optimized_params)

# fit complete_model on all data from the training data
print("Fitting complete model...")
complete_model = trainer.train_model(X, y, iterations=boost_rounds, params=optimized_params)

# save preprocessor on target directory
print("Saving preprocessor...")
preprocessor.save_preprocessor()

# save trained pipeline on target directory
print("Saving pipeline...")
pipeline.save_pipeline()

# save model on target directory
print("Saving fitted model...")
save_model(complete_model)