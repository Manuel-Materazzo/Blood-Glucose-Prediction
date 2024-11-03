import pandas as pd

from src.pipelines.brist1d_blood_glucose_prediction_dt_pipeline import BrisT1DBloodGlucosePredictionDTPipeline

from src.hyperparameter_optimizers.accurate_grid_optimizer import AccurateGridOptimizer
from src.hyperparameter_optimizers.balanced_grid_optimizer import BalancedGridOptimizer
from src.hyperparameter_optimizers.fast_bayesian_optimizer import FastBayesianOptimizer
from src.hyperparameter_optimizers.optuna_optimizer import OptunaOptimizer

from src.trainers.accurate_cross_trainer import AccurateCrossTrainer


def load_data():
    # Load the data
    file_path = '../resources/brist1d/train.csv'
    data = pd.read_csv(file_path)

    # Remove rows with missing target, separate target from predictors
    data.dropna(axis=0, subset=['bg+1:00'])
    y = data['bg+1:00']
    X = data.drop(['bg+1:00'], axis=1)
    return X, y


X, y = load_data()

pipeline = BrisT1DBloodGlucosePredictionDTPipeline(X, True)

trainer = AccurateCrossTrainer(pipeline)

# Create a cached trainer for the optimizer in order to speed up the optimization process
# optimizer_trainer = CachedAccurateCrossTrainer(pipeline, X, y)
# optimizer = OptunaOptimizer(optimizer_trainer)

# optimize parameters
optimizer = BalancedGridOptimizer(trainer)
optimized_params = optimizer.tune(X, y, 0.03)

print("params")
print(optimized_params)

print()
_, boost_rounds = trainer.validate_model(X, y, log_level=1, **optimized_params)
print()

# fit complete_model on all data from the training data
complete_model = trainer.train_model(X, y, rounds=boost_rounds, **optimized_params)
