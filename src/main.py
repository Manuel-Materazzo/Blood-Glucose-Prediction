import pandas as pd
<<<<<<< HEAD

from src.pipelines.brist1d_blood_glucose_prediction_dt_pipeline import BrisT1DBloodGlucosePredictionDTPipeline
from src.trainer import Trainer
from src.hyperparameter_optimizers.accurate_grid_optimizer import AccurateGridOptimizer
from src.hyperparameter_optimizers.balanced_grid_optimizer import BalancedGridOptimizer
from src.hyperparameter_optimizers.fast_bayesian_optimizer import FastBayesianOptimizer
=======
import time
from src.pipelines.housing_prices_competition_dt_pipeline import HousingPricesCompetitionDTPipeline
from src.trainers.simple_trainer import SimpleTrainer
from src.trainers.fast_cross_trainer import FastCrossTrainer
from src.trainers.accurate_cross_trainer import AccurateCrossTrainer
from src.trainers.cached_accurate_cross_trainer import CachedAccurateCrossTrainer
from src.hyperparameter_optimizers.custom_grid_optimizer import CustomGridOptimizer
>>>>>>> 851e3cd5579b762c7a05c46c0c0a671c3b781069


def load_data():
    # Load the data
<<<<<<< HEAD
    file_path = '../resources/brist1d/train.csv'
    data = pd.read_csv(file_path)

    # Remove rows with missing target, separate target from predictors
    data.dropna(axis=0, subset=['bg+1:00'])
    y = data['bg+1:00']
    X = data.drop(['bg+1:00'], axis=1)
=======
    iowa_file_path = '../resources/train.csv'
    home_data = pd.read_csv(iowa_file_path)

    # Remove rows with missing target, separate target from predictors
    pruned_home_data = home_data.dropna(axis=0, subset=['SalePrice'])
    y = pruned_home_data.SalePrice
    X = pruned_home_data.drop(['SalePrice'], axis=1)
>>>>>>> 851e3cd5579b762c7a05c46c0c0a671c3b781069
    return X, y


X, y = load_data()

<<<<<<< HEAD
pipeline = BrisT1DBloodGlucosePredictionDTPipeline(X, True)

trainer = Trainer(pipeline)

# optimize parameters
optimizer = FastBayesianOptimizer(trainer)
optimized_params = optimizer.tune(X, y, 0.03)

print("params")
print(optimized_params)

print()
_, boost_rounds = trainer.cross_validation(X, y, log_level=1, **optimized_params)
print()

# fit complete_model on all data from the training data
complete_model = trainer.train_model(X, y, rounds=boost_rounds, **optimized_params)
=======
pipeline = HousingPricesCompetitionDTPipeline(X, True)

trainer = CachedAccurateCrossTrainer(pipeline, X, y)

optimizer = CustomGridOptimizer(trainer)

# optimize parameters
start = time.time()
optimized_params = optimizer.tune(X, y, 0.03)
end = time.time()

print("Tuning took {} seconds".format(end - start))

_, boost_rounds = trainer.validate_model(X, y, log_level=1, **optimized_params)
print()

# fit complete_model on all data from the training data
# complete_model = trainer.train_model(X, y, rounds=boost_rounds, **params)
>>>>>>> 851e3cd5579b762c7a05c46c0c0a671c3b781069
