import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipelines.dt_pipeline import DTPipeline

# Create a dictionary that maps each unique activity to a number, activities are sorted by their physical strain
activity_dictionary = {'None': 0, 'Yoga': 1, 'Walking': 2, 'Walk': 2, 'Dancing': 3, 'Zumba': 4, 'Strength training': 5,
                       'Weights': 6, 'Aerobic Workout': 7, 'Workout': 8, 'HIIT': 9, 'Run': 10, 'Running': 10,
                       'Bike': 11, 'Outdoor Bike': 11, 'Stairclimber': 12, 'Spinning': 13, 'Swim': 14, 'Swimming': 14,
                       'Tennis': 15, 'Indoor climbing': 16, 'Hike': 17, 'Sport': 18}

partecipants_dictionary = {'p01': 1, 'p02': 2, 'p03': 3, 'p04': 4, 'p05': 5, 'p06': 6, 'p07': 7, 'p08': 8, 'p09': 9,
                           'p10': 10, 'p11': 11, 'p12': 12, 'p13': 13, 'p14': 14, 'p15': 15, 'p16': 16, 'p17': 17,
                           'p18': 18, 'p19': 19, 'p20': 20, 'p21': 21, 'p22': 22, 'p23': 23, 'p24': 24}


class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        steps_columns = []
        activity_columns = []
        carbs_columns = []
        cals_columns = []

        for column in X.columns:
            # if the column is a metric with a survey time
            if 'steps-' in column:
                steps_columns.append(column)
            if 'activity-' in column:
                activity_columns.append(column)
            if 'carbs-' in column:
                carbs_columns.append(column)
            if 'cals-' in column:
                cals_columns.append(column)

        X[activity_columns] = X[activity_columns].fillna('None')  # assume no activity when empty

        # apply static encoding for activities
        for col in activity_columns:
            # Assign values based on the dictionary and set -1 for empty values
            X[col] = X[col].apply(lambda x: activity_dictionary.get(x, -1) if x != '' else -1)

        # apply static encoding for patients
        # Assign values based on the dictionary and set -1 for empty values
        # X['p_num'] = X['p_num'].apply(lambda x: partecipants_dictionary.get(x, -1) if x != '' else -1)

        X[steps_columns] = X[steps_columns].fillna(0)  # assume 0 steps when empty
        X[carbs_columns] = X[carbs_columns].fillna(0)  # assume 0 carbohydrate intake when empty
        X[cals_columns] = X[cals_columns].fillna(0)  # assume 0 calories burned when empty

        return X


class TimeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid altering the original data
        X_copy = X.copy()

        # extract a hour and minute series
        time_col = pd.to_datetime(X_copy['time'], format='%H:%M:%S')
        hours = time_col.dt.hour
        minutes = time_col.dt.minute

        # create a hour sin and cos wave
        X_copy['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        X_copy['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        # create a minute sin and cos wave
        X_copy['minute_sin'] = np.sin(2 * np.pi * minutes / 60)
        X_copy['minute_cos'] = np.cos(2 * np.pi * minutes / 60)
        # remove time
        X_copy.drop(['time'], axis=1, inplace=True)

        return X_copy


class BackfillForwardFillTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid altering the original data
        X_copy = X.copy()

        # create a dictionary of metrics and their corresponding columns
        metric_columns = {}
        for column in X.columns:
            if '-' in column:
                # get the metric name
                metric_name = column.split("-")[0]
                # get the array of existing columns with the same metrics (or create it) and add the current column
                metric_columns_names = metric_columns.get(metric_name) or []
                metric_columns_names.append(column)
                # save the new array on the dictionary
                metric_columns[metric_name] = metric_columns_names

        # iterate metrics
        for key, value in metric_columns.items():
            # Sort the metric columns in descending order
            columns = sorted(value, key=lambda x: x.split('-')[1], reverse=True)

            # Fill missing values using backfill (and forward fill if the latest values are empty)
            X[columns] = X[columns].bfill()
            X[columns] = X[columns].ffill()

        return X_copy


class BrisT1DBloodGlucosePredictionDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame, imputation_enabled: bool):
        super().__init__(X, imputation_enabled)

    def refresh_columns(self, X):
        """
        :param X:
        :return:
        """

        self.categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
        self.numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        return X

    def create_preprocessor(self, X):
        # Encoding for categorical data
        categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='median')

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', categorical_encoder)
        ], memory=None)

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, self.numerical_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])

        return preprocessor.fit_transform(X)

    def build_pipeline(self) -> Pipeline | ColumnTransformer:
        # Bundle preprocessing
        return Pipeline(steps=[
            # sin cos time
            ('transform_time_columns', TimeTransformer()),
            # 0fill some columns and map to dictionaries
            ('custom_imputate_metric_columns', CustomImputer()),
            # refreshes column names
            ('refresh_columns', FunctionTransformer(self.refresh_columns, validate=False)),
            # backfill and forwardfill metric columns
            ('fill_metric_columns', BackfillForwardFillTransformer()),
            # impute remaining data
            ('preprocessor', FunctionTransformer(self.create_preprocessor, validate=False)),
            ('scale', StandardScaler()),
        ], memory=None)
