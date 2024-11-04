import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipelines.dt_pipeline import DTPipeline


class CustomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        steps_columns = []
        activity_columns = []
        carbs_columns = []

        for column in X.columns:
            # if the column is a metric with a survey time
            if 'steps-' in column:
                steps_columns.append(column)
            if 'activity-' in column:
                activity_columns.append(column)
            if 'carbs-' in column:
                carbs_columns.append(column)

        X[steps_columns] = X[steps_columns].fillna(0)  # assume 0 steps when empty
        X[carbs_columns] = X[carbs_columns].fillna(0)  # assume 0 carbohydrate intake when empty
        X[activity_columns] = X[activity_columns].fillna("None")  # assume no activity when empty

        return X


class TimeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid altering the original data
        X_copy = X.copy()

        X_copy['hour'] = pd.to_datetime(X_copy['time'].values, format="%H:%M:%S").hour
        X_copy['minute'] = pd.to_datetime(X_copy['time'].values, format="%H:%M:%S").minute
        X_copy.drop(['time'], axis=1, inplace=True)

        return X_copy


class BackfillTransformer(BaseEstimator, TransformerMixin):

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
            columns = sorted(value, key=lambda x: int(x.split('-')[1]), reverse=True)

            # Fill missing values using backfill (and forward fill if the latest values are empty)
            X[columns] = X[columns].bfill()
            X[columns] = X[columns].ffill()

        return X_copy


class BrisT1DBloodGlucosePredictionDTPipeline(DTPipeline):
    def __init__(self, X: DataFrame, imputation_enabled: bool):
        super().__init__(X, imputation_enabled)

    def rename_columns(self, X):
        """
        Reformat metrics delta column names by replacing the time in hour:minutes into just minutes.
        :param X:
        :return:
        """
        columns_to_rename = {}

        for column in X.columns:
            # if the column is a metric with a survey time
            if '-' in column:
                # split column name into metric - time
                column_name_splits = column.split('-')
                metric_name = column_name_splits[0]
                survey_time_delta = column_name_splits[1]
                # split time into hour : minute
                survey_time_delta_splits = survey_time_delta.split(':')
                # get the total delta minutes by summing 60*hours and the minutes
                survey_time_delta_minutes = int(survey_time_delta_splits[0] * 60) + (int(survey_time_delta_splits[1]))
                # save rename operation
                new_column_name = metric_name + '-' + str(survey_time_delta_minutes)
                columns_to_rename[column] = new_column_name

        # rename all columns
        X = X.rename(columns=columns_to_rename)

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
            ('transform_time_columns', TimeTransformer()),
            ('rename_columns', FunctionTransformer(self.rename_columns, validate=False)),
            ('fill_metric_columns', BackfillTransformer()),
            ('custom_imputate_metric_columns', CustomImputer()),
            ('preprocessor', FunctionTransformer(self.create_preprocessor, validate=False))
        ], memory=None)
