from pandas import DataFrame
import pandas as pd
import numpy as np

from src.preprocessors.data_preprocessor import DataPreprocessor

activity_dictionary = {'None': 0, 'Yoga': 1, 'Walking': 2, 'Walk': 2, 'Dancing': 3, 'Zumba': 4,
                       'Strength training': 5,
                       'Weights': 6, 'Aerobic Workout': 7, 'Workout': 8, 'HIIT': 9, 'Run': 10, 'Running': 10,
                       'Bike': 11, 'Outdoor Bike': 11, 'Stairclimber': 12, 'Spinning': 13, 'Swim': 14,
                       'Swimming': 14,
                       'Tennis': 15, 'Indoor climbing': 16, 'Hike': 17, 'Sport': 18}

# Create a dictionary that maps each patient to a number
partecipants_dictionary = {'p01': 1, 'p02': 2, 'p03': 3, 'p04': 4, 'p05': 5, 'p06': 6, 'p07': 7, 'p08': 8, 'p09': 9,
                           'p10': 10, 'p11': 11, 'p12': 12, 'p13': 13, 'p14': 14, 'p15': 15, 'p16': 16, 'p17': 17,
                           'p18': 18, 'p19': 19, 'p20': 20, 'p21': 21, 'p22': 22, 'p23': 23, 'p24': 24}


def transform_time(X):
    # extract a hour and minute series
    time_col = pd.to_datetime(X['time'], format='%H:%M:%S')
    hours = time_col.dt.hour
    minutes = time_col.dt.minute

    # create a hour sin and cos wave
    X['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    X['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    # create a minute sin and cos wave
    X['minute_sin'] = np.sin(2 * np.pi * minutes / 60)
    X['minute_cos'] = np.cos(2 * np.pi * minutes / 60)
    # remove time
    X.drop(['time'], axis=1, inplace=True)


def encode_and_0fill(X):
    steps_columns = []
    activity_columns = []
    carbs_columns = []
    cals_columns = []

    for column in X.columns:
        # if the column is a metric with a survey time
        if 'steps_' in column:
            steps_columns.append(column)
        if 'activity_' in column:
            activity_columns.append(column)
        if 'carbs_' in column:
            carbs_columns.append(column)
        if 'cals_' in column:
            cals_columns.append(column)

    X[activity_columns] = X[activity_columns].fillna('None')  # assume no activity when empty

    # apply static encoding for activities
    for col in activity_columns:
        # Assign values based on the dictionary and set -1 for empty values
        X[col] = X[col].apply(lambda x: activity_dictionary.get(x, -1) if x != '' else -1)

    # apply static encoding for patients
    # Assign values based on the dictionary and set -1 for empty values
    X['p_num'] = X['p_num'].apply(lambda x: partecipants_dictionary.get(x, -1) if x != '' else -1)

    X[steps_columns] = X[steps_columns].fillna(0)  # assume 0 steps when empty
    X[carbs_columns] = X[carbs_columns].fillna(0)  # assume 0 carbohydrate intake when empty
    X[cals_columns] = X[cals_columns].fillna(0)  # assume 0 calories burned when empty


def fill_time_series(X):
    # create a dictionary of metrics and their corresponding columns
    metric_columns = {}
    for column in X.columns:
        if '_' in column:
            # get the metric name
            metric_name = column.split("_")[0]
            # get the array of existing columns with the same metrics (or create it) and add the current column
            metric_columns_names = metric_columns.get(metric_name) or []
            metric_columns_names.append(column)
            # save the new array on the dictionary
            metric_columns[metric_name] = metric_columns_names

    # iterate metrics
    for key, value in metric_columns.items():
        # Sort the metric columns in descending order
        columns = sorted(value, key=lambda x: '_'.join(x.split('_')[1:]), reverse=True)

        # Fill missing values using backfill (and forward fill if the latest values are empty)
        X[columns] = X[columns].bfill()
        X[columns] = X[columns].ffill()


class BloodGlucoseDataPreprocessor(DataPreprocessor):

    def preprocess_data(self, X: DataFrame):
        transform_time(X)
        encode_and_0fill(X)
        fill_time_series(X)
