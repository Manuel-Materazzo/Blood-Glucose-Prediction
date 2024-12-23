from pandas import DataFrame
import pandas as pd
import numpy as np

from src.preprocessors.data_preprocessor import DataPreprocessor
from pandas.api.types import is_datetime64_any_dtype as is_datetime

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

time_series = {
    '0': ['_0_55', '_0_50', '_0_45', '_0_40', '_0_35', '_0_30', '_0_25', '_0_20', '_0_15', '_0_10', '_0_05', '_0_00'],
    '1': ['_1_55', '_1_50', '_1_45', '_1_40', '_1_35', '_1_30', '_1_25', '_1_20', '_1_15', '_1_10', '_1_05', '_1_00'],
    '2': ['_2_55', '_2_50', '_2_45', '_2_40', '_2_35', '_2_30', '_2_25', '_2_20', '_2_15', '_2_10', '_2_05', '_2_00'],
    '3': ['_3_55', '_3_50', '_3_45', '_3_40', '_3_35', '_3_30', '_3_25', '_3_20', '_3_15', '_3_10', '_3_05', '_3_00'],
    '4': ['_4_55', '_4_50', '_4_45', '_4_40', '_4_35', '_4_30', '_4_25', '_4_20', '_4_15', '_4_10', '_4_05', '_4_00'],
    '5': ['_5_55', '_5_50', '_5_45', '_5_40', '_5_35', '_5_30', '_5_25', '_5_20', '_5_15', '_5_10', '_5_05', '_5_00']
}

mesas_cols = ['bg', 'insulin', 'steps', 'hr', 'cals']
divisible_cols = ['bg', 'hr']


def extract_sub_row(row, time_serie_suffixes, target_column, time_delta):
    # transform the row into a dictionary
    row_dict = row.to_dict()

    # create subrow object
    sub_row = {}

    # add patient number
    sub_row['p_num'] = row['p_num']

    # set the real time, by subtracting the delta from the time column
    sub_row['time'] = pd.to_datetime(row['time'], format='%H:%M:%S') - pd.Timedelta(hours=time_delta)

    # set the new target
    sub_row['target'] = row[target_column]

    # add columns containing the provided time series suffix list
    for key, value in row_dict.items():

        # check if the column contains one of the time series suffixes
        for index, suffix in enumerate(time_serie_suffixes):

            if suffix in key:
                # get metric name from key(bg_X_XX)
                metric_name = key.split('_')[0]

                # generate new column name
                new_column_name = f'{metric_name}_{str(index)}'

                # add column to the row
                sub_row[new_column_name] = value

    return sub_row


def split_data(df):
    """
    Split a row with 6 hours of data into 6 rows with 1 hour of data
    :param df:
    :return:
    """
    rows = []

    # for each row in the dataset
    for idx, row in df.iterrows():
        # extract 6 one hour rows
        sub_row_0 = extract_sub_row(row, time_series['0'], 'target', 0)
        rows.append(sub_row_0)
        for i in range(1, 6):
            target_column = f'bg_{str(i - 1)}_00'
            sub_row = extract_sub_row(row, time_series[str(i)], target_column, i)
            rows.append(sub_row)

    return pd.DataFrame(rows).sort_values(by=['p_num', 'time']).reset_index(drop=True)


def drop_prediction_columns(df):
    """
    Drops all the columns with data older than one hour
    :param df:
    :return:
    """
    cols_to_drop = []
    for column in df.columns:
        # rename first hour data columns
        for idx, suffix in enumerate(time_series['0']):
            if suffix in column:
                # get metric name from key(bg_X_XX)
                metric_name = column.split('_')[0]
                # generate new column name
                new_column_name = f'{metric_name}_{str(idx)}'
                # add new column
                df[new_column_name] = df[column]
                # remove old column
                cols_to_drop.append(column)

        for i in range(1, 6):
            if any(time_serie_suffix in column for time_serie_suffix in time_series[str(i)]):
                cols_to_drop.append(column)
    df.drop(cols_to_drop, axis=1, inplace=True)


def create_features(df):
    # Create empty dictionaries to hold intermediate columns
    new_cols = {}

    # Generate mean, std, and zscore of latest value
    for col in mesas_cols:
        # Calculate mean and standard deviation
        mean_col = f'mean_patient_{col}_11'
        std_col = f'std_patient_{col}_11'
        norm_col = f'norm_patient_{col}_11'

        mean_series = df.groupby('p_num')[f'{col}_11'].transform('mean')
        std_series = df.groupby('p_num')[f'{col}_11'].transform('std')
        norm_series = (df[f'{col}_11'] - mean_series) / std_series

        new_cols[mean_col] = mean_series
        new_cols[std_col] = std_series
        new_cols[norm_col] = norm_series

    # Pairwise feature interactions on 3 latest values
    for i in range(len(mesas_cols)):
        for j in range(i + 1, len(mesas_cols)):
            for k in [1, 2, 3]:
                time_index = 12 - k
                col_1 = f'{mesas_cols[i]}_{time_index}'
                col_2 = f'{mesas_cols[j]}_{time_index}'

                # bg_11_plus_hr_11 = bg_11 + hr_11
                new_cols[f'{col_1}_plus_{col_2}'] = df[col_1] + df[col_2]

                # bg_11_minus_hr_11 = bg_11 - hr_11
                new_cols[f'{col_1}_minus_{col_2}'] = df[col_1] - df[col_2]

                # bg_11_multiplied_hr_11 = bg_11 * hr_11
                new_cols[f'{col_1}_multiplied_{col_2}'] = df[col_1] * df[col_2]

                # bg_11_divided_hr_11 = bg_11 / hr_11 + 1e-15
                new_cols[f'{col_1}_divided_{col_2}'] = df[col_1] / (df[col_2] + 1e-15)

            diff_col_name = f'{mesas_cols[i]}_11_minus_{mesas_cols[j]}_11_minus_{mesas_cols[i]}_10_minus_{mesas_cols[j]}_10'
            new_cols[diff_col_name] = new_cols[f'{mesas_cols[i]}_11_minus_{mesas_cols[j]}_11'] - new_cols[
                f'{mesas_cols[i]}_10_minus_{mesas_cols[j]}_10']

    # Compute differences and combinations of all values
    for col in mesas_cols:
        for i in range(1, 12):
            # bg_11_minus_bg_10 = bg_11 - bg_10
            new_cols[f'{col}_{i}_minus_{col}_{i - 1}'] = df[f'{col}_{i}'] - df[f'{col}_{i - 1}']
            # bg_11_plus_bg_10 = bg_11 + bg_10
            new_cols[f'{col}_{i}_plus_{col}_{i - 1}'] = df[f'{col}_{i}'] + df[f'{col}_{i - 1}']
            # bg_11_multiplied_bg_10 = bg_11 * bg_10
            new_cols[f'{col}_{i}_multiplied_{col}_{i - 1}'] = df[f'{col}_{i}'] * df[f'{col}_{i - 1}']

    # Compute division only on selected columns
    for col in divisible_cols:
        for i in range(1, 12):
            # bg_11_divided_bg_10 = bg_11 / bg_10
            new_cols[f'{col}_{i}_divided_{col}_{i - 1}'] = df[f'{col}_{i}'] / df[f'{col}_{i - 1}']

    # Assign all new columns to the DataFrame at once to reduce fragmentation
    df.assign(**new_cols)

    return df


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
        # if 'carbs_' in column:
        #    carbs_columns.append(column)
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

    X[steps_columns] = X[steps_columns].fillna(0).astype(np.float32)  # assume 0 steps when empty
    # X[carbs_columns] = X[carbs_columns].fillna(0)  # assume 0 carbohydrate intake when empty
    X[cals_columns] = X[cals_columns].fillna(0)  # assume 0 calories burned when empty

    # log1p on step columns to reduce skeweness
    X[steps_columns] = np.log1p(X[steps_columns])


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


def transform_time(X):
    # extract a hour and minute series
    if not is_datetime(X['time']):
        X['time'] = pd.to_datetime(X['time'], format='%H:%M:%S')
    X["hour"] = X['time'].dt.hour
    X["minute"] = X['time'].dt.minute

    # create half day sin cos wave
    X['sin_halfday'] = np.sin(2 * np.pi * X['hour'] / 12)
    X['cos_halfday'] = np.cos(2 * np.pi * X['hour'] / 12)
    # create a hour sin and cos wave
    X['hour_sin'] = np.sin(2 * np.pi * X["hour"] / 24)
    X['hour_cos'] = np.cos(2 * np.pi * X["hour"] / 24)
    # create a minute sin and cos wave
    X['minute_sin'] = np.sin(2 * np.pi * X["minute"] / 60)
    X['minute_cos'] = np.cos(2 * np.pi * X["minute"] / 60)

    # remove time
    X.drop(['time'], axis=1, inplace=True)


def drop_features(X):
    # drop carbs
    X.drop([c for c in X.columns if 'carbs' in c], axis=1, inplace=True)


def delete_strong_correlated_columns(X):
    # reduce  overfitting, strong correlations between nearest timestep features
    X.drop([f'bg_{i}' for i in range(12) if i % 2 != 0], axis=1, inplace=True)


class BloodGlucoseDataPreprocessor(DataPreprocessor):

    def preprocess_data(self, X):
        drop_features(X)
        encode_and_0fill(X)
        fill_time_series(X)
        drop_prediction_columns(X)
        create_features(X)
        transform_time(X)
        delete_strong_correlated_columns(X)

    def preprocess_train_data(self, X):
        drop_features(X)
        encode_and_0fill(X)
        fill_time_series(X)
        X = split_data(X)
        create_features(X)
        transform_time(X)
        delete_strong_correlated_columns(X)
        return X
