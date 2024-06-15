import random
import pandas as pd
from datetime import timedelta
from datasets import Dataset
from matplotlib import pyplot


def resample_dataset_frame(df_labels, df_leg, df_bike):
    """Returns a combined dataframe with data randomly sampled and interpolated from the given dataframes."""

    # resample all data to a frequency of 10 points per second (one data point each 100 milliseconds)
    df_labels = df_labels.resample('100L').interpolate(method='linear')
    df_leg = df_leg.resample('100L').mean(numeric_only=False)
    # TODO: resampling drops the Gain colum? Probably because of strange -inf. That should be handled in preprocessing
    df_bike = df_bike.resample('100L').mean(numeric_only=False)

    # combine individual dataframes
    df = pd.concat([df_labels, df_leg, df_bike], axis=1, keys=['df_labels', 'df_leg', 'df_bike'])

    # slice of beginning and end, where some of the columns only contain NaN values
    valid_start_indices = df.apply(pd.Series.first_valid_index)
    valid_end_indices = df.apply(pd.Series.last_valid_index)
    start_time = max(valid_start_indices)
    end_time = min(valid_end_indices)
    df = df.loc[start_time:end_time]

    return df


def get_future_and_past_neighbours(df):
    for index, row in df.iterrows():
        if row['time_delta'] < timedelta(0):
            past_neighbour = row
        elif row['time_delta'] > timedelta(0):
            future_neighbour = row
            return future_neighbour, past_neighbour


def sample_random_dataset_frame(df_labels, df_leg, df_bike, n_examples):
    """Returns a combined dataframe with data randomly sampled and interpolated from the given dataframes."""
    # exclude 35 seconds from outer bounds in each direction
    # to allow for sampling of up to 30 second windows i either direction around random timestamps
    max_label_time = df_labels['time'].max() - timedelta(seconds=35)
    min_label_time = df_labels['time'].min() + timedelta(seconds=35)
    interval = max_label_time - min_label_time

    randomly_sampled_times = [min_label_time + random.random() * interval for _ in range(n_examples)]

    data_dictionary = dict()

    # interpolate label data
    for t in randomly_sampled_times:
        # get closest future and past labels for label interpolation
        df_labels['time_delta'] = (df_labels['time'] - t)
        future_neighbour, past_neighbour = get_future_and_past_neighbours(df_labels)
        t_interval = future_neighbour['time'] - past_neighbour['time']
        future_factor = 1 - abs(future_neighbour['time_delta']) / t_interval
        past_factor = 1 - abs(past_neighbour['time_delta']) / t_interval
        # add interpolated label
        data_dictionary[t] = {
            'label': future_factor * future_neighbour['label'] + past_factor * past_neighbour['label']
        }

    # interpolate leg device data
    for t in randomly_sampled_times:
        # get closest future and past labels for label interpolation
        df_leg['time_delta'] = (df_leg['time'] - t)
        future_neighbour, past_neighbour = get_future_and_past_neighbours(df_leg)
        t_interval = future_neighbour['time'] - past_neighbour['time']
        future_factor = 1 - abs(future_neighbour['time_delta']) / t_interval
        past_factor = 1 - abs(past_neighbour['time_delta']) / t_interval
        # add interpolated leg device data
        target_columns = [c for c in df_leg.columns if c not in ['time', 'Unnamed: 13', 'time_delta']]
        for column in target_columns:
            data_dictionary[t][f"leg_{column}"] = future_factor * future_neighbour[column] \
                                                  + past_factor * past_neighbour[column]

    # interpolate bike device data
    for t in randomly_sampled_times:
        # get closest future and past labels for label interpolation
        df_bike['time_delta'] = (df_bike['time'] - t)
        future_neighbour, past_neighbour = get_future_and_past_neighbours(df_bike)
        t_interval = future_neighbour['time'] - past_neighbour['time']
        future_factor = 1 - abs(future_neighbour['time_delta']) / t_interval
        past_factor = 1 - abs(past_neighbour['time_delta']) / t_interval
        # add interpolated leg device data
        target_columns = [c for c in df_bike.columns if c not in ['time', 'Unnamed: 13', 'time_delta']]
        for column in target_columns:
            data_dictionary[t][f"bike_{column}"] = future_factor * future_neighbour[column] \
                                                  + past_factor * past_neighbour[column]


