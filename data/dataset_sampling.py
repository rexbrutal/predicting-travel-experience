import random
from datetime import timedelta
from datasets import Dataset


def sample_dataset(df_labels, df_leg, df_bike, n_examples):
    """Returns a combined dataframe with data randomly sampled and interpolated from the given dataframes."""
    # exclude 35 seconds from outer bounds in each direction
    # to allow for sampling of up to 30 second windows i either direction around random timestamps
    max_label_time = df_labels['time'].max() - timedelta(seconds=35)
    min_label_time = df_labels['time'].min() + timedelta(seconds=35)
    interval = max_label_time - min_label_time

    sampled_times = [min_label_time + random.random() * interval for _ in range(n_examples)]

    print(max_label_time)
    print(min_label_time)
    print(interval)
    print("random time:", min_label_time + random.random() * interval)
    print(sampled_times)
    exit()
