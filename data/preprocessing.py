import time
from datetime import datetime, timedelta
import pandas as pd


def load_participant_data(participant_id):
    """Returns a dataframe holding the recorded data for the given participant."""

    # TODO: we have to decide on a general data format that allows for automated processing
    labels_path = '/home/seb/predicting-travel-experience/raw_data/01/Proband01_label.txt'
    leg_device_path = '/home/seb/predicting-travel-experience/raw_data/01/Proband01_2024-06-09_15.38.10.csv'
    bike_device_path = '/home/seb/predicting-travel-experience/raw_data/01/Proband01_2024-06-09_15.37.41.csv'

    df_labels = pd.read_csv(labels_path, sep=';')[['timestamp', 'label']]
    df_labels = df_labels.drop([0, 1, 2, 3, 4]).reset_index(drop=True, inplace=False)
    # label time synchronisation
    time_offset = datetime.strptime('15:35:08', '%H:%M:%S').time()
    time_delta = timedelta(hours=time_offset.hour, minutes=time_offset.minute, seconds=time_offset.second)
    df_labels['timestamp'] = df_labels['timestamp'].apply(lambda x: datetime.strptime(x, '%M:%S') + time_delta)
    df_labels.rename(columns={'timestamp': 'time'}, inplace=True)

    df_leg = pd.read_csv(leg_device_path)
    df_leg['time'] = df_leg['time'].apply(lambda x: datetime.strptime(x, '%H:%M:%S:%f'))

    df_bike = pd.read_csv(bike_device_path)
    df_bike['time'] = df_bike['time'].apply(lambda x: datetime.strptime(x, '%H:%M:%S:%f'))

    return df_labels, df_leg, df_bike
