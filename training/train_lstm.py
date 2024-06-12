from ..data.dataset_sampling import sample_dataset
from ..data.preprocessing import load_participant_data


def train_lstm():
    print("Train an lstm model")

    # load training data
    df_labels, df_leg, df_bike = load_participant_data(participant_id='01')
    training_data = sample_dataset(df_labels, df_leg, df_bike, n_examples=500)
