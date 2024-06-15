import torch
import random
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(self, participants_data, window_size, window_label):
        """Takes a list if individual participants dataframes and build a sliding window dataset from them."""
        self.window_size = window_size

        extracted_data = []
        extracted_labels = []

        for df in participants_data:
            # TODO: decide on nan strategy
            df = df.fillna(0)

            df = df.loc[:, ~df.columns.get_level_values(1).str.contains('^participant_id')]
            df = df.reset_index(drop=True)
            raw_data = torch.tensor(df.drop(columns=[('df_labels', 'label')]).values, dtype=torch.float)

            labels = torch.tensor(df[('df_labels', 'label')].values)
            # prepare labels such that the correct label for window i is at position i of the label tensor
            if window_label == 'end':
                labels = labels[window_size - 1:]
            elif window_label == 'middle':
                label_offset = int(window_size / 2)
                labels = labels[label_offset: labels.size()[0] - (window_size - label_offset)]
            else:
                raise ValueError(f'window_label positions {window_label} has not been implemented yet')

            def extract_item(idx):
                return raw_data[idx:idx + self.window_size], labels[idx]

            for i in range(raw_data.size()[0] - self.window_size):
                data, label = extract_item(i)
                extracted_data.append(data)
                extracted_labels.append(label.item())

        self.data = torch.stack(extracted_data)
        self.labels = torch.tensor(extracted_labels, dtype=torch.float)

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, i):
        return {'data': self.data[i], 'label': self.labels[i]}


