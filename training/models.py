import torch


class LSTMModel(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            sequence_length: int,
    ):
        super(LSTMModel, self).__init__()
        self.device = None

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # for now the prediction layer just sees all the hidden states at all time steps equally/simultaneously
        self.prediction_layer = torch.nn.Linear(hidden_size * sequence_length, 1)

    def to(self, device):
        self.device = device
        super().to(device)

    def forward(
            self,
            inputs,
            labels=None,
    ):
        lstm_out, _ = self.lstm(inputs)
        prediction = self.prediction_layer(lstm_out.reshape(lstm_out.size(0), -1))
        output = {'prediction': prediction}

        # for now, I just use absolute distance as loss. Maybe we should look for something better?
        if labels is not None:
            distances = torch.abs(prediction.squeeze() - labels)
            # TODO: probably the result of nan in the df. Should we exclude non inputs or replace them?
            #nan_mask = torch.isnan(distances)
            #loss = distances[~nan_mask].sum()

            loss = distances.sum()
            output['loss'] = loss
        return output

