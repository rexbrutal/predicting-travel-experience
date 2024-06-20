import torch


class FeedForwardModel(torch.nn.Module):
    def __init__(
            self,
            sequence_length: int,
            input_size: int,
    ):
        super(FeedForwardModel, self).__init__()
        self.device = None
        self.fc1 = torch.nn.Linear(sequence_length * input_size, 50)
        self.fc2 = torch.nn.Linear(50, 1)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.MSELoss()

    def to(self, device):
        self.device = device
        super().to(device)

    def forward(
            self,
            inputs,
            labels=None,
    ):
        # skip the batch dimension 0 for flattening
        out1 = self.relu(self.fc1(torch.flatten(inputs, start_dim=1)))
        #print('out1:', out1)
        prediction = self.fc2(out1)
        #print('DEBUG prediction:', prediction)
        output = {'prediction': prediction}
        if labels is not None:
            loss = self.loss_fn(prediction.squeeze(), labels)
            output['loss'] = loss
        return output


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
        self.loss_fn = torch.nn.MSELoss()

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
            loss = self.loss_fn(prediction.squeeze(), labels)
            output['loss'] = loss
        return output

