from ..data.dataset_sampling import resample_dataset_frame
from ..data.preprocessing import load_participant_data
from ..data.ml_datasets import SlidingWindowDataset
from .models import LSTMModel, FeedForwardModel

import torch
import time
from torch.utils.data import DataLoader, random_split


def train_model(
        window_size=100,
        train_ratio=0.9,
        learning_rate=0.1,
        num_epochs=10,
):
    print("Train a new model")

    # load data
    df_labels, df_leg, df_bike = load_participant_data(participant_id='01')
    df = resample_dataset_frame(df_labels, df_leg, df_bike)
    dataset = SlidingWindowDataset(participants_data=[df], window_size=window_size, window_label='end', slide_step_size=10)

    # create train and eval splits
    # TODO: for now this is not that useful, as both datasets are essentially the same and windows even overlap
    train_size = int(train_ratio * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # create model
    sequence_length, input_size = train_dataset.__getitem__(0)['data'].size()

    #model = LSTMModel(
    #    input_size=input_size,
    #    hidden_size=32,
    #    num_layers=2,
    #    sequence_length=window_size,
    #)

    model = FeedForwardModel(
        sequence_length=sequence_length,
        input_size=input_size,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    stats = {
        'train_loss': [],
        'eval_loss': [],
        'epoch_duration': [],
    }

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch} ---")
        model.train()
        start_time = time.time()

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

        train_loss = 0.0

        for batch in train_loader:
            model.zero_grad()
            output = model.forward(inputs=batch['data'], labels=batch['label'])
            loss = output['loss']
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataset)

        # do evaluation for this epoch
        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for batch in eval_loader:
                output = model.forward(inputs=batch['data'], labels=batch['label'])
                eval_loss += output['loss'].item()

            if epoch == num_epochs - 1:
                n = 0
                for batch in eval_loader:
                    n += 1
                    if n > 3:
                        break
                    output = model.forward(inputs=batch['data'], labels=batch['label'])
                    # print outputs and labels for inspection
                    print('predictions:', output['prediction'])
                    print('labels:', batch['label'])

            eval_loss /= len(eval_dataset)

        # collect statistics
        duration = time.time() - start_time
        stats['train_loss'].append(train_loss)
        stats['eval_loss'].append(eval_loss)
        stats['epoch_duration'].append(duration)
        print(f"train loss: {train_loss}")
        print(f"eval loss: {eval_loss}")
        print(f"epoch duration: {round(duration, 3)} seconds")




