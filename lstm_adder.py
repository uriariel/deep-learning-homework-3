import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import Module, LSTM, Linear, MSELoss, CrossEntropyLoss, Sigmoid
from torch.optim import Adam

from data_maker import get_data

INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1
SEQ_SIZE = 9
BATCH_SIZE = 5
NUM_LAYERS = 2

gradients_norm = 5


def get_loss_and_train_op(net, lr=0.001):
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=lr)

    return criterion, optimizer


class RNNAdder(Module):
    def __init__(self, input_dim, hidden_dim, batch_dim=1, output_dim=1, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_dim = batch_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.fc = Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = Sigmoid()

    def forward(self, data):
        lstm_out, hidden = self.lstm(data)
        fc_output = self.fc(lstm_out)
        return self.sigmoid(fc_output)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_dim, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_dim, self.hidden_dim))


def get_batches(X, y, batch_size):
    num_batches = X.size(1) // batch_size
    for i in range(0, num_batches, batch_size):
        yield i, X[:, i:i + batch_size], y[:, i:i + batch_size]


def train(X, y, device, model):
    model.train()

    criterion, optimizer = get_loss_and_train_op(model, 0.01)
    hist = np.zeros(10000000)

    for batch, X_batch, y_batch in get_batches(X, y, batch_size=BATCH_SIZE):
        # Forward pass
        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)

        if batch % 100 == 0:
            print("Epoch ", batch, "MSE: ", loss.item())
        hist[batch] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

    return model, hist


def normalize_data(data):
    return torch.from_numpy(data).transpose(0, 2).transpose(1, 2).float()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data = get_data(256)
    X_train, X_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2:])

    rnn = RNNAdder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)
    model = train(X=normalize_data(X_train), y=normalize_data(y_train), model=rnn, device='cpu')


if __name__ == '__main__':
    main()
