import itertools
import random
from typing import List

import numpy as np
import torch
from torch.nn import Module, LSTM, Linear, MSELoss, Sigmoid
from torch.optim import Adam

from utils import get_torch_device, plot_losses, plot_accuracy_per_bit

LEARNING_RATE = 0.01

INPUT_DIM = 2
HIDDEN_DIM = 2
OUTPUT_DIM = 1
BATCH_SIZE = 10
NUM_LAYERS = 2

NUM_EPOCHS = 20
BATCH_COUNT = 1000


class AdderDataset:
    def __init__(self, min_seq_length=3, max_seq_length=20, train_batch_size=20, test_batch_size=5, batch_count=10,device='cpu'):
        self.seq_sizes = np.random.randint(min_seq_length, max_seq_length, batch_count)

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.device = device
        for seq_size in self.seq_sizes:
            X_train, y_train, X_test, y_test = self.make_batch(seq_size=seq_size,
                                                               train_batch_size=train_batch_size,
                                                               test_batch_size=test_batch_size)
            self.X_train += [X_train]
            self.X_test += [X_test]
            self.y_train += [y_train]
            self.y_test += [y_test]

    def make_batch(self, seq_size, train_batch_size, test_batch_size):
        binarrays = [x[::-1] for x in itertools.product([0], *itertools.repeat([0, 1], seq_size - 1))]
        train_data = [random.choices(binarrays, k=2) for _ in range(train_batch_size)]
        test_data = [x for _ in range(test_batch_size) if (x := random.choices(binarrays, k=2)) not in train_data]

        while len(test_data) < test_batch_size:
            test_data = [x for _ in range(test_batch_size) if (x := random.choices(binarrays, k=2)) not in train_data]

        return train_data, (
            self.get_binarray_product_with_sum(train_data)), test_data, self.get_binarray_product_with_sum(test_data)

    def get_train_batches(self):
        for batch in range(len(self.X_train)):
            yield batch, self.normalize_data(self.X_train[batch], self.device), self.normalize_data(self.y_train[batch], self.device)

    def get_test_batches(self):
        for batch in range(len(self.X_train)):
            yield batch, self.normalize_data(self.X_test[batch],self.device), self.normalize_data(self.y_test[batch],self.device)

    @classmethod
    def get_binarray_product_with_sum(cls, binarrays_product):
        return [[cls.binarray_sum(x, y)] for x, y in binarrays_product]

    @staticmethod
    def normalize_data(data, device):
        return torch.from_numpy(np.array(data)).transpose(0, 2).transpose(1, 2).float().to(device)

    @staticmethod
    def int_to_binarray(integer: int, seq_size: int):
        return [int(c) for c in f'{integer:0{seq_size}b}'][::-1]

    @staticmethod
    def binarray_to_int(binarray: List[int]):
        return int(''.join([str(x) for x in binarray[::-1]]), base=2)

    @classmethod
    def binarray_sum(cls, a, b):
        return cls.int_to_binarray(cls.binarray_to_int(a) + cls.binarray_to_int(b), seq_size=len(a))


def get_loss_and_train_op(net, lr):
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
        self.lstm.to('cpu')

    def forward(self, data):
        lstm_out, hidden = self.lstm(data)
        fc_output = self.fc(lstm_out)
        return self.sigmoid(fc_output)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_dim, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_dim, self.hidden_dim))

    def evaluate(model, X_test, y_test):
        return np.average(bitwise_accuracy(model(X_test), y_test))


def train(model, dataset: AdderDataset):
    model.train()

    criterion, optimizer = get_loss_and_train_op(model, LEARNING_RATE)
    training_losses = []
    test_losses = []

    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        for batch, X_batch, y_batch in dataset.get_train_batches():
            # Forward pass
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            if batch % 100 == 0:
                print('Epoch: {}/{}'.format(epoch + 1, NUM_EPOCHS),
                      'Batch: {}'.format(batch),
                      'Loss: {:.20f}'.format(loss.item()))
            running_loss += loss.item() / BATCH_COUNT
        training_losses += [running_loss]
        test_losses += [
            sum(criterion(model(X_test), y_test) for batch, X_test, y_test in dataset.get_test_batches()) / BATCH_COUNT]

    return model, training_losses, test_losses


def predict(model, X_test):
    model.evaluate()

    state_h, state_c = model.init_hidden()

    return model(X_test).round().int()


def evaluate(model, dataset):
    dict_per_bit = {i: [] for i in range(20)}

    for batch, x_test, y_test in dataset.get_test_batches():
        y_pred = model(x_test)
        for i in range(len(y_pred)):
            dict_per_bit[i] += [np.average(bitwise_accuracy(y_pred[i], y_test[i]))]

    dict_acc_per_bit = {x: np.average(dict_per_bit[x]) for x in dict_per_bit if len(dict_per_bit[x]) > 0}
    plot_accuracy_per_bit(dict_acc_per_bit)

    return np.average([float(x) for x in dict_acc_per_bit.values()])


def bitwise_accuracy(y_pred: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
    return torch.sum(y_test == torch.round(y_pred), dim=1) / float(y_test.size(1))


def main():
    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)

    adder_dataset = AdderDataset(batch_count=BATCH_COUNT, train_batch_size=BATCH_SIZE, test_batch_size=3, device=device)

    rnn = RNNAdder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)
    model, training_losses, test_losses = train(model=rnn, dataset=adder_dataset)

    #plot_losses(title='LSTM Adder', train_loss=training_losses, test_loss=test_losses, epochs=range(NUM_EPOCHS))
    #print(evaluate(model, dataset=adder_dataset))

    adder_dataset1 = AdderDataset(batch_count=1000, train_batch_size=0, test_batch_size=3, device=device)
    print(evaluate(model, dataset=adder_dataset1))


if __name__ == '__main__':
    main()
