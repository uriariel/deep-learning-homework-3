import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima_process import arma_generate_sample

from utils import get_torch_device, plot_losses

LEARNING_RATE = 0.00001
NUM_EPOCHS = 50

SEQ_SIZE = 10
INPUT_SIZE = 1
BATCH_SIZE = 10
OUTPUT_DIM = 1
NUM_LAYERS = 2

gradients_norm = 5
train_size = 2000
test_size = 500


class DataMaker:
    def __init__(self, noise):
        arparams = np.array([0.6, -0.5, -0.2], dtype=np.double)
        maparams = np.array([], dtype=np.double)
        noise = noise

        # Generate train and test sequences
        ar = np.r_[1, -arparams]  # add zero-lag and negate
        ma = np.r_[1, maparams]  # add zero-lag

        np.random.seed(1000)

        self.batch_size = BATCH_SIZE
        self.seq_size = SEQ_SIZE
        self.train_data = arma_generate_sample(ar=ar, ma=ma, nsample=train_size, scale=noise, distrvs=np.random.uniform)
        self.test_data = arma_generate_sample(ar=ar, ma=ma, nsample=test_size, scale=noise, distrvs=np.random.uniform)

    def get_train_data(self):
        return self.prepare_data(self.train_data, self.batch_size, self.seq_size)

    def get_test_data(self):
        return self.prepare_data(self.test_data, self.batch_size, self.seq_size)

    @staticmethod
    def prepare_data(data, batch_size, seq_size):
        data = torch.from_numpy(data)
        num_batches = int(len(data) / (seq_size * batch_size))
        X = data[:num_batches * batch_size * seq_size]

        y = torch.from_numpy(np.zeros_like(X))
        y[:-1] = X[1:]
        y[-1] = X[0]

        X = X.view(SEQ_SIZE, -1, INPUT_SIZE).float()
        y = y.view(SEQ_SIZE, -1, INPUT_SIZE).float()

        return X, y


def get_batches(X, y, batch_size):
    num_batches = X.size(1) // batch_size
    for i in range(0, num_batches):
        yield i, X[:, i:i + batch_size], y[:, i:i + batch_size]


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden = None

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input)
        y_pred = self.fc(lstm_out)
        return y_pred


def get_loss_and_train_op(net, lr):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def train(model, X_train, y_train, X_test, y_test, batch_size):
    train_losses = []
    test_losses = []
    criterion, optimizer = get_loss_and_train_op(model, LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        model.init_hidden()

        num_batches = X_train.size(1) // batch_size

        for iteration, X_batch, y_batch in get_batches(X_train, y_train, batch_size):

            # Tell it we are in training mode
            model.train()

            # Reset all gradients
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()

            # Update the network's parameters
            optimizer.step()

            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), gradients_norm)

            optimizer.step()

            running_loss += loss.item() / num_batches

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(epoch + 1, NUM_EPOCHS),
                      'Iteration: {}'.format(iteration),
                      'Loss: {:.20f}'.format(loss_value))

        train_losses += [running_loss]
        test_losses += [sum([criterion(model(X_test), y_test).item()])]
    return model, train_losses, test_losses


def predict(net, X_test):
    net.evaluate()

    state_h, state_c = net.init_hidden()

    return net(X_test, (state_h, state_c))


def evaluate(model, X_test, y_test):
    return np.average(model(X_test) == y_test)


def main():
    criterion = torch.nn.MSELoss()

    noise_values = torch.linspace(0.1, 2.5, 10)
    noise_loss = []
    for noise in noise_values:
        data_maker = DataMaker(noise=noise)

        X_train, y_train = data_maker.get_train_data()
        X_test, y_test = data_maker.get_test_data()

        model = LSTM(input_dim=INPUT_SIZE, hidden_dim=SEQ_SIZE, batch_size=BATCH_SIZE, output_dim=OUTPUT_DIM,
                     num_layers=NUM_LAYERS)

        model, training_losses, test_losses = train(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                    model=model, batch_size=BATCH_SIZE)

        plot_losses(title='Sequence LSTM noise = {}'.format(round(noise.item(), 3)), train_loss=training_losses, test_loss=test_losses, epochs=range(NUM_EPOCHS))

        noise_loss += [criterion(model(X_test), y_test)]

    plt.style.use("ggplot")
    plt.plot(noise_values,noise_loss, 'b', label="Test Loss")
    plt.xlabel("Noise")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.title('Sequence LSTM - noises')
    plt.show()


if __name__ == '__main__':
    main()
