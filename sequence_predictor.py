import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima_process import arma_generate_sample

NUM_EPOCHS = 50

SEQ_SIZE = 1
INPUT_SIZE = 1
BATCH_SIZE = 10
OUTPUT_DIM = 1
NUM_LAYERS = 2

gradients_norm = 5
train_size = 5000
test_size = 100


class DataMaker:
    def __init__(self):
        arparams = np.array([0.6, -0.5, -0.2], dtype=np.double)
        maparams = np.array([], dtype=np.double)
        noise = 0.1

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


def get_batches(X, y, batch_size, seq_size):
    num_batches = np.prod(X.shape) // (seq_size * batch_size)
    for i in range(0, num_batches):
        if y.shape == (1, 9, 1):
            breakpoint()
        yield X[:, i:i + batch_size], y[:, i:i + batch_size]


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


def get_loss_and_train_op(net, lr=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def train(X, y, batch_size, seq_size, device, net):
    criterion, optimizer = get_loss_and_train_op(net, 0.01)
    iteration = 0

    for epoch in range(NUM_EPOCHS):
        batches = get_batches(X, y, batch_size, seq_size)
        net.init_hidden()

        for x, y in batches:
            iteration += 1

            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits = net(x)
            loss = criterion(logits, y)

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()

            # Update the network's parameters
            optimizer.step()

            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)

            optimizer.step()

            if iteration:
                print('Epoch: {}/{}'.format(epoch, NUM_EPOCHS),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

    return net


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.init_hidden()
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words))


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    data_maker = DataMaker()

    X_train, y_train = data_maker.get_train_data()
    X_test, y_test = data_maker.get_test_data()

    net = LSTM(input_dim=INPUT_SIZE, hidden_dim=SEQ_SIZE, batch_size=BATCH_SIZE, output_dim=OUTPUT_DIM,
               num_layers=NUM_LAYERS)

    net = net.to(device)

    net = train(X=X_train, y=y_train, device=device, net=net, batch_size=BATCH_SIZE, seq_size=SEQ_SIZE)

    # print("Generating Example...")
    # predict(device, net, initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5)


if __name__ == '__main__':
    main()
