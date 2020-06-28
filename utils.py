import torch
import matplotlib.pyplot as plt


def get_torch_device():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return 'cuda'
    else:
        return 'cpu'


def plot_losses(epochs, train_loss, test_loss, title=''):
    plt.style.use("ggplot")
    plt.plot(epochs, train_loss, 'r', label="Training Loss")
    plt.plot(epochs, test_loss, 'b', label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="center")
    plt.title(title)
    plt.show()


def plot_accuracy_per_bit(dict_acc_per_bit):
    plt.style.use("ggplot")
    plt.plot([int(x) for x in dict_acc_per_bit.keys()],
             [float(x) for x in dict_acc_per_bit.values()],
             'b', label="Accuracy per bit")
    plt.xlabel("bits")
    plt.ylabel("accuracy")
    plt.legend(loc="upper left")
    plt.title("dict_acc_per_bit per bits")
    plt.show()