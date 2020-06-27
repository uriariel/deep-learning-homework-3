import torch
import matplotlib.pyplot as plt


def get_torch_device():
    return 'cpu'
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
