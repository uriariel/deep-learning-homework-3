import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

PLOT = True
LENGTH = 1000
trainLength = 100
testLength = 10


def print_shape(t, title=None):
    if title is not None:
        print("{}:\t".format(title), end="")
    if isinstance(t, np.ndarray):
        print(t.shape)
        return

    if isinstance(t, list):
        t = np.asarray(t)
        print(t.shape)
        return
    print(t.detach().numpy().shape)


arparams = np.array([0.6, -0.5, -0.2])
maparams = np.array([])
noise = 0.1

# Generate train and test sequences
ar = np.r_[1, -arparams]  # add zero-lag and negate
ma = np.r_[1, maparams]  # add zero-lag

np.random.seed(1000)

# randomGenerator = np.random.rand  # ~Uniform[0,1]

trainData = arma_generate_sample(ar=ar, ma=ma, nsample=trainLength, scale=noise, distrvs=np.random.uniform)

testData = arma_generate_sample(ar=ar, ma=ma, nsample=testLength, scale=noise, distrvs=np.random.uniform)

print_shape(trainData, title="Train Data:")
print_shape(testData, title="Test Data:")

if PLOT:
    fig, axs = plt.subplots(2, 1, figsize=(18, 6))
    axs[0].plot(trainData[1:200])
    axs[0].grid(True)
    axs[1].plot(testData[1:200])
    axs[1].grid(True)
    plt.show()
