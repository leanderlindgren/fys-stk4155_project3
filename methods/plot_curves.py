import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 1001)

tol = 1e-3

def forward(x):
    return -np.log(1-(x-tol))

def inverse(x):
    return 1 + tol - np.exp(-(x))

data = np.load("neural_network/history_split.npz")
accs = data["accs"]

fig, ax = plt.subplots()
ax.plot(epochs, accs.mean(axis=0), label=f"Neural Network", c="#0011FF")
ax.plot(epochs, np.clip(accs.mean(axis=0) + accs.std(axis=0), 0, 1), c="#0077FF")
ax.plot(epochs, np.clip(accs.mean(axis=0) - accs.std(axis=0), 0, 1), c="#0077FF")

data = np.load("convolutional_neural_network/history_split.npz")
accs = data["accs"]

ax.plot(epochs, accs.mean(axis=0), label=f"Convolutional Neural Network", c="#EE0004")
ax.plot(epochs, np.clip(accs.mean(axis=0) + accs.std(axis=0), 0, 1), c="#EE6666")
ax.plot(epochs, np.clip(accs.mean(axis=0) - accs.std(axis=0), 0, 1), c="#EE6666")

ax.set_xlabel("Epoch")
# ax.set_ylim(0.5, 0.62)
ax.set_ylabel("Accuracy")
ax.set_yscale("function", functions=(forward, inverse))
plt.legend()
plt.show()