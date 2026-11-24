import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 1001)

data = np.load("convolutional_neural_network/ungraded_selection_full.npz")
accs = data["accs"]
lambdas = data["lambdas"]
n_lambdas = lambdas.shape[0]

fig, ax = plt.subplots()
for l in range(n_lambdas):
    ax.plot(epochs, accs[l], label=f"$\\lambda = {lambdas[l]}$")

ax.set_xlabel("Epoch")
# ax.set_ylim(0.5, 0.62)
ax.set_ylabel("Accuracy")
plt.legend()
plt.show()