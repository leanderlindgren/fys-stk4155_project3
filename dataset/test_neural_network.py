import numpy as np
import sys
sys.path.append("../")
from dataset.tictactoe import TicTacToe
from models.neural_network_classifier import NeuralNetworkClassifier
from functions.activation_functions import sigmoid, sigmoid_der, softmax, ReLU, ReLU_der
from functions.cost_functions import mse, mse_der
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(69)

net = NeuralNetworkClassifier(
    network_input_size=9,
    layer_output_sizes=[100, 100, 3],
    classes=[1, 0, -1],
    activation_funcs=[ReLU, sigmoid, softmax],
    activation_ders=[ReLU_der, sigmoid_der, None],
    cost_fun=mse,
    cost_der=mse_der,
    batch_size=1000,
    epochs=1000,
    method="adam",
    rng=rng
)

data = np.load("tablebase.npz")
X = data["positions"]
y = data["evaluations"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)

net.fit(X_train, y_train)
print(f"Accuracy: {net.accuracy(X_test, y_test)}")

W1 = net.nn_.layers_[0][0]
b1 = net.nn_.layers_[0][1]
W2 = net.nn_.layers_[1][0]
b2 = net.nn_.layers_[1][1]

np.savez_compressed("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)