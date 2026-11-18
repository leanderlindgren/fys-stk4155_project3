import numpy as np
import sys
sys.path.append("../")
from dataset.tictactoe import TicTacToe
from models.neural_network_classifier import NeuralNetworkClassifier
from functions.activation_functions import sigmoid, sigmoid_der, softmax, ReLU, ReLU_der
from functions.cost_functions import mse, mse_der
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

rng = np.random.default_rng(69)
torch.manual_seed(69)

epochs = 1000
batch = 100
lr = 0.03

class MaskedConvolutionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = torch.tensor([
            [[True, True, True],
            [False, False, False],
            [False, False, False]],

            [[False, False, False],
            [True, True, True],
            [False, False, False]],

            [[False, False, False],
            [False, False, False],
            [True, True, True]],

            [[True, False, False],
            [True, False, False],
            [True, False, False]],

            [[False, True, False],
            [False, True, False],
            [False, True, False]],

            [[False, False, True],
            [False, False, True],
            [False, False, True]],

            [[True, False, False],
            [False, True, False],
            [False, False, True]],

            [[False, False, True],
            [False, True, False],
            [True, False, False]],
        ])
        self.conv = nn.Conv1d(1, 1, 3)

    def forward(self, x):
        x_conv = torch.zeros(x.shape[0], 17)
        for i, mask in enumerate(torch.unbind(self.masks, dim=0)):
            ms = torch.masked_select(x.unsqueeze(0), mask).reshape(-1, 1, 3)
            x_conv[:, i] = self.conv(ms)[:, 0, 0]
        x_conv[:, 8:] = x.reshape(-1, 9)
        return x_conv

data = np.load("tablebase.npz")
X = data["positions"]
y = data["evaluations"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)

Xtr = torch.tensor(X_train.reshape(-1, 1, 3, 3).astype("float32"))
Ytr = torch.tensor(y_train+1).type(torch.long)
Xte = torch.tensor(X_test.reshape(-1, 1, 3, 3).astype("float32"))
Yte = torch.tensor(y_test+1).type(torch.long)

trunk = [MaskedConvolutionLayer()]
trunk = [nn.Conv2d(1, 9, 3, padding=1), nn.MaxPool2d(3)]
head = [nn.Flatten(), nn.Linear(9, 100), nn.Sigmoid(), nn.Sigmoid(), nn.Linear(100, 3)]
# trunk = []
# head = [nn.Flatten(), nn.Linear(9, 50), nn.Sigmoid(), nn.Sigmoid(), nn.Linear(50, 3)]

net = nn.Sequential(*(trunk + head))
opt = torch.optim.Adam(net.parameters(), lr=lr)
lossf = nn.CrossEntropyLoss()
hist = []
for ep in range(1, epochs+1):
    net.train()
    perm = torch.randperm(len(Xtr))
    run_loss, nb = 0.0, 0
    for i in range(0, len(Xtr) - batch + 1, batch):
        idx = perm[i:i + batch]
        opt.zero_grad(set_to_none=True)
        loss = lossf(net(Xtr[idx]), Ytr[idx])
        loss.backward()
        opt.step()
        run_loss += loss.item(); nb += 1
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte[i:i + 1000]).argmax(1) for i in range(0, len(Xte), 1000)])
        acc = (pred == Yte).double().mean().item()
    hist.append((ep, run_loss/nb, acc))
    print(f" epoch {ep}: loss {run_loss / nb:.4f}  test accuracy {acc:.4f})")
print(np.array(hist))


print(f"Accuracy: {net.accuracy(Xte.reshape(-1, 3, 3), y_test)}")