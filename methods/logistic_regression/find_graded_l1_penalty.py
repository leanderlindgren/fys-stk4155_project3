import numpy as np
import sys
sys.path.append("../../")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from methods.penalty import penalty_grade
import torch
import torch.nn as nn

seed = 69
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

epochs = 1000
batch = 1000
lr = 0.03
n_lambdas = 16
lambdas = np.zeros(n_lambdas)
lambdas[1:] = np.logspace(-7, 0, n_lambdas-1)

data = np.load("../../dataset/tablebase.npz")
X = data["positions"]
y = data["evaluations"]

Xtr = torch.tensor(X.reshape(-1, 1, 3, 3).astype("float32")).cuda()
Ytr = torch.tensor(y+1).type(torch.long).cuda()
Xte = torch.tensor(X.reshape(-1, 1, 3, 3).astype("float32")).cuda()
Yte = torch.tensor(y+1).type(torch.long).cuda()


best_loss = []
best_acc = []

losses = np.empty((n_lambdas, epochs))
accs = np.empty((n_lambdas, epochs))
cms = np.empty((n_lambdas, epochs, 3, 3))

for l, lambda_1 in enumerate(lambdas):
    trunk = []
    head = [nn.Flatten(), nn.Linear(9, 3), nn.Sigmoid()]
    net = nn.Sequential(*(trunk + head)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(1, epochs+1):
        net.train()
        perm = torch.randperm(len(Xtr))
        run_loss, run_loss_raw, nb = 0.0, 0.0, 0
        for i in range(0, len(Xtr) - batch + 1, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr[idx]), Ytr[idx])
            run_loss_raw += loss.item()
            l1 = 0
            for i, p in enumerate(net.parameters()):
                if i not in [0]:
                    continue
                l1 += p.abs().sum()
            loss = lossf(net(Xtr[idx]), Ytr[idx]) + l1*penalty_grade(lambda_1, ep, epochs)
            loss.backward()
            opt.step()
            run_loss += loss.item(); nb += 1
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xte[i:i + 1000]).argmax(1) for i in range(0, len(Xte), 1000)])
            acc = (pred == Yte).double().mean().item()
            cms[l, ep-1] = confusion_matrix(Yte.cpu(), pred.cpu())
            hist.append((ep, run_loss_raw/nb, acc))
            print(f" epoch {ep}: loss {run_loss_raw / nb:.4f}  test accuracy {acc:.4f})")

    hist = np.array(hist)
    loss = hist[:, 1]
    acc = hist[:, 2]
    losses[l] = loss
    accs[l] = acc
    best_loss.append((loss).min().item())
    best_acc.append((acc).max().item())

with open("graded_selection_full.txt", "w") as outfile:
    outfile.write(f"Lambdas: {lambdas}\n")
    outfile.write(f"Loss: {best_loss}\n")
    outfile.write(f"Accuracy: {best_acc}")

np.savez("graded_selection_full.npz", lambdas=lambdas, epochs=epochs, losses=losses, accs=accs, cms=cms)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=70)

Xtr = torch.tensor(X_train.reshape(-1, 1, 3, 3).astype("float32")).cuda()
Ytr = torch.tensor(y_train+1).type(torch.long).cuda()
Xte = torch.tensor(X_val.reshape(-1, 1, 3, 3).astype("float32")).cuda()
Yte = torch.tensor(y_val+1).type(torch.long).cuda()

best_loss = []
best_acc = []

losses = np.empty((n_lambdas, epochs))
accs = np.empty((n_lambdas, epochs))
cms = np.empty((n_lambdas, epochs, 3, 3))

for l, lambda_1 in enumerate(lambdas):
    trunk = []
    head = [nn.Flatten(), nn.Linear(9, 3), nn.Sigmoid()]
    net = nn.Sequential(*(trunk + head)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(1, epochs+1):
        net.train()
        perm = torch.randperm(len(Xtr))
        run_loss, run_loss_raw, nb = 0.0, 0.0, 0
        for i in range(0, len(Xtr) - batch + 1, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr[idx]), Ytr[idx])
            run_loss_raw += loss.item()
            l1 = 0
            for i, p in enumerate(net.parameters()):
                if i not in [0]:
                    continue
                l1 += p.abs().sum()
            loss = lossf(net(Xtr[idx]), Ytr[idx]) + l1*penalty_grade(lambda_1, ep, epochs)
            loss.backward()
            opt.step()
            run_loss += loss.item(); nb += 1
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xte[i:i + 1000]).argmax(1) for i in range(0, len(Xte), 1000)])
            acc = (pred == Yte).double().mean().item()
            cms[l, ep-1] = confusion_matrix(Yte.cpu(), pred.cpu())
            hist.append((ep, run_loss_raw/nb, acc))
            print(f" epoch {ep}: loss {run_loss_raw / nb:.4f}  test accuracy {acc:.4f})")

    hist = np.array(hist)
    loss = hist[:, 1]
    acc = hist[:, 2]
    losses[l] = loss
    accs[l] = acc
    best_loss.append((loss).min().item())
    best_acc.append((acc).max().item())

with open("graded_selection_split.txt", "w") as outfile:
    outfile.write(f"Lambdas: {lambdas}\n")
    outfile.write(f"Loss: {best_loss}\n")
    outfile.write(f"Accuracy: {best_acc}")

np.savez("graded_selection_split.npz", lambdas=lambdas, epochs=epochs, losses=losses, accs=accs, cms=cms)