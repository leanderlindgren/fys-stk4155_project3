import numpy as np
import sys
sys.path.append("../")
from dataset.tictactoe import TicTacToe

game = TicTacToe()

position_evals = {}
seen = set()
queue = []
minmax = [[]]

while len(position_evals) < 5478:
    moves = game.legalmoves

    for move in moves[len(minmax[-1]):]:
        game.makeMove(move)
        if tuple(game.position.tolist()) not in seen:
            seen.add(tuple(game.position.tolist()))
            # If game over
            if game.gameover:
                position_evals[tuple(game.position.tolist())] = game.winner
                minmax[-1].append(game.winner)
                game.undo()
                continue

            # Now we go deeper
            minmax.append([])
            break

        # If reached evaluated position already
        elif tuple(game.position.tolist()) in position_evals.keys():
            minmax[-1].append(position_evals[tuple(game.position.tolist())])
            game.undo()
            continue

    if len(minmax[-1]) == 0:
        continue

    match game.toplay:
        case 1:
            position_evals[tuple(game.position.tolist())] = int(np.max(minmax[-1]))
        case -1:
            position_evals[tuple(game.position.tolist())] = int(np.min(minmax[-1]))

    game.undo()
    minmax.pop()

keys = position_evals.keys()

X = np.array(list(keys))

n, _ = X.shape
y = np.empty(n)

for i, key in enumerate(keys):
    y[i] = position_evals[key]

np.savez_compressed("tablebase.npz", positions=X, evaluations=y)