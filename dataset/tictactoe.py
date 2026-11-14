import numpy as np

class TicTacToe:
    def __init__(self):
        self.position_history = [
            np.array([0 for _ in range(9)]).astype(int)
        ]
        self.movehistory = []

    @property
    def position(self):
        return self.position_history[-1]

    @property
    def gameover(self):
        winner = self.winner
        if winner:
            return True

        elif np.sum(abs(self.position)) == 9:
            return True

        else:
            return False

    @property
    def winner(self):
        position = self.position_history[-1]

        # Rows
        for i in range(3):
            match np.sum(position[3*i:3*(i+1)]):
                case 3:
                    return 1

                case -3:
                    return -1

                case _:
                    continue

        # Columns
        for j in range(3):
            match np.sum(position[j::3]):
                case 3:
                    return 1

                case -3:
                    return -1

                case _:
                    continue

        # Diagonals
        for k in [0, 2]:
            # UIOGPT (GPT-5.6 Terra) helped identify indexing issue here.
            # It identified that with [k::4-k], we get (2, 4, 6, 8) for k=2
            # instead of (2, 4, 6), where the latter is desired. The fix was
            # entirely written by me.
            match np.sum(position[k:9-k:4-k]):
                case 3:
                    return 1

                case -3:
                    return -1

                case _:
                    continue

        return 0

    @property
    def toplay(self):
        match np.sum(self.position_history[-1]):
            case 1:
                return -1

            case 0:
                return 1

            case _:
                raise ValueError(f"Illegal position {self.position_history[-1]}")

    @property
    def outcome(self):
        return self.winner

    @property
    def legalmoves(self):
        return np.arange(9).astype(int)[np.where(self.position == 0)]

    def __str__(self):
        return str(self.position.reshape(3, 3))

    def __call__(self, move):
        self.makeMove(move)
        if self.show:
            print("###")
            print(self)
            if self.winner != "-":
                print(f"{self.winner} wins!")
            elif self.draw:
                print(f"Draw!")

    def undo(self):
        if len(self.position_history) > 1:
            self.position_history.pop()

    def makeMove(self, move):
        if move not in self.legalmoves:
            raise ValueError(f"Illegal move {move}")

        self.position_history.append(self.position.copy())
        self.position[move] = self.toplay

    def play(self):
        while self.winner == "-" and not self.draw:
            move = input("Type your move: ")
            self(move)
            print(str(self))

if __name__ == "__main__":
    game = TicTacToe()
    print(str(game))
    print(game.toplay)
    print(game.legalmoves)
    print(game.gameover)
    print(game.winner)
    game.makeMove(4)
    print(str(game))
    print(game.toplay)
    print(game.legalmoves)
    print(game.gameover)
    print(game.winner)
    game.makeMove(1)
    print(str(game))
    print(game.toplay)
    print(game.legalmoves)
    print(game.gameover)
    print(game.winner)
    game.makeMove(2)
    print(str(game))
    print(game.toplay)
    print(game.legalmoves)
    print(game.gameover)
    print(game.winner)
    game.makeMove(0)
    print(str(game))
    print(game.toplay)
    print(game.legalmoves)
    print(game.gameover)
    print(game.winner)
    game.makeMove(6)
    print(str(game))
    print(game.toplay)
    print(game.legalmoves)
    print(game.gameover)
    print(game.winner)