import random

from nimmt.Board import Board


class TestPlayer:
    def __init__(self, player_number: int):
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        indices = [i for i, x in enumerate(board.playercards[self.player_number]) if x == 1]
        randomcard = random.randint(0, len(indices) - 1)
        selectedcard = indices[randomcard]
        selectedbatch = random.randint(0, board.amountofbatches - 1)
        move_vector = [0] * board.cardamount * board.amountofbatches
        move_vector[selectedcard * selectedbatch] = 1
        return move_vector, board.get_input(self.player_number), board.get_allow(self.player_number)

    def get_player_number(self) -> int:
        return self.player_number
