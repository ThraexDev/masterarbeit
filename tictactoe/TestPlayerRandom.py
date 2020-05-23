import random

from tictactoe.Board import Board


class TestPlayerRandom:
    def __init__(self, player_number: int):
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        move = -1
        while move == -1:
            rand = random.randint(0, 8)
            if board.get_allow()[rand] == 1:
                move = rand
                move_vector = [0] * 9
                move_vector[move] = 1
                return move_vector, board.get_input(self.player_number), board.get_allow()

    def get_player_number(self) -> int:
        return self.player_number
