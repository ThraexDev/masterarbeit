import random

from nimmt.Board import Board


class TestPlayer:
    def __init__(self, player_number: int):
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        return [1, 0, 0], board.get_input(self.player_number)

    def get_player_number(self) -> int:
        return self.player_number
