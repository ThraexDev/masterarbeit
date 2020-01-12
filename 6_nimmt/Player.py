from tictactoe.Board import Board
from tictactoe.MCSTRootNode import MCSTRootNode
import numpy as np


class Player:
    def __init__(self, ai_model, player_number: int):
        self.model = ai_model
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        monte_carlo_search_tree = MCSTRootNode(self.model, board, self.player_number)
        result = monte_carlo_search_tree.get_result()
        allow = board.get_allow()
        return [result[i]*allow[i] for i in range(len(result))], board.get_input(self.player_number), allow

    def get_player_number(self) -> int:
        return self.player_number
