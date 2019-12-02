import random
from copy import deepcopy

from tictactoe.Board import Board


class TestPlayer:
    def __init__(self, player_number: int):
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        for move_number in range(0, 9):
            new_board = deepcopy(board)
            current_player_won, game_not_finished = new_board.add_move(self.player_number, move_number)
            if not game_not_finished and current_player_won:
                move_vector = [0] * 9
                move_vector[move_number] = 1
                return move_vector, board.get_input(self.player_number), board.get_allow()
        other_player = (self.player_number + 1) % 2
        for move_number in range(0, 9):
            new_board = deepcopy(board)
            current_player_won, game_not_finished = new_board.add_move(other_player, move_number)
            if not game_not_finished and current_player_won:
                move_vector = [0] * 9
                move_vector[move_number] = 1
                return move_vector, board.get_input(self.player_number), board.get_allow()
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
