import numpy as np

from rock.Board import Board


class Player:
    def __init__(self, ai_model, player_number: int):
        self.model = ai_model
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        prediction = self.model.predict_on_batch({'input': np.array([board.get_input(self.player_number)])})
        return prediction[0].numpy(), board.get_input(self.player_number)

    def get_player_number(self) -> int:
        return self.player_number
