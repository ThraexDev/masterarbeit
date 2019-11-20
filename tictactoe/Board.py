class Board:

    def __init__(self):
        self.fields_player_0 = [0] * 9
        self.fields_player_1 = [0] * 9

    def add_move(self, player_number: int, move: int) -> (bool, bool):
        if player_number == 0:
            self.fields_player_0[move] = 1
        else:
            self.fields_player_1[move] = 1
