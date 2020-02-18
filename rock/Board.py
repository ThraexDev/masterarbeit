

class Board:

    def __init__(self):
        self.player_0_picks = []
        self.player_1_picks = []
        self.player_0_wins = 0
        self.player_1_wins = 0

    def add_move(self, selected_moves: list):
        self.player_0_picks.insert(0, selected_moves[0])
        self.player_1_picks.insert(0, selected_moves[1])
        if (selected_moves[0] + 1) % 3 == selected_moves[1]:
            self.player_1_wins += 1
        if (selected_moves[1] + 1) % 3 == selected_moves[0]:
            self.player_0_wins += 1

    def get_input(self, player_number: int) -> list:
        input_vector = [0] * 300
        if player_number == 0:
            for input_index in range(0, len(self.player_0_picks)):
                input_vector[(input_index * 3) + self.player_0_picks[input_index]] = 1
        if player_number == 1:
            for input_index in range(0, len(self.player_1_picks)):
                input_vector[(input_index * 3) + self.player_1_picks[input_index]] = 1
        return input_vector

    def get_feedback_for_player(self, player_number: int) -> int:
        if player_number == 0:
            return self.player_0_wins - self.player_1_wins
        if player_number == 1:
            return self.player_1_wins - self.player_0_wins

    def get_correct_move_for(self, player_number: int) -> int:
        vector = [0] * 3
        if player_number == 0:
            vector[(self.player_1_picks[0] + 1) % 3] = 1
        if player_number == 1:
            vector[(self.player_0_picks[0] + 1) % 3] = 1
        return vector
