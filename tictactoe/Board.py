class Board:

    def __init__(self):
        self.amount_of_fields = 9
        self.fields_player_0 = [0] * self.amount_of_fields
        self.fields_player_1 = [0] * self.amount_of_fields

    def add_move(self, player_number: int, move: int) -> (int, bool):
        game_not_finished = True
        game_feedback = 0
        if player_number == 0:
            if self.fields_player_0[move] == 0:
                self.fields_player_0[move] = 1
            check_fields = self.fields_player_0
        else:
            if self.fields_player_1[move] == 0:
                self.fields_player_1[move] = 1
            check_fields = self.fields_player_1
        if (check_fields[0] == 1 and check_fields[1] == 1 and check_fields[2] == 1) or (
                check_fields[3] == 1 and check_fields[4] == 1 and check_fields[5] == 1) or (
                check_fields[6] == 1 and check_fields[7] == 1 and check_fields[8] == 1) or (
                check_fields[0] == 1 and check_fields[3] == 1 and check_fields[6] == 1) or (
                check_fields[1] == 1 and check_fields[4] == 1 and check_fields[7] == 1) or (
                check_fields[2] == 1 and check_fields[5] == 1 and check_fields[8] == 1) or (
                check_fields[0] == 1 and check_fields[4] == 1 and check_fields[8] == 1) or (
                check_fields[2] == 1 and check_fields[4] == 1 and check_fields[6] == 1):
            return 1, False
        if sum(self.get_allow()) == 0:
            game_not_finished = False
            if player_number == 0:
                game_feedback = -0.0
            else:
                game_feedback = 0.0
        return game_feedback, game_not_finished

    def get_input(self, player_number: int) -> list:
        input_vector = []
        if player_number == 0:
            input_vector.extend(self.fields_player_0)
            input_vector.extend(self.fields_player_1)
        else:
            input_vector.extend(self.fields_player_1)
            input_vector.extend(self.fields_player_0)
        input_vector.extend(self.get_allow())
        return input_vector

    def get_allow(self) -> list:
        allow_vector = [1] * self.amount_of_fields
        for field_number in range(0, self.amount_of_fields):
            if self.fields_player_0[field_number] == 1 or self.fields_player_1[field_number] == 1:
                allow_vector[field_number] = 0
        return allow_vector
