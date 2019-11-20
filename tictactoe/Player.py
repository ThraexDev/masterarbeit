from tictactoe.MCSTRootNode import MCSTRootNode


class Player:
    def __init__(self, ai_model, player_number):
        self.model = ai_model
        self.player_number = player_number

    def calculate_turn(self, board):
        monte_carlo_search_tree = MCSTRootNode(self.model, board, self.player_number)
        return monte_carlo_search_tree.get_result(), board.get_input(self.player_number), board.get_allow(self.player_number)

