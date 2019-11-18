class Player:
    def __init__(self, ai_model):
        self.model = ai_model

    def calculate_turn(self, board):
        monte_carlo_search_tree = MCST(board)

