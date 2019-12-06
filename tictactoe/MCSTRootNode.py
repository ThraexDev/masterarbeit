from tictactoe.MCSTNode import MCSTNode


class MCSTRootNode:

    def __init__(self, model, board, player_number):
        self.root_node = MCSTNode(0, model, board, player_number, True)

    def get_result(self):
        for i in range(0, 51):
            self.root_node.expand()
        return self.root_node.get_visit_distribution()
