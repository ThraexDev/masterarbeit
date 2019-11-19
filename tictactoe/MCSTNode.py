import numpy as np

from tictactoe.MCSTNodeInterface import AbstractMCSTNode


class MCSTNode(AbstractMCSTNode):

    def __init__(self, move, p_value, model, board, player_number, is_own_move):
        self.is_not_existing = True
        self.move = move
        self.p_value = p_value
        self.model = model
        self.board = board
        self.player_number = player_number
        self.visit_count = 0
        self.q_value = 0
        self.own_move = is_own_move
        self.sub_nodes = []
        prediction = model. model.predict_on_batch({'input': np.array([board.get_input(player_number)]), 'allow': np.array([board.get_allow(player_number)]).astype(float)})
        self.p_values = prediction[0]
        self.v_value = prediction[1]
        self.v_value = prediction[1]
        # turn the win probability if it is the enemies move
        if not is_own_move:
            self.v_value = 1 - self.v_value

    def expand(self):
        self.visit_count = self.visit_count + 1
        if self.is_not_existing:
            for node_number in range(0, len(self.p_values)):
                if self.board.isAllowedMove(self.player_number, node_number):
                    new_board = self.board.copy()
                    current_player_won, game_not_finished = new_board.add_move(self.player_number, node_number)
                    self.sub_nodes.append(MCSTNode(node_number, self.p_values[node_number], self.model, new_board, (self.player_number + 1) % 2, not self.is_own_move))
            self.is_not_existing = False
            return
        best_node = self.sub_nodes[0]
        for sub_node_number in range(1, len(self.sub_nodes)):
            if self.sub_nodes[sub_node_number].get_q_and_u_score > best_node.get_q_and_u_score:
                best_node = self.sub_nodes[sub_node_number]
        best_node.expand()

    def get_q_and_u_score(self):
        u_score = self.p_value / (1 + self.visit_count)
        if self.is_not_existing:
            return u_score
        q_score = self.get_combined_v_values() / self.visit_count
        return q_score + u_score

    def get_combined_v_values(self):
        v_value = self.v_value.copy()
        for sub_node in self.sub_nodes:
            v_value = v_value + sub_node.get_combined_q_values
        return v_value
