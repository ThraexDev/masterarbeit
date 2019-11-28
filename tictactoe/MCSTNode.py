import numpy as np

from tictactoe.MCSTLeafNode import MCSTLeafNode
from tictactoe.MCSTNodeInterface import AbstractMCSTNode
from copy import deepcopy


class MCSTNode(AbstractMCSTNode):

    def get_visit_counter(self):
        return self.visit_count

    def __init__(self, p_value, model, board, player_number, is_own_move):
        self.is_not_existing = True
        self.p_value = p_value
        self.model = model
        self.board = board
        self.player_number = player_number
        self.visit_count = 0
        self.q_value = 0
        self.is_own_move = is_own_move
        self.sub_nodes = []
        prediction = model.predict_on_batch({'input': np.array([board.get_input(player_number)]), 'allow': np.array([board.get_allow()]).astype(float)})
        self.p_values = prediction[0].numpy()[0].tolist()
        self.v_value = float(prediction[1].numpy()[0][0])
        # turn the win probability if it is the enemies move
        if not is_own_move:
            self.v_value = 1 - self.v_value

    def expand(self):
        self.visit_count = self.visit_count + 1
        if self.is_not_existing:
            for node_number in range(0, len(self.p_values)):
                new_board = deepcopy(self.board)
                current_player_won, game_not_finished = new_board.add_move(self.player_number, node_number)
                if game_not_finished:
                    self.sub_nodes.append(MCSTNode(self.p_values[node_number], self.model, new_board, (self.player_number + 1) % 2, not self.is_own_move))
                else:
                    v_value_for_sub_node = -1
                    if current_player_won: v_value_for_sub_node = 1
                    self.sub_nodes.append(MCSTLeafNode(self.p_values[node_number], v_value_for_sub_node, not self.is_own_move))
            self.is_not_existing = False
            return
        best_node = self.sub_nodes[0]
        for sub_node_number in range(1, len(self.sub_nodes)):
            if self.sub_nodes[sub_node_number].get_q_and_u_score() > best_node.get_q_and_u_score():
                best_node = self.sub_nodes[sub_node_number]
        best_node.expand()

    def get_q_and_u_score(self):
        u_score = self.p_value / (1 + self.visit_count)
        if self.is_not_existing:
            return u_score
        q_score = self.get_combined_v_values() / self.visit_count
        return q_score + u_score

    def get_combined_v_values(self):
        if self.is_not_existing:
            return 0
        v_value = deepcopy(self.v_value)
        for sub_node in self.sub_nodes:
            v_value = v_value + sub_node.get_combined_v_values()
        return v_value

    def get_visit_distribution(self):
        visit_distribution = []
        for i in range(0, len(self.sub_nodes)):
            visit_distribution.append(self.sub_nodes[i].get_visit_counter())
        sum_of_visits = sum(visit_distribution)
        for j in range(0, len(visit_distribution)):
            visit_distribution[j] = visit_distribution[j] / sum_of_visits
        return visit_distribution
