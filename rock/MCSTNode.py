import numpy as np

from nimmt.MCSTLeafNode import MCSTLeafNode
from nimmt.MCSTNodeInterface import AbstractMCSTNode
from copy import deepcopy


class MCSTNode(AbstractMCSTNode):

    def get_visit_counter(self):
        return self.visit_count

    def __init__(self, p_value, model, board, player_number):
        self.is_not_existing = True
        self.p_value = p_value
        self.model = model
        self.board = board
        self.player_number = player_number
        self.visit_count = 0
        self.v_value = 0
        self.sub_nodes = []
        self.p_values = []
        self.enemy_moves = []


    def expand(self):
        self.visit_count = self.visit_count + 1
        if self.is_not_existing:
            prediction = self.model.predict_on_batch({'input': np.array([self.board.get_input(self.player_number)]),
                                                 'allow': np.array([self.board.get_allow(self.player_number)]).astype(float)})
            self.p_values = prediction[0].numpy().tolist()[0]
            self.v_value = prediction[1].numpy().tolist()[0][0]
            enemy_move_probabilities = prediction[2].numpy().tolist()[0]
            self.enemy_moves = self.board.get_enemy_moves(self.player_number, enemy_move_probabilities)
            for node_number in range(0, len(self.p_values)):
                if self.p_values[node_number] > 0:
                    new_board = deepcopy(self.board)
                    selected_moves = deepcopy(self.enemy_moves)
                    selected_moves.insert(self.player_number, node_number)
                    game_not_finished = new_board.add_move(selected_moves)
                    if game_not_finished:
                        self.sub_nodes.append(MCSTNode(self.p_values[node_number], self.model, new_board, self.player_number))
                    else:
                        game_feedback = new_board.get_feedback_for_player(self.player_number)
                        self.sub_nodes.append(MCSTLeafNode(self.p_values[node_number], game_feedback))
                else:
                    self.sub_nodes.append(MCSTLeafNode(self.p_values[node_number], 0))
            self.is_not_existing = False
            return
        for sub_node_number in range(0, len(self.sub_nodes)):
            if self.board.get_allow(self.player_number)[sub_node_number] == 1:
                best_node = self.sub_nodes[sub_node_number]
                break
        for sub_node_number in range(0, len(self.sub_nodes)):
            if self.sub_nodes[sub_node_number].get_q_and_u_score() > best_node.get_q_and_u_score():
                if self.board.get_allow(self.player_number)[sub_node_number] == 1:
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
