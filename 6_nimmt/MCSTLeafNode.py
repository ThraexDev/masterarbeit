from tictactoe.MCSTNodeInterface import AbstractMCSTNode


class MCSTLeafNode(AbstractMCSTNode):

    def get_visit_counter(self):
        return self.visit_counter

    def __init__(self, p_value, v_value, is_own_move):
        self.p_value = p_value
        self.v_value = v_value
        self.is_own_move = is_own_move
        self.visit_counter = 0
        self.is_not_existing = True

    def get_q_and_u_score(self):
        u_score = self.p_value / (1 + self.visit_counter)
        if self.is_not_existing:
            return u_score
        q_score = self.get_combined_v_values() / self.visit_counter
        return q_score + u_score

    def get_combined_v_values(self):
        if self.is_not_existing:
            return 0
        return self.v_value

    def expand(self):
        self.visit_counter = self.visit_counter + 1
        self.is_not_existing = False
