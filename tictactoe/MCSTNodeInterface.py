from abc import ABC, abstractmethod


class AbstractMCSTNode(ABC):

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def get_q_and_u_score(self):
        pass

    @abstractmethod
    def get_combined_v_values(self):
        pass
