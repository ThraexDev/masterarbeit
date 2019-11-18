from abc import ABC, abstractmethod


class AbstractMCSTNode(ABC):

    @abstractmethod
    def get_q_and_u_score(self):
        pass