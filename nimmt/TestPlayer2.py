import random

import numpy as np

from nimmt.Board import Board


class TestPlayer2:
    def __init__(self, player_number: int):
        self.player_number = player_number

    def calculate_turn(self, board: Board):
        indices = [i for i, x in enumerate(board.playercards[self.player_number]) if x == 1]
        highestcardinbatches = []
        length_of_batches = []
        for batchnumber in range(0, board.amountofbatches):
            highestcardinbatches.append(max(board.batch[batchnumber]))
            length_of_batches.append(len(board.batch[batchnumber]))
        lowest_length = length_of_batches[np.argmin(length_of_batches)]
        lowest_batches = []
        for batchnumber in range(0, board.amountofbatches):
            if len(board.batch[batchnumber]) == lowest_length:
                lowest_batches.append(batchnumber)
        best_card = board.cardamount + 1
        difference = board.cardamount + 1
        for batchnumber in range(0, len(lowest_batches)):
            for card_number in range(0, len(indices)):
                new_differnce = indices[card_number] - highestcardinbatches[lowest_batches[batchnumber]]
                if 0 <= new_differnce < difference:
                    difference = new_differnce
                    best_card = indices[card_number]
        if best_card == board.cardamount + 1:
            best_card = indices[0]
        selectedcard = best_card
        bulls_in_batches = []
        for batchnumber in range(0, board.amountofbatches):
            bulls_in_batches.append(board.calculatebulls(board.batch[batchnumber]))
        selectedbatch = np.argmin(bulls_in_batches)
        move_vector = [0] * board.cardamount * board.amountofbatches
        move_vector[selectedcard * selectedbatch] = 1
        return move_vector, board.get_input(self.player_number), board.get_allow(self.player_number)

    def get_player_number(self) -> int:
        return self.player_number