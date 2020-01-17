import random

import numpy as np

class Board:

    def __init__(self, cardshandedtoeachplayer, playeramount, cardamount, amountofbatches, maxbatchcards):
        self.maxbatchcards = maxbatchcards
        self.cardshandedtoeachplayer = cardshandedtoeachplayer
        self.playeramount = playeramount
        self.cardamount = cardamount
        self.amountofbatches = amountofbatches
        self.playerbulls = []
        self.playercards = []
        self.playedcards = [0] * cardamount
        self.batch = []
        for playernumber in range(0, playeramount):
            self.playerbulls.append(0)
        cardshandedout = []
        for playernumber in range(0, playeramount):
            self.playercards.append([0] * cardamount)
            for cardnumber in range(0, cardshandedtoeachplayer):
                cardnotset = True
                while cardnotset:
                    randomcard = random.randint(0, cardamount-1)
                    if randomcard not in cardshandedout:
                        self.playercards[playernumber][randomcard] = 1
                        cardshandedout.append(randomcard)
                        cardnotset = False
        for batchnumber in range(0, amountofbatches):
            self.batch.append([])
            cardnotset = True
            while cardnotset:
                randomcard = random.randint(0, cardamount-1)
                if randomcard not in cardshandedout:
                    self.playedcards[randomcard] = 1
                    self.batch[batchnumber].append(randomcard)
                    cardshandedout.append(randomcard)
                    cardnotset = False

    def add_move(self, selected_moves: list) -> (int, bool):
        selectedcards = []
        selectedbatches = []
        for player_number in range(0, len(selected_moves)):
            selected_card = selected_moves[player_number] % self.cardamount
            selected_batch = int((selected_moves[player_number] - selected_card) / self.cardamount)
            selectedcards.append(selected_card)
            selectedbatches.append(selected_batch)
        for number in range(0, len(selected_moves)):
            player_of_lowest_card = np.argmin(selectedcards)
            self.add_move_for_player(player_of_lowest_card, selectedcards[player_of_lowest_card], selectedbatches[player_of_lowest_card])
            selectedcards[player_of_lowest_card] = self.cardamount + 1
        game_not_finished = True
        for player in range(0, self.playeramount):
            if sum(self.playercards[player]) == 0:
                game_not_finished = False
        return game_not_finished

    def add_move_for_player(self, player_number: int, selected_card: int, selected_batch: int):
        highestcardinbatches = []
        for batchnumber in range(0, self.amountofbatches):
            highestcardinbatches.append(max(self.batch[batchnumber]))
        self.playedcards[selected_card] = 1
        self.playercards[player_number][selected_card] = 0
        differencetocard = [selected_card - highcardofbatch for highcardofbatch in
                            highestcardinbatches]
        if max(differencetocard) < 0:
            self.playerbulls[player_number] = self.playerbulls[player_number] + self.calculatebulls(
                self.batch[selected_batch])
            self.batch[selected_batch] = [selected_card]
        else:
            for difference in range(0, len(differencetocard)):
                if differencetocard[difference] < 0:
                    differencetocard[difference] = self.cardamount + 1
            assingedbatch = np.argmin(differencetocard)
            if len(self.batch[assingedbatch]) == self.maxbatchcards:
                self.playerbulls[player_number] = self.playerbulls[player_number] + self.calculatebulls(
                    self.batch[assingedbatch])
                self.batch[assingedbatch] = [selected_card]
            else:
                self.batch[assingedbatch].append(selected_card)

    def get_input(self, player_number: int) -> list:
        situationvector = []
        situationvector.extend(self.playercards[player_number])
        numberofcardsvector = [0] * 10
        numberofcardsonhand = sum(self.playercards[player_number]) - 1
        numberofcardsvector[numberofcardsonhand] = 1
        situationvector.extend(numberofcardsvector)
        for batchnumber in range(0, len(self.batch)):
            batchvector = [0] * self.cardamount
            batchvector[max(self.batch[batchnumber]) - 1] = 1
            situationvector.extend(batchvector)
            numberofcardsvector = [0] * 5
            numberofcardsinbatch = len(self.batch[batchnumber]) - 1
            numberofcardsvector[numberofcardsinbatch] = 1
            bullvector = [0] * 30
            bullvector[self.calculatebulls(self.batch[batchnumber])] = 1
            situationvector.extend(bullvector)
        situationvector.extend(self.playedcards)
        situationvector.extend(self.get_bull_vector(player_number))
        for player in range(0, self.playeramount):
            if player != player_number:
                situationvector.extend(self.get_bull_vector(player))
        return situationvector

    def get_allow(self, player_number) -> list:
        allow_vector = []
        for batch_number in range(0, self.amountofbatches):
            allow_vector.extend(self.playercards[player_number])
        return allow_vector

    def calculatebulls(self, batch):
        bulls = 0
        for cardnumber in range(0, len(batch)):
            bulls = bulls + 1
            if batch[cardnumber] % 5 == 0:
                bulls = bulls + 1
            if batch[cardnumber] % 10 == 0:
                bulls = bulls + 1
            if batch[cardnumber] == 11 or batch[cardnumber] == 22 or batch[cardnumber] == 33 or batch[
                cardnumber] == 44 or batch[cardnumber] == 55 or batch[cardnumber] == 66 or batch[cardnumber] == 77 or \
                    batch[cardnumber] == 88 or batch[cardnumber] == 99:
                bulls = bulls + 4
        return bulls

    def get_bull_vector(self, player_number):
        bull_vector = [0] * 10
        bulls = self.playerbulls[player_number]
        if bulls == 0:
            bull_vector[0] = 1
        if 0 < bulls < 1:
            bull_vector[1] = 1
        if 1 < bulls < 2:
            bull_vector[2] = 1
        if 2 < bulls < 4:
            bull_vector[3] = 1
        if 4 < bulls < 7:
            bull_vector[4] = 1
        if 7 < bulls < 10:
            bull_vector[5] = 1
        if 10 < bulls < 15:
            bull_vector[6] = 1
        if 15 < bulls < 25:
            bull_vector[7] = 1
        if 25 < bulls < 40:
            bull_vector[8] = 1
        if bulls > 40:
            bull_vector[9] = 1
        return bull_vector

    def get_feedback_for_player(self, player_number: int) -> int:
        if min(self.playerbulls) == max(self.playerbulls):
            return 0
        if self.playerbulls[player_number] == max(self.playerbulls):
            return -1
        if self.playerbulls[player_number] == min(self.playerbulls):
            return 1
        return 0

    def get_enemy_moves(self, player_number, probabilities):
        enemy_moves = []
        enemy_cards = []
        while len(enemy_moves) < (self.playeramount - 1):
            enemy_move = np.argmax(probabilities)
            probabilities[enemy_move] = 0
            selected_card = enemy_move % self.cardamount
            if selected_card not in enemy_cards and self.playercards[player_number][selected_card] == 0 and self.playedcards[selected_card] == 0:
                enemy_cards.append(selected_card)
                enemy_moves.append(enemy_move)
        return enemy_moves
