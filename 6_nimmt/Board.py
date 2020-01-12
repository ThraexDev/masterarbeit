import random

import numpy as np

class Board:

    def __init__(self, cardshandedtoeachplayer, playeramount, cardamount, amountofbatches):
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
                    randomcard = random.randint(1, cardamount)
                    if randomcard not in cardshandedout:
                        self.playercards[playernumber][randomcard - 1] = 1
                        cardshandedout.append(randomcard)
                        cardnotset = False
        for batchnumber in range(0, amountofbatches):
            self.batch.append([])
            cardnotset = True
            while cardnotset:
                randomcard = self.random.randint(1, cardamount)
                if randomcard not in cardshandedout:
                    self.playedcards[randomcard - 1] = 1
                    self.batch[batchnumber].append(randomcard)
                    cardshandedout.append(randomcard)
                    cardnotset = False

    def add_move(self, player_number: int, move: int) -> (int, bool):
        highestcardinbatches = []
        selected_card = move % 10
        selected_batch = int((move - selected_card) / 10)
        for batchnumber in range(0, amountofbatches):
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
            assingedbatch = np.argmin(differencetocard)
            if len(self.batch[assingedbatch]) == 5:
                self.playerbulls[player_number] = self.playerbulls[player_number] + self.calculatebulls(
                    self.batch[assingedbatch])
                self.batch[assingedbatch] = [selected_card]
            else:
                self.batch[assingedbatch].append(selected_card)
        game_not_finished = True
        game_feedback = 0
        if min(self.playerbulls) == self.playerbulls[player_number]:
            game_feedback = 1
        if max(self.playerbulls) == self.playerbulls[player_number]:
            game_feedback = -1
        sum_of_not_played_cards = 0
        for player in range(0, playeramount):
            sum_of_not_played_cards = sum_of_not_played_cards + sum(self.playercards[player])
        if sum_of_not_played_cards == 0:
            game_not_finished = False
        return game_feedback, game_not_finished

    def get_input(self, player_number: int) -> list:
        situationvector = []
        situationvector.extend(self.playercards[player_number])
        numberofcardsvector = [0] * 10
        numberofcardsonhand = sum(self.playercards[player_number]) - 1
        numberofcardsvector[numberofcardsonhand] = 1
        situationvector.extend(numberofcardsvector)
        for batchnumber in range(0, len(self.batch)):
            batchvector = [0] * cardamount
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
        for player in range(0, playeramount):
            if player != player_number:
                situationvector.extend(self.get_bull_vector(player))
        return situationvector

    def get_allow(self, player_number) -> list:
        allow_vector = []
        cards_not_played = sum(self.playercards[player_number])
        played_cards = cardshandedtoeachplayer - cards_not_played
        for player in range(0, playeramount):
            allow = [1] * cards_not_played
            allow_vector.extend(allow)
            not_allow_vector = [0] * played_cards
            allow_vector.extend(not_allow_vector)
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
        if 0 < bulls < 10:
            bull_vector[1] = 1
        if 10 < bulls < 20:
            bull_vector[2] = 1
        if 20 < bulls < 30:
            bull_vector[3] = 1
        if 30 < bulls < 40:
            bull_vector[4] = 1
        if 40 < bulls < 50:
            bull_vector[5] = 1
        if 50 < bulls < 60:
            bull_vector[6] = 1
        if 60 < bulls < 70:
            bull_vector[7] = 1
        if 70 < bulls < 80:
            bull_vector[8] = 1
        if bulls > 80:
            bull_vector[9] = 1
        return bull_vector

