import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import add

input = tf.keras.Input(shape=(754,), name='input')
allowed_moves = tf.keras.Input(shape=(104,), name='allow')
middle = tf.keras.layers.Dense(250, activation=tf.nn.tanh, name='middle')(input)
cardprobability = tf.keras.layers.Dense(104, activation=tf.nn.tanh, name='cardprobability')(middle)
batchprobability = tf.keras.layers.Dense(4, activation=tf.nn.tanh, name='batchprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply()([allowed_moves, cardprobability])

model = tf.keras.Model(inputs=[input, allowed_moves],
                    outputs=[allowedcardprobability, batchprobability])

model.compile(optimizer='adam',
              loss=[tf.compat.v2.losses.mean_squared_error, tf.compat.v2.losses.mean_squared_error],
              metrics=['accuracy'])

historywon = []
vertifygames = 1000
distanceofvertification = 100

class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel

    def __len__(self):
        return 10000

    def calculatebulls(self, batch):
        bulls = 0
        for cardnumber in range(0, len(batch)):
            bulls = bulls + 1
            if batch[cardnumber] % 5 == 0:
                bulls = bulls + 1
            if batch[cardnumber] % 10 == 0:
                bulls = bulls + 1
            if batch[cardnumber] == 11 or batch[cardnumber] == 22 or batch[cardnumber] == 33 or batch[cardnumber] == 44 or batch[cardnumber] == 55 or batch[cardnumber] == 66 or batch[cardnumber] == 77 or batch[cardnumber] == 88 or batch[cardnumber] == 99:
                bulls = bulls + 4
        return bulls

    def getplayersituationvector(self, playervector, batch, playedcards):
        situationvector = []
        situationvector.extend(playervector)
        numberofcardsvector = [0] * 10
        numberofcardsonhand = sum(playervector)-1
        numberofcardsvector[numberofcardsonhand] = 1
        situationvector.extend(numberofcardsvector)
        for batchnumber in range(0, len(batch)):
            batchvector = [0] * 104
            batchvector[max(batch[batchnumber])-1] = 1
            situationvector.extend(batchvector)
            numberofcardsvector = [0] * 5
            numberofcardsinbatch = len(batch[batchnumber]) - 1
            numberofcardsvector[numberofcardsinbatch] = 1
            bullvector = [0] * 30
            bullvector[self.calculatebulls(batch[batchnumber])] = 1
            situationvector.extend(bullvector)
        situationvector.extend(playedcards)
        return situationvector

    def __getitem__(self, item):
        amountofbatches = 4
        cardshandedtoeachplayer = 10
        cardamount = 104
        playeramount = 4
        ################################################################################################################
        if item % distanceofvertification == 0 or item == 0:
            wongames = 0
            for testgame in range(0, vertifygames):
                playerbulls = []
                for playernumber in range(0, playeramount):
                    playerbulls.append(0)
                gamefinished = False
                while not gamefinished:
                    batch = []
                    cardshandedout = []
                    playercards = []
                    playedcards = [0] * cardamount
                    for playernumber in range(0, playeramount):
                        playercards.append([0] * cardamount)
                        for cardnumber in range(0, cardshandedtoeachplayer):
                            cardnotset = True
                            while cardnotset:
                                randomcard = random.randint(1, cardamount)
                                if randomcard not in cardshandedout:
                                    playercards[playernumber][randomcard - 1] = 1
                                    cardshandedout.append(randomcard)
                                    cardnotset = False
                    for batchnumber in range(0, amountofbatches):
                        batch.append([])
                        cardnotset = True
                        while cardnotset:
                            randomcard = random.randint(1, cardamount)
                            if randomcard not in cardshandedout:
                                playedcards[randomcard - 1] = 1
                                batch[batchnumber].append(randomcard)
                                cardshandedout.append(randomcard)
                                cardnotset = False
                    for cardnumber in range(0, cardshandedtoeachplayer):
                        selectedcards = []
                        selectedbatches = []
                        for playernumber in range(0, playeramount):
                            selectedcard = 0
                            selectedbatch = 0
                            if playernumber == 0:
                                cardsonhandsvector = playercards[playernumber]
                                situtationvector = self.getplayersituationvector(cardsonhandsvector, batch, playedcards)
                                prediction = model.predict_on_batch(
                                    {'input': np.array([situtationvector]), 'allow': np.array([cardsonhandsvector]).astype(float)})
                                cardpredicion = prediction[0].numpy()
                                batchpredicion = prediction[1].numpy()
                                selectedcard = np.argmax(cardpredicion)
                                selectedbatch = np.argmax(batchpredicion)
                            else:
                                indices = [i for i, x in enumerate(playercards[playernumber]) if x == 1]
                                randomcard = random.randint(0, len(indices)-1)
                                selectedcard = indices[randomcard]
                                selectedbatch = random.randint(0, amountofbatches-1)
                            # ist nötig, da die vektoren nullbasiert sind
                            selectedcard = selectedcard + 1
                            #############################################
                            selectedcards.append(selectedcard)
                            selectedbatches.append(selectedbatch)
                        highestcardinbatches = []
                        for batchnumber in range(0, amountofbatches):
                            highestcardinbatches.append(max(batch[batchnumber]))
                        for playernumber in range(1, playeramount):
                            lowestcard = min(selectedcards)
                            playedcards[lowestcard] = 1
                            playeroflowestcard = np.argmin(selectedcards)
                            playercards[playeroflowestcard][lowestcard - 1] = 0
                            differencetocard = [lowestcard - highcardofbatch for highcardofbatch in highestcardinbatches]
                            if max(differencetocard) < 0:
                                playerbulls[playeroflowestcard] = playerbulls[playeroflowestcard] + self.calculatebulls(
                                    batch[selectedbatches[playeroflowestcard]])
                                batch[selectedbatches[playeroflowestcard]] = [lowestcard]
                                if np.argmax(playerbulls) <= 100:
                                    gamefinished = True
                                    break
                                continue
                            for difference in range(0, len(differencetocard)):
                                if differencetocard[difference] < 0:
                                    differencetocard[difference] = 105
                            assingedbatch = np.argmin(differencetocard)
                            if len(batch[assingedbatch]) == 5:
                                playerbulls[playeroflowestcard] = playerbulls[playeroflowestcard] + self.calculatebulls(
                                    batch[assingedbatch])
                                batch[assingedbatch] = [lowestcard]
                                if np.argmax(playerbulls) <= 100:
                                    gamefinished = True
                                    break
                            else:
                                batch[assingedbatch].append(lowestcard)
                            selectedcards[playeroflowestcard] = 105
                bestplayer = np.argmin(playerbulls)
                if bestplayer == 0:
                    wongames = wongames + 1
            historywon.append(wongames/(vertifygames/100))
        ################################################################################################################
        playertargetvectors = []
        playerinputvectors = []
        playerbulls = []
        for playernumber in range(0, playeramount):
            playerbulls.append(0)
            playertargetvectors.append([[], []])
            playerinputvectors.append([[], []])
        gamefinished = False
        while not gamefinished:
            batch = []
            cardshandedout = []
            playercards = []
            playedcards = [0] * cardamount
            for playernumber in range(0, playeramount):
                playercards.append([0]*cardamount)
                for cardnumber in range(0, cardshandedtoeachplayer):
                    cardnotset = True
                    while cardnotset:
                        randomcard = random.randint(1, cardamount)
                        if randomcard not in cardshandedout:
                            playercards[playernumber][randomcard-1] = 1
                            cardshandedout.append(randomcard)
                            cardnotset = False
            for batchnumber in range(0, amountofbatches):
                batch.append([])
                cardnotset = True
                while cardnotset:
                    randomcard = random.randint(1, cardamount)
                    if randomcard not in cardshandedout:
                        playedcards[randomcard - 1] = 1
                        batch[batchnumber].append(randomcard)
                        cardshandedout.append(randomcard)
                        cardnotset = False
            for cardnumber in range(0, cardshandedtoeachplayer):
                selectedcards = []
                selectedbatches = []
                for playernumber in range(0, playeramount):
                    cardsonhandsvector = playercards[playernumber]
                    situtationvector = self.getplayersituationvector(cardsonhandsvector, batch, playedcards)
                    prediction = model.predict_on_batch({'input': np.array([situtationvector]), 'allow': np.array([cardsonhandsvector]).astype(float)})
                    cardpredicion = prediction[0].numpy()
                    batchpredicion = prediction[1].numpy()
                    selectedcard = np.argmax(cardpredicion)
                    selectedbatch = np.argmax(batchpredicion)
                    cardtargetvector = [0] * cardamount
                    batchtargetvector = [0] * amountofbatches
                    playerinputvectors[playernumber][0].append(situtationvector)
                    playerinputvectors[playernumber][1].append(cardsonhandsvector)
                    cardtargetvector[selectedcard] = -1
                    playertargetvectors[playernumber][0].append(cardtargetvector)
                    batchtargetvector[selectedbatch] = -1
                    playertargetvectors[playernumber][1].append(batchtargetvector)
                    # ist nötig, da die vektoren nullbasiert sind
                    selectedcard = selectedcard + 1
                    #############################################
                    selectedcards.append(selectedcard)
                    selectedbatches.append(selectedbatch)
                highestcardinbatches = []
                for batchnumber in range(0, amountofbatches):
                    highestcardinbatches.append(max(batch[batchnumber]))
                for playernumber in range(1, playeramount):
                    lowestcard = min(selectedcards)
                    playedcards[lowestcard] = 1
                    playeroflowestcard = np.argmin(selectedcards)
                    differencetocard = [lowestcard - highcardofbatch for highcardofbatch in highestcardinbatches]
                    playercards[playeroflowestcard][lowestcard-1] = 0
                    if max(differencetocard) < 0:
                        playerbulls[playeroflowestcard] = playerbulls[playeroflowestcard] + self.calculatebulls(batch[selectedbatches[playeroflowestcard]])
                        batch[selectedbatches[playeroflowestcard]] = [lowestcard]
                        if np.argmax(playerbulls) <= 100:
                            gamefinished = True
                            break
                        continue
                    for difference in range(0, len(differencetocard)):
                        if differencetocard[difference] < 0:
                            differencetocard[difference] = 105
                    assingedbatch = np.argmin(differencetocard)
                    if len(batch[assingedbatch]) == 5:
                        playerbulls[playeroflowestcard] = playerbulls[playeroflowestcard] + self.calculatebulls(batch[assingedbatch])
                        batch[assingedbatch] = [lowestcard]
                        if np.argmax(playerbulls) <= 100:
                            gamefinished = True
                            break
                    else:
                        batch[assingedbatch].append(lowestcard)
                    selectedcards[playeroflowestcard] = 105
        bestplayer = np.argmin(playerbulls)
        worstplayer = np.argmax(playerbulls)
        inputexport = []
        allowexport = []
        cardprobabilityexport = []
        batchprobabilityexport = []
        inputexport.extend(playerinputvectors[worstplayer][0])
        allowexport.extend(playerinputvectors[worstplayer][1])
        cardprobabilityexport.extend(playertargetvectors[worstplayer][0])
        batchprobabilityexport.extend(playertargetvectors[worstplayer][1])
        inputexport.extend(playerinputvectors[bestplayer][0])
        allowexport.extend(playerinputvectors[bestplayer][1])
        bestplayertagetcards = playertargetvectors[bestplayer][0]
        bestplayertagetbatches = playertargetvectors[bestplayer][1]
        for turnnumber in range(0, len(bestplayertagetcards)):
            for cardnumber in range(0, len(bestplayertagetcards[turnnumber])):
                if bestplayertagetcards[turnnumber][cardnumber] == -1:
                    bestplayertagetcards[turnnumber][cardnumber] = 1
        for turnnumber in range(0, len(bestplayertagetbatches)):
            for cardnumber in range(0, len(bestplayertagetbatches[turnnumber])):
                if bestplayertagetbatches[turnnumber][cardnumber] == -1:
                    bestplayertagetbatches[turnnumber][cardnumber] = 1
        cardprobabilityexport.extend(bestplayertagetcards)
        batchprobabilityexport.extend(bestplayertagetbatches)
        return {'input': np.array(inputexport), 'allow': np.array(allowexport)}, {'multiply': np.array(cardprobabilityexport), 'batchprobability': np.array(batchprobabilityexport)}

generator = GameStateGenerator(model)

model.fit_generator(generator=generator, epochs=1, workers=10, max_queue_size=10, verbose=1, shuffle=False)

N = 1
cumsum, moving_aves_won = [0], []
for i, x in enumerate(historywon, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves_won.append(moving_ave)
x = list(range(0,len(moving_aves_won)))
plt.ylim(top=max(moving_aves_won))
plt.ylim(bottom=min(moving_aves_won))
plt.ylabel('win rate against base line ai in %')
plt.xlabel('iteration (in '+str(distanceofvertification)+')')
plt.plot(x, moving_aves_won, color='green', label='win rate')
plt.legend(loc='upper left')
plt.show()