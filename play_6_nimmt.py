import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_vector = tf.keras.Input(shape=(754,), name='input')
allowed_moves = tf.keras.Input(shape=(104,), name='allow')
middle = tf.keras.layers.Dense(250, activation=tf.nn.tanh, name='middle')(input_vector)
cardprobability = tf.keras.layers.Dense(104, activation=tf.nn.tanh, name='cardprobability')(middle)
batchprobability = tf.keras.layers.Dense(4, activation=tf.nn.tanh, name='batchprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply()([allowed_moves, cardprobability])

model = tf.keras.Model(inputs=[input_vector, allowed_moves],
                       outputs=[allowedcardprobability, batchprobability])

model.compile(optimizer='adam',
              loss=[tf.compat.v2.losses.mean_squared_error, tf.compat.v2.losses.mean_squared_error],
              metrics=['accuracy'])

model.load_weights("6model")

amountofbatches = 4
cardshandedtoeachplayer = 10
cardamount = 104
playeramount = 4


def calculatebulls(batch):
    bulls = 0
    for cardnumber in range(0, len(batch)):
        bulls = bulls + 1
        if batch[cardnumber] % 5 == 0:
            bulls = bulls + 1
        if batch[cardnumber] % 10 == 0:
            bulls = bulls + 1
        if batch[cardnumber] == 11 or batch[cardnumber] == 22 or batch[cardnumber] == 33 or batch[cardnumber] == 44 or \
                batch[cardnumber] == 55 or batch[cardnumber] == 66 or batch[cardnumber] == 77 or batch[
            cardnumber] == 88 or batch[cardnumber] == 99:
            bulls = bulls + 4
    return bulls


def getplayersituationvector(playervector, batch, playedcards):
    situationvector = []
    situationvector.extend(playervector)
    numberofcardsvector = [0] * 10
    numberofcardsonhand = sum(playervector) - 1
    numberofcardsvector[numberofcardsonhand] = 1
    situationvector.extend(numberofcardsvector)
    for batchnumber in range(0, len(batch)):
        batchvector = [0] * 104
        batchvector[max(batch[batchnumber]) - 1] = 1
        situationvector.extend(batchvector)
        numberofcardsvector = [0] * 5
        numberofcardsinbatch = len(batch[batchnumber]) - 1
        numberofcardsvector[numberofcardsinbatch] = 1
        bullvector = [0] * 30
        bullvector[calculatebulls(batch[batchnumber])] = 1
        situationvector.extend(bullvector)
    situationvector.extend(playedcards)
    return situationvector
################################################################################################################

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
            print(playernumber)
            if playernumber == 0:
                indices = [i for i, x in enumerate(playercards[playernumber]) if x == 1]
                print('bulls:')
                print(playerbulls[0])
                print(playerbulls[1])
                print(playerbulls[2])
                print(playerbulls[3])
                print('batch:')
                for print_batch in batch:
                    print(print_batch)
                print('hand:')
                print(indices)
                randomcard = int(input('card: '))
                selectedcard = indices[randomcard]
                selectedbatch = int(input('batch: '))
            else:
                cardsonhandsvector = playercards[playernumber]
                situtationvector = getplayersituationvector(cardsonhandsvector, batch, playedcards)
                prediction = model.predict_on_batch(
                    {'input': np.array([situtationvector]),
                     'allow': np.array([cardsonhandsvector]).astype(float)})
                cardpredicion = prediction[0].numpy()
                batchpredicion = prediction[1].numpy()
                selectedcard = np.argmax(cardpredicion)
                selectedbatch = np.argmax(batchpredicion)
                print('card:')
                print(selectedcard)
                print('batch:')
                print(selectedbatch)
            # ist nötig, da die vektoren nullbasiert sind
            selectedcard = selectedcard + 1
            #############################################
            selectedcards.append(selectedcard)
            selectedbatches.append(selectedbatch)
        for playernumber in range(0, playeramount):
            highestcardinbatches = []
            for batchnumber in range(0, amountofbatches):
                highestcardinbatches.append(max(batch[batchnumber]))
            lowestcard = min(selectedcards)
            playedcards[lowestcard-1] = 1
            playeroflowestcard = np.argmin(selectedcards)
            playercards[playeroflowestcard][lowestcard - 1] = 0
            differencetocard = [lowestcard - highcardofbatch for highcardofbatch in highestcardinbatches]
            if max(differencetocard) < 0:
                playerbulls[playeroflowestcard] = playerbulls[playeroflowestcard] + calculatebulls(
                    batch[selectedbatches[playeroflowestcard]])
                batch[selectedbatches[playeroflowestcard]] = [lowestcard]
                if max(playerbulls) >= 100:
                    gamefinished = True
            else:
                for difference in range(0, len(differencetocard)):
                    if differencetocard[difference] < 0:
                        differencetocard[difference] = 105
                assingedbatch = np.argmin(differencetocard)
                if len(batch[assingedbatch]) == 5:
                    playerbulls[playeroflowestcard] = playerbulls[playeroflowestcard] + calculatebulls(
                        batch[assingedbatch])
                    batch[assingedbatch] = [lowestcard]
                    if max(playerbulls) >= 100:
                        gamefinished = True
                else:
                    batch[assingedbatch].append(lowestcard)
            selectedcards[playeroflowestcard] = 105
bestplayer = np.argmin(playerbulls)
if bestplayer == 0:
    print('won')
