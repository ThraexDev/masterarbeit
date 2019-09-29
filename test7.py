import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import add
import datetime
import time


def makenewmodel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation=tf.nn.tanh, input_shape=(18,)),
        tf.keras.layers.Dense(27, activation=tf.nn.tanh),
        tf.keras.layers.Dense(9, activation=tf.nn.tanh)
    ])

    model.compile(optimizer='adam',
                  loss=tf.compat.v2.losses.mean_squared_error,
                  metrics=['accuracy'])
    return model


def calculateGameState(playerState1, playerState2):
    if (playerState1[0] == 1 and playerState1[1] and playerState1[2] == 1) or (playerState1[3] == 1 and playerState1[4] and playerState1[5] == 1) or (playerState1[6] == 1 and playerState1[7] and playerState1[8] == 1) or (playerState1[0] == 1 and playerState1[3] and playerState1[6] == 1) or (playerState1[1] == 1 and playerState1[4] and playerState1[7] == 1) or (playerState1[2] == 1 and playerState1[5] and playerState1[8] == 1) or (playerState1[0] == 1 and playerState1[4] and playerState1[8] == 1) or (playerState1[2] == 1 and playerState1[4] and playerState1[6] == 1):
        return 'player1won'
    if sum(playerState1)+sum(playerState2) == 9:
        return 'stalemate'
    return 'pass'


def checkViolation(playerState1, playerState2, action):
    if playerState1[action] == 1 or playerState2[action] == 1:
        return True
    return False

historywon = []
historystale = []
xvalue = []

def evaluate(model):
    wongames = 0
    stalegames = 0
    for i in range(0, 100):
        playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        currentPlayer = 0
        gameResult = calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
        while gameResult == 'pass':
            gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer + 1) % 2]
            action = -1
            if currentPlayer % 2 == 0:
                prediction = model.predict_on_batch(np.array([gameState])).numpy()[0]
                while checkViolation(playerState[0], playerState[1], np.argmax(prediction)):
                    prediction[np.argmax(prediction)] = -1
                action = np.argmax(prediction)
            else:
                for j in range(0, len(playerState[0])):
                    if not checkViolation(playerState[0], playerState[1], j):
                        statecopy = playerState[currentPlayer % 2].copy()
                        statecopy[j] = 1
                        gameResult = calculateGameState(statecopy, playerState[(currentPlayer + 1) % 2])
                        if gameResult == 'player1won' or gameResult == 'stalemate':
                            action = j
                            break
                if action == -1:
                    if not checkViolation(playerState[0], playerState[1], j):
                        for j in range(0, len(playerState[0])):
                            statecopy = playerState[(currentPlayer + 1) % 2].copy()
                            statecopy[j] = 1
                            gameResult = calculateGameState(statecopy, playerState[currentPlayer % 2])
                            if gameResult == 'player1won' or gameResult == 'stalemate':
                                action = j
                                break
                while action == -1:
                    rand = random.randint(0, 8)
                    if not checkViolation(playerState[0], playerState[1], rand):
                        action = rand
            playerState[currentPlayer % 2][action] = 1
            gameResult = calculateGameState(playerState[currentPlayer % 2],
                                                 playerState[(currentPlayer + 1) % 2])
            if gameResult == 'player1won':
                if currentPlayer % 2 == 0:
                    wongames = wongames + 1
            if gameResult == 'stalemate':
                stalegames = stalegames + 1
            currentPlayer = currentPlayer + 1
    historywon.append(wongames)
    historystale.append(stalegames)

oldmodellist = [makenewmodel()]
model = makenewmodel()
startoldmodel = len(oldmodellist) -1
for turnnumber in range(0, 200000):
    #print(str(turnnumber)+'/200000')
    trainmodel = False
    currentoldmodel = startoldmodel
    exportData = []
    exportLabels = []
    nnposition = len(oldmodellist) % 2
    while not trainmodel:
        data = []
        labels = []
        playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        currentPlayer = 0
        gameResult = calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer+1) % 2])
        #print('Spieler ' + str(len(oldmodellist)) + ' gegen ' + str(currentoldmodel))
        while gameResult == 'pass':
            gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer+1) % 2]
            if currentPlayer % 2 == nnposition:
                prediction = model.predict_on_batch(np.array([gameState])).numpy()[0]
            else:
                prediction = oldmodellist[currentoldmodel].predict_on_batch(np.array([gameState])).numpy()[0]
            while checkViolation(playerState[0], playerState[1], np.argmax(prediction)):
                prediction[np.argmax(prediction)] = -1
            action = np.argmax(prediction)
            newLabel = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(0, len(newLabel)):
                if checkViolation(playerState[0], playerState[1], i):
                    newLabel[i] = -1
            newLabel[action] = 1
            if currentPlayer % 2 == nnposition:
                data.append(gameState)
                labels.append(newLabel)
            playerState[currentPlayer % 2][action] = 1
            gameResult = calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
            if gameResult == 'player1won':
                if currentPlayer % 2 == nnposition:
                    currentoldmodel = currentoldmodel - 1
                    if startoldmodel == len(oldmodellist) - 1:
                        if currentoldmodel == -1:
                            trainmodel = True
                            oldmodellist.append(model)
                            if nnposition == 0:
                                evaluate(model)
                                xvalue.append(turnnumber)
                            model.save_weights('testmodel')
                            model = makenewmodel()
                            startoldmodel = len(oldmodellist) - 1
                            ts = time.time()
                            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                            print('Model: ' + str(len(oldmodellist)) + ' Time: ' + st + ' Turn: ' + str(turnnumber))
                    else:
                        trainmodel = True
                        startoldmodel = len(oldmodellist) -1
                else:
                    trainmodel = True
                    startoldmodel = currentoldmodel
                    for i in range(0, len(labels)):
                        for j in range(0, len(labels[i])):
                            if labels[i][j] == 1:
                                labels[i][j] = -1
            if gameResult == 'stalemate':
                if nnposition == 0:
                    trainmodel = True
                    startoldmodel = currentoldmodel
                    for i in range(0, len(labels)):
                        for j in range(0, len(labels[i])):
                            if labels[i][j] == 1:
                                labels[i][j] = -1
                else:
                    currentoldmodel = currentoldmodel - 1
                    if startoldmodel == len(oldmodellist) - 1:
                        if currentoldmodel == -1:
                            trainmodel = True
                            oldmodellist.append(model)
                            if nnposition == 0:
                                evaluate(model)
                                xvalue.append(turnnumber)
                            model.save_weights('testmodel')
                            model = makenewmodel()
                            startoldmodel = len(oldmodellist) - 1
                            ts = time.time()
                            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                            print('Model: ' + str(len(oldmodellist)) + ' Time: ' + st + ' Turn: ' + str(turnnumber))
                    else:
                        trainmodel = True
                        startoldmodel = len(oldmodellist) -1
            currentPlayer = currentPlayer + 1
        exportData.extend(data)
        exportLabels.extend(labels)
    model.train_on_batch(np.array(exportData), np.array(exportLabels))

wonandstale = list( map(add, historywon, historystale))
x = list(range(0,len(historywon)))
plt.ylim(top=100)
plt.ylim(bottom=0)
plt.ylabel('win rate against base line ai in %')
plt.xlabel('iteration')
plt.plot(x, historywon, color='green', label='win rate')
plt.plot(x, historystale, color='blue', label='stalemate rate')
plt.plot(x, wonandstale, color='red', label='combined win and stalemate rate')
plt.legend(loc='upper left')
plt.show()