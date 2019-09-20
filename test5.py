import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import add
from copy import copy

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(72, activation=tf.nn.tanh, input_shape=(18,)),
  tf.keras.layers.Dense(500, activation=tf.nn.tanh),
  tf.keras.layers.Dense(36, activation=tf.nn.tanh),
  tf.keras.layers.Dense(9, activation=tf.nn.tanh)
])

model.compile(optimizer='adam',
              loss=tf.compat.v2.losses.mean_absolute_error,
              metrics=['accuracy'])

historywon = []
historystale = []
oldmodellist = [model]

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(72, activation=tf.nn.tanh, input_shape=(18,)),
  tf.keras.layers.Dense(500, activation=tf.nn.tanh),
  tf.keras.layers.Dense(36, activation=tf.nn.tanh),
  tf.keras.layers.Dense(9, activation=tf.nn.tanh)
])

model.compile(optimizer='adam',
              loss=tf.compat.v2.losses.mean_absolute_error,
              metrics=['accuracy'])


class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel
        self.len = 10000

    def __len__(self):
        return self.len

    def calculateGameState(self, playerState1, playerState2):
        if (playerState1[0] == 1 and playerState1[1] and playerState1[2] == 1) or (playerState1[3] == 1 and playerState1[4] and playerState1[5] == 1) or (playerState1[6] == 1 and playerState1[7] and playerState1[8] == 1) or (playerState1[0] == 1 and playerState1[3] and playerState1[6] == 1) or (playerState1[1] == 1 and playerState1[4] and playerState1[7] == 1) or (playerState1[2] == 1 and playerState1[5] and playerState1[8] == 1) or (playerState1[0] == 1 and playerState1[4] and playerState1[8] == 1) or (playerState1[2] == 1 and playerState1[4] and playerState1[6] == 1):
            return 'player1won'
        if sum(playerState1)+sum(playerState2) == 9:
            return 'stalemate'
        return 'pass'

    def checkViolation(self, playerState1, playerState2, action):
        if playerState1[action] == 1 or playerState2[action] == 1:
            return True
        return False

    def evaluate(self):
        wongames = 0
        stalegames = 0
        for i in range(0, 10):
            playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
            currentPlayer = 0
            gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
            while gameResult == 'pass':
                gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer + 1) % 2]
                action = -1
                if currentPlayer % 2 == 0:
                    prediction = self.model.predict_on_batch(np.array([gameState])).numpy()[0]
                    while self.checkViolation(playerState[0], playerState[1], np.argmax(prediction)):
                        prediction[np.argmax(prediction)] = -1
                    action = np.argmax(prediction)
                else:
                    for j in range(0, len(playerState[0])):
                        if not self.checkViolation(playerState[0], playerState[1], j):
                            statecopy = playerState[currentPlayer % 2].copy()
                            statecopy[j] = 1
                            gameResult = self.calculateGameState(statecopy, playerState[(currentPlayer + 1) % 2])
                            if gameResult == 'player1won' or gameResult == 'stalemate':
                                action = j
                                break
                    if action == -1:
                        if not self.checkViolation(playerState[0], playerState[1], j):
                            for j in range(0, len(playerState[0])):
                                statecopy = playerState[(currentPlayer + 1) % 2].copy()
                                statecopy[j] = 1
                                gameResult = self.calculateGameState(statecopy, playerState[currentPlayer % 2])
                                if gameResult == 'player1won' or gameResult == 'stalemate':
                                    action = j
                                    break
                    while action == -1:
                        rand = random.randint(0, 8)
                        if not self.checkViolation(playerState[0], playerState[1], rand):
                            action = rand
                playerState[currentPlayer % 2][action] = 1
                gameResult = self.calculateGameState(playerState[currentPlayer % 2],
                                                     playerState[(currentPlayer + 1) % 2])
                if gameResult == 'player1won':
                    if currentPlayer % 2 == 0:
                        wongames = wongames + 1
                if gameResult == 'stalemate':
                    stalegames = stalegames + 1
                currentPlayer = currentPlayer + 1
        historywon.append(wongames)
        historystale.append(stalegames)

    def __getitem__(self, item):
        trainmodel = False
        currentoldmodel = len(oldmodellist) -1
        exportData = []
        exportLabels = []
        nnposition = len(oldmodellist) % 2
        while not trainmodel:
            data = []
            labels = []
            playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
            currentPlayer = 0
            gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer+1) % 2])
            while gameResult == 'pass':
                gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer+1) % 2]
                if currentPlayer % 2 == nnposition:
                    prediction = self.model.predict_on_batch(np.array([gameState])).numpy()[0]
                else:
                    prediction = oldmodellist[currentoldmodel].predict_on_batch(np.array([gameState])).numpy()[0]
                while self.checkViolation(playerState[0], playerState[1], np.argmax(prediction)):
                    prediction[np.argmax(prediction)] = -1
                action = np.argmax(prediction)
                newLabel = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                for i in range(0, len(newLabel)):
                    if self.checkViolation(playerState[0], playerState[1], i):
                        newLabel[i] = 0
                newLabel[action] = 1
                if currentPlayer % 2 == nnposition:
                    data.append(gameState)
                    labels.append(newLabel)
                playerState[currentPlayer % 2][action] = 1
                gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
                if gameResult == 'player1won':
                    if currentPlayer % 2 == nnposition:
                        currentoldmodel = currentoldmodel - 1
                        if currentoldmodel == -1:
                            self.oldmodellist.append(copy(self.model))
                            trainmodel = True
                            self.evaluate()
                            print('')
                            print(len(oldmodellist))
                    else:
                        trainmodel = True
                        for i in range(0,len(labels)):
                            for j in range(0, len(labels)):
                                if labels[i][j] == 1:
                                    labels[i][j] = -1
                if gameResult == 'stalemate':
                    if nnposition == 0:
                        trainmodel = True
                        for i in range(0, len(labels)):
                            for j in range(0, len(labels)):
                                if labels[i][j] == 1:
                                    labels[i][j] = -1
                    else:
                        currentoldmodel = currentoldmodel - 1
                        if currentoldmodel == -1:
                            self.oldmodellist.append(copy(self.model))
                            trainmodel = True
                            self.evaluate()
                            print('')
                            print(len(oldmodellist))
                currentPlayer = currentPlayer + 1
            exportData.extend(data)
            exportLabels.extend(labels)
        return np.array(exportData), np.array(exportLabels)


generator = GameStateGenerator(model)


model.fit_generator(generator=generator, epochs=1, workers=0, max_queue_size=1, verbose=1, shuffle=False)
N = 1
cumsum, moving_aves_won = [0], []
for i, x in enumerate(historywon, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves_won.append(moving_ave)
cumsum, moving_aves_stale = [0], []
for i, x in enumerate(historystale, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves_stale.append(moving_ave)
wonandstale = list( map(add, historywon, historystale))
cumsum, moving_aves_stale_and_won = [0], []
for i, x in enumerate(wonandstale, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves_stale_and_won.append(moving_ave)
x = list(range(0,len(moving_aves_won)))
plt.ylim(top=10)
plt.ylim(bottom=0)
plt.ylabel('win rate against base line ai in %')
plt.xlabel('iteration (in 10)')
plt.plot(x, moving_aves_won, color='green', label='win rate')
plt.plot(x, moving_aves_stale, color='blue', label='stalemate rate')
plt.plot(x, moving_aves_stale_and_won, color='red', label='combined win and stalemate rate')
plt.legend(loc='upper left')
plt.show()
