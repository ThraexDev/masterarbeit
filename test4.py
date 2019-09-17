import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import add

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(72, activation=tf.nn.relu, input_shape=(18,)),
  tf.keras.layers.Dense(36, activation=tf.nn.relu),
  tf.keras.layers.Dense(9, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.compat.v2.losses.mean_absolute_error,
              metrics=['accuracy'])

historywon = []
historystale = []


class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel

    def __len__(self):
        return 100000

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

    def __getitem__(self, item):
        if item % 1000 == 0:
            wongames = 0
            stalegames = 0
            for i in range(0, 100):
                playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                currentPlayer = 0
                gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
                while gameResult == 'pass':
                    gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer+1) % 2]
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
                    gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
                    if gameResult == 'player1won':
                        if currentPlayer % 2 == 0:
                            wongames = wongames + 1
                    if gameResult == 'stalemate':
                        stalegames = stalegames + 1
                    currentPlayer = currentPlayer + 1
            historywon.append(wongames)
            historystale.append(stalegames)
######################################################################################################################################
        data = [[], []]
        labels = [[], []]
        playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        currentPlayer = 0
        gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer+1) % 2])
        while gameResult == 'pass':
            gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer+1) % 2]
            prediction = self.model.predict_on_batch(np.array([gameState])).numpy()[0]
            while self.checkViolation(playerState[0], playerState[1], np.argmax(prediction)):
                prediction[np.argmax(prediction)] = -1
            action = np.argmax(prediction)
            data[currentPlayer % 2].append(gameState)
            newLabel = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            for i in range(0, len(newLabel)):
                if self.checkViolation(playerState[0], playerState[1], i):
                    newLabel[i] = 0
            newLabel[action] = 1
            labels[currentPlayer % 2].append(newLabel)
            playerState[currentPlayer % 2][action] = 1
            gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
            if gameResult == 'player1won':
                for i in range(0,len(labels[(currentPlayer+1) % 2])):
                    for j in range(0, len(labels[(currentPlayer+1) % 2][i])):
                        if labels[(currentPlayer+1) % 2][i][j] == 1:
                            labels[(currentPlayer + 1) % 2][i][j] = 0
                print('Player '+str( (currentPlayer+1) % 2) + ' lost ')
            if gameResult == 'stalemate':
                for i in range(0,len(labels[(currentPlayer) % 2])):
                    for j in range(0, len(labels[(currentPlayer) % 2][i])):
                        if labels[(currentPlayer) % 2][i][j] == 1:
                            labels[(currentPlayer) % 2][i][j] = 0
            currentPlayer = currentPlayer + 1
        exportData = data[0]
        exportData.extend(data[1])
        exportLabels = labels[0]
        exportLabels.extend(labels[1])
        return np.array(exportData), np.array(exportLabels)


generator = GameStateGenerator(model)

model.fit_generator(generator=generator, epochs=1, workers=0, max_queue_size=1, verbose=1, shuffle=False)
N = 10
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
plt.ylim(top=100)
plt.ylim(bottom=0)
plt.ylabel('win rate against base line ai in %')
plt.xlabel('iteration (in 10)')
plt.plot(x, moving_aves_won, color='green', label='win rate')
plt.plot(x, moving_aves_stale, color='blue', label='stalemate rate')
plt.plot(x, moving_aves_stale_and_won, color='red', label='combined win and stalemate rate')
plt.legend(loc='upper left')
plt.show()
model.save_weights('testmodel')