import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(18, activation=tf.nn.tanh),
  tf.keras.layers.Dense(200, activation=tf.nn.tanh),
  tf.keras.layers.Dense(100, activation=tf.nn.tanh),
  tf.keras.layers.Dense(50, activation=tf.nn.tanh),
  tf.keras.layers.Dense(9, activation=tf.nn.tanh)
])

model.compile(optimizer='adam',
              loss=tf.compat.v2.losses.mean_squared_error,
              metrics=['accuracy'])

history = []
correctAnswer = [0,0,0,0,0,0,0,0,0,0]


class GameStateGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, tfmodel):
        self.model = tfmodel
        self.validationData = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
        self.validationTarget = [2, 5, 8, 6, 7, 2, 4, 6, 0, 3]

    def __len__(self):
        return 10000

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
        if item % 100 == 0:
            correctpredictions = 0
            for i in range(0, len(self.validationData)):
                prediction = self.model.predict_on_batch(np.array([self.validationData[i]])).numpy()[0]
                while self.checkViolation(self.validationData[i][0:9], self.validationData[i][9:18],
                                          np.argmax(prediction)):
                    prediction[np.argmax(prediction)] = -1
                if np.argmax(prediction) == self.validationTarget[i]:
                    correctpredictions = correctpredictions + 1
                    correctAnswer[i] = correctAnswer[i] + 1
            history.append((correctpredictions / len(self.validationData)) * 100)
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
            newLabel = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(0, len(newLabel)):
                if self.checkViolation(playerState[0], playerState[1], i):
                    newLabel[i] = -1
            newLabel[action] = 1
            labels[currentPlayer % 2].append(newLabel)
            playerState[currentPlayer % 2][action] = 1
            gameResult = self.calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
            if gameResult == 'player1won':
                for i in range(0,len(labels[(currentPlayer+1) % 2])):
                    for j in range(0, len(labels[(currentPlayer+1) % 2][i])):
                        if labels[(currentPlayer+1) % 2][i][j] == 1:
                            labels[(currentPlayer + 1) % 2][i][j] = -1
                        if labels[(currentPlayer+1) % 2][i][j] == 0:
                            labels[(currentPlayer + 1) % 2][i][j] = 1
                print('Player '+str( (currentPlayer+1) % 2) + ' lost ')
            if gameResult == 'stalemate':
                for i in range(0,len(labels[(currentPlayer) % 2])):
                    for j in range(0, len(labels[(currentPlayer) % 2][i])):
                        if labels[(currentPlayer) % 2][i][j] == 1:
                            labels[(currentPlayer) % 2][i][j] = -1
                        if labels[(currentPlayer) % 2][i][j] == 0:
                            labels[(currentPlayer) % 2][i][j] = 1
                print('stalemate')
            currentPlayer = currentPlayer + 1
        exportData = data[0]
        exportData.extend(data[1])
        exportLabels = labels[0]
        exportLabels.extend(labels[1])
        return np.array(exportData), np.array(exportLabels)


generator = GameStateGenerator(model)

model.fit_generator(generator=generator, epochs=1, workers=0)
x = list(range(0,len(history)))
plt.plot(x, history, 'bo')
plt.show()
model.save_weights('testmodel')
print(correctAnswer)