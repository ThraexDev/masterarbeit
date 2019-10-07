import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.tanh, input_shape=(18,)),
    tf.keras.layers.Dense(27, activation=tf.nn.tanh),
    tf.keras.layers.Dense(9, activation=tf.nn.tanh)
])

model.compile(optimizer='adam',
              loss=tf.compat.v2.losses.mean_squared_error,
              metrics=['accuracy'])

model.load_weights('testmodel')


def calculateGameState(playerState1, playerState2):
    if (playerState1[0] == 1 and playerState1[1] and playerState1[2] == 1) or (
            playerState1[3] == 1 and playerState1[4] and playerState1[5] == 1) or (
            playerState1[6] == 1 and playerState1[7] and playerState1[8] == 1) or (
            playerState1[0] == 1 and playerState1[3] and playerState1[6] == 1) or (
            playerState1[1] == 1 and playerState1[4] and playerState1[7] == 1) or (
            playerState1[2] == 1 and playerState1[5] and playerState1[8] == 1) or (
            playerState1[0] == 1 and playerState1[4] and playerState1[8] == 1) or (
            playerState1[2] == 1 and playerState1[4] and playerState1[6] == 1):
        return 'player1won'
    if sum(playerState1) + sum(playerState2) == 9:
        return 'stalemate'
    return 'pass'


def checkViolation(playerState1, playerState2, action):
    if playerState1[action] == 1 or playerState2[action] == 1:
        return True
    return False


playerState = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
currentPlayer = 0
gameResult = calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
while gameResult == 'pass':
    gameState = playerState[currentPlayer % 2] + playerState[(currentPlayer + 1) % 2]
    action = 0
    if currentPlayer % 2 == 1:
        prediction = model.predict_on_batch(np.array([gameState])).numpy()[0]
        while checkViolation(playerState[0], playerState[1], np.argmax(prediction)):
            prediction[np.argmax(prediction)] = -1
        action = np.argmax(prediction)
        print(action)
    else:
        print(gameState)
        action = int(input("Feld eingeben: "))
    playerState[currentPlayer % 2][action] = 1
    gameResult = calculateGameState(playerState[currentPlayer % 2], playerState[(currentPlayer + 1) % 2])
    if gameResult == 'player1won':
        print('player1won')
        print(action)
    if gameResult == 'stalemate':
        print('stalemate')
    currentPlayer = currentPlayer + 1