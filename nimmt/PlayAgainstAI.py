import numpy as np
import tensorflow as tf

from nimmt.Board import Board
from nimmt.Player import Player

input_layer = tf.keras.Input(shape=(27,), name='input')
allowed_moves = tf.keras.Input(shape=(9,), name='allow')
big_layer = tf.keras.layers.Dense(100, activation=tf.nn.tanh, name='big')(input_layer)
middle = tf.keras.layers.Dense(27, activation=tf.nn.tanh, name='middle')(big_layer)
moveprobability = tf.keras.layers.Dense(9, activation=tf.nn.sigmoid, name='moveprobability')(middle)
winprobability = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='winprobability')(middle)
allowedcardprobability = tf.keras.layers.Multiply(name='finalmoveprobability')([allowed_moves, moveprobability])

model = tf.keras.Model(inputs=[input_layer, allowed_moves],
                            outputs=[allowedcardprobability, winprobability])
model.compile(optimizer='Adam',
                   loss=[tf.keras.losses.categorical_crossentropy, tf.compat.v2.losses.mean_squared_error],
                   metrics=['accuracy'])
model.load_weights("model6800")

ai_player = Player(model, 0)
board = Board()
game_not_finished = True
ai_turn = True
game_feedback = 0.0

while game_not_finished:
    if ai_turn:
        prob, input_ai, allow = ai_player.calculate_turn(board)
        print(prob)
        print(input_ai)
        print(allow)
        move = np.argmax(prob)
        game_feedback, game_not_finished = board.add_move(0, move)
    else:
        print(board.get_allow())
        move = int(input('move: '))
        game_feedback, game_not_finished = board.add_move(1, move)
    ai_turn = not ai_turn

if game_feedback == 1:
    print('current player won')
